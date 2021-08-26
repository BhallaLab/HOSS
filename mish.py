# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 

'''
*******************************************************************
 * File:            mish.py
 * Description:
 * Author:          Upinder S. Bhalla
 * E-mail:          bhalla@ncbs.res.in
 ********************************************************************/

/**********************************************************************
** This program uses HILLTAU and MOOSE and optimizes parameters of the
** MOOSE model to fit the HILLTAU one.
**           copyright (C) 2021 Upinder S. Bhalla. and NCBS
**********************************************************************/
'''
from __future__ import print_function
import datetime
import getpass
import sys
import os
from scipy.optimize import minimize
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import moose
import moose.model_utils as mu
import hillTau

t1 = 20
t2 = 60
t3 = 100
i1 = 1e-3

numEval = 0
numIter = 0
plotDt = 1
stimVec = [[0, 0.0, 20.0], [0, 1e-3, 40.0], [0, 0, 40.0]]
stimRange = [ 0.1, 0.2, 0.5, 1, 2.0, 5.0, 10.0 ]
settleTimeScale = stimRange[-1]  # How much longer is settleTime than midTime?

class Stim:
    ### Advance to specified time, and then set the conc to the stim value.
    def __init__( self, mol, conc, time ):
        self.mooseMol = getMooseName( mol )
        self.hillTauMol = getHillTauName( mol )
        self.conc = conc
        self.time = time
        self.molIndex = 0

class Mish:
    def __init__( self, chem, reference, args, stimVec ):
        self.chemModelName = chem
        self.reference = reference
        #self.plotnum = { i:model.molInfo[ i ].index for i in htNames }
        self.stimVec = stimVec
        self.numIter = 0
        self.simt = 0
        self.molMap = { getHillTauName(i):getMooseName(i) for i in args.monitor }
        self.pathMap = {} # Looks up full mol path from mol name.

        # Load the moose model
        filename, file_extension = os.path.splitext(chem)
        if file_extension == ".g":
            self.modelId = moose.loadModel( chem, 'model', 'gsl' )
        elif file_extension == ".xml":
            self.modelId = moose.readSBML( chem, 'model', 'gsl' )

        for i in args.monitor:
            mooseName = getMooseName( i )
            el = moose.wildcardFind( "/model/kinetics/" + mooseName + ",/model/kinetics/##/" + mooseName )
            if len( el ) == 0:
                raise( ValueError( "Output molecule '{}' not found".format(mooseName) ) )
            self.pathMap[mooseName] = el[0].path
        for i in stimVec:
            el = moose.wildcardFind( "/model/kinetics/" + i.mooseMol + ",/model/kinetics/##/" + i.mooseMol )
            if len( el ) == 0:
                raise( ValueError( "Stim molecule '{}' not found".format(i.mooseMol )) )
            self.pathMap[i.mooseMol] = el[0].path

        tabs = moose.Neutral( "/model/tabs" )
        for i in args.monitor:
            mooseName = getMooseName( i )
            el = moose.wildcardFind( "/model/kinetics/" + mooseName + ",/model/kinetics/##/" + mooseName )
            if len( el ) > 0:
                # Make an output table
                tab = moose.Table2( "/model/tabs/" + mooseName )
                moose.connect( tab, "requestOut", el[0], "getConc" )
                #print( "Making output for {} on {}".format( el[0].path, tab.path ) )
            else:
                raise( ValueError( "Error: Molecule '{}' not found in moose model: {}".format( mooseName, chem ) ) )
                
        for i in range( 10, 20 ):
            moose.setClock( i, plotDt )
        moose.reinit()
        self.params = self.extractParams(args.addParams, args.removeParams)

    def findMooseObjectsOnTree( self, objNameList ):
        ### Return dict of full paths with key as the object name.
        ret = {}
        for i in objNameList:
            [j, k] = i.rsplit( '.', 1 )
            el = moose.wildcardFind( "/model/kinetics/" + j + ",/model/kinetics/##/" + j )
            if len( el ) > 0:
                ret[ i ] = el[0].path + '.' + k
        return ret


    def extractParams( self, add, remove ):
        if len( add ) > 0:
            pv = [ v for v in self.findMooseObjectsOnTree( add ).values()]
        else:
            pv = []
            el = moose.wildcardFind( "/model/kinetics/##[ISA=PoolBase]" )
            for i in el:
                pv.append( i.path + ".concInit" )
            el = moose.wildcardFind( "/model/kinetics/##[ISA=Reac]" )
            for i in el:
                pv.append( i.path + ".tau" )
                pv.append( i.path + ".Kd" )
            el = moose.wildcardFind( "/model/kinetics/##[ISA=EnzBase]" )
            for i in el:
                pv.append( i.path + ".kcat" )
                pv.append( i.path + ".Km" )
        rem = [ v for v in self.findMooseObjectsOnTree( remove ).values()]
        for i in rem:
            if i in pv:
                pv.remove( i )
        return pv

    def scaleParams( self, x ):
        for i, scaleFactor in zip( self.params, x ):
            spl = i.rsplit( '.' ,1)
            objPath, field = spl
            obj = moose.element( objPath )
            if field == "concInit" or field == "conc":
                obj.concInit *= scaleFactor
            elif field == "tau":
                obj.Kf /= scaleFactor
                obj.Kb /= scaleFactor
            elif field == "Kd":
                sk = np.sqrt( scaleFactor )
                obj.Kf /= sk
                obj.Kb *= sk
            elif field == "Km":
                obj.Km *= scaleFactor
            elif field == "kcat":
                obj.kcat *= scaleFactor

    def doRun(self, x ):
        self.scaleParams( x )
        moose.reinit()
        lastt = 0.0
        for stim in self.stimVec:
            #print( "STIM = ", stim.mooseMol, "   ", stim.conc, " ", stim.time )
            objName = self.pathMap.get( stim.mooseMol )
            if objName:
                if stim.time > lastt:
                    t0 = time.time()
                    moose.start( stim.time - lastt )
                    self.simt += time.time() - t0
                    lastt = stim.time
                obj = moose.element( objName )
                obj.concInit = stim.conc #assign conc even if no sim advance
                #print( "STIM: {} = {}".format( obj.path, obj.concInit ) )
            else:
                print( "Warning: Stimulus molecule '{}' not found in MOOSE".format( stim.mooseMol ) )
        #nt = np.transpose( np.array( self.model.plotvec ) )
        #ret = { name:nt[index] for name, index in self.plotnum.items() }
        vecs = { i.name:i.vector for i in moose.wildcardFind("/model/tabs/#") }
        #print( moose.element( '/clock' ).currentTime )
        #print( moose.element( '/clock' ).runTime )
        self.scaleParams( 1.0/x )
        self.numIter += 1
        return vecs

    def doScore( self, outDict ):
        sq = 0.0
        for htname, ref in self.reference.items():
            yrange = max( ref )
            expt = outDict[self.molMap[htname]]
            dl = len( expt) - len(ref )
            if abs( dl ) > 2:
                raise ValueError( "Output vec lengths differ too much, {} vs {}".format( len( expt ), len(ref ) ) )
            if dl > 0:
                expt = expt[:-dl]
            elif dl < 0:
                ref = ref[:-dl]

            y = ( expt - ref ) / yrange
            sq += np.dot( y, y ) / len( ref )
        return np.sqrt( sq )

    def doEval( self, x ):
        ret = self.doRun( x )
        dotter()
        return self.doScore( ret )

    def dumpScaledFile( self, x, fname ):
        self.scaleParams( x )
        print( "Saving optimized model to: ", fname )
        filename, file_extension = os.path.splitext(fname)
        if file_extension == ".g":
            moose.writeKkit( "/model", fname )
        elif file_extension == ".xml":
            moose.writeSBML( "/model", fname )

# Callback function for minimizer. Just prints out dots.
def dotter():
    global numEval
    numEval += 1
    if (numEval % 50) == 0:
        print( ". ", numEval, flush = True )
    else:
        print( ".", end = "", flush = True )

def iterPrint( xk ):
    global numIter
    numIter += 1
    print( " Iter = ", numIter, flush = True )


def makeMish( args, stimVec, referenceOutputs ):
    return Mish( referenceOutputs, pv, args.monitor, stimVec )

def plotBoilerplate( xlabel = 'Time (s)', ylabel = 'Conc ($\mu$M)', title = "" ):
    ax = plt.subplot( 1, 1, 1 )
    ax.spines['top'].set_visible( False )
    ax.spines['right'].set_visible( False )
    ax.set_xlabel( xlabel, fontsize = 14 )
    ax.set_ylabel( ylabel, fontsize = 14 )
    ax.set_title( title )
    return ax

def runHillTau( ht, stimVec, outMols ):
    jsonDict = hillTau.loadHillTau( ht )
    hillTau.scaleDict( jsonDict, hillTau.getQuantityScale( jsonDict ) )
    model = hillTau.parseModel( jsonDict )
    model.dt = plotDt
    for i in stimVec:
        mi = model.molInfo.get( i.hillTauMol )
        if mi:
            inputMolIndex = mi.index
            i.molIndex = inputMolIndex
            if i.conc < 0:  # Hack to specify use of initial conc
                i.conc = mi.concInit
        else:
            raise ValueError( "Nonexistent stimulus molecule: ", i.hillTauMol )
    outMolIndex = {}
    for i in outMols:
        mi = model.molInfo.get( i )
        if mi:
            outMolIndex[i] = mi.index
        else:
            raise ValueError( "Nonexistent output molecule: ", i )
    model.reinit()
    lastt = 0.0
    for stim in stimVec:
        model.advance( stim.time - lastt )
        model.conc[ stim.molIndex ] = stim.conc
        lastt = stim.time
    #nt = np.transpose( np.array( self.model.plotvec ) )
    #ret = { name:nt[index] for name, index in self.plotnum.items() }
    ret = { name:np.array(model.getConcVec( index )) for name, index in outMolIndex.items() }
    return ret

def runMoose( chem, stimVec, outMols ):
    filename, file_extension = os.path.splitext(chem)
    if file_extension == ".g":
        modelId = moose.loadModel( chem, 'model', 'gsl' )
    elif file_extension == ".xml":
        #modelId = mu.mooseReadSBML( chem, 'model', 'gsl' )
        modelId = moose.readSBML( chem, 'model', 'gsl' )
    '''
    moose.le( "/model/kinetics" )
    for i in moose.wildcardFind ( "/model/kinetics/##[ISA=PoolBase]" ):
        print( i.name, i.concInit )
    for i in moose.wildcardFind ( "/model/kinetics/##[ISA=Reac]" ):
        print( i.name, i.Kf, i.Kb )
    '''
    tabs = moose.Neutral( "/model/tabs" )
    mooseMols = [ getMooseName( i ) for i in outMols ]
    for i in mooseMols:
        el = moose.wildcardFind( "/model/kinetics/" + i + ",/model/kinetics/##/" + i )
        if len( el ) > 0:
            # Make an output table
            tab = moose.Table2( "/model/tabs/" + i )
            moose.connect( tab, "requestOut", el[0], "getConc" )
    for i in range( 10, 20 ):
        moose.setClock( i, plotDt )

    moose.reinit()
    lastt = 0.0

    for stim in stimVec:
        #print( "STIM = ", stim.mol, "   ", stim.conc, " ", stim.time )
        el = moose.wildcardFind( "/model/kinetics/" + stim.mooseMol + ",/model/kinetics/##/" + stim.mooseMol )
        if len( el ) > 0:
            if stim.time > lastt:
                moose.start( stim.time - lastt )
                lastt = stim.time
            el[0].concInit = stim.conc # assign conc even if no sim advance
        else:
            print( "Warning: Stimulus molecule '{}' not found in MOOSE".format( stim.mooseMol ) )

    vecs = { i.name:i.vector for i in moose.wildcardFind("/model/tabs/#") }
    return vecs

def paramVec( jsonDict ):
    pv = []
    for grp in jsonDict['Groups'].values():
        reacDict = grp.get( 'Reacs' )
        if not reacDict:
            continue
        for reacname, reac in reacDict.items():
            pv.append( reacname + ".KA" )
            pv.append( reacname + ".tau")
            tau2 = reac.get( "tau2" )
            if tau2:
                pv.append( reacname + ".tau2")
            gain = reac.get( "gain" )
            if gain:
                pv.append( reacname + ".gain")
            baseline = reac.get( "baseline" )
            if baseline:
                pv.append( reacname + ".baseline")
            Kmod = reac.get( "Kmod" )
            if Kmod:
                pv.append( reacname + ".Kmod")
            Amod = reac.get( "Amod" )
            if Amod:
                pv.append( reacname + ".Amod")

        if 'Species' in grp:
            for molname, mol in grp['Species'].items():
                pv.append( molname + ".concInit")

    return pv

def parseDoser( stimVec, d, t ):
    assert( len(d) == 3 )
    mol, midconc, settleTime = d
    midconc = float( midconc )
    settleTime = float( settleTime )
    #print("'{}'     '{}'     '{}'".format( mol, midconc, settleTime) )
    # Build dose=response
    stimVec.append( Stim( mol, 0.0, t ) )
    t += settleTime
    for x in stimRange: 
        stimVec.append( Stim( mol, midconc * x, t ) )
        t += settleTime
    stimVec.append( Stim( mol, 0.0, t ) ) 
    t += settleTime
    return t

def parseCycle( stimVec, c, t ):
    mol = c[0]
    conc, onTime, offTime = [ float( x ) for x in c[1:4] ]
    numCycles = int( c[4] )
    stimVec.append( Stim( mol, 0.0, t ) )
    for i in range( numCycles ):
        t += float( offTime )
        stimVec.append( Stim( mol, float( conc ), t ) )
        t += float( onTime )
        stimVec.append( Stim( mol, 0.0, t ) )
    t += float( offTime ) # final zero level stim
    stimVec.append( Stim( mol, 0.0, t ) )
    return t

def parseStims( stimArg, builtin, cyclic, doser ):
    stimVec = []
    t = 0.0
    for b in builtin:
        assert( len(b) == 3 ) # molecule, midconc, midTime
        mol, midconc, midTime = b
        midconc = float( midconc )
        midTime = float( midTime )
        #print("'{}'     '{}'        '{}'".format( mol, midconc, midTime) )
        settleTime = midTime * settleTimeScale
        # Build dose=response
        t = parseDoser( stimVec, [mol, midconc, settleTime], t)
        # Build cyclic stimulus
        sr0 = stimRange[0]
        t = parseCycle( stimVec, [mol, midconc, midTime*sr0, midTime*sr0, len(stimRange)*25 ], t)
        t = parseCycle( stimVec, [mol, midconc, midTime, midTime, int( len(stimRange) * 2.5 ) ], t)

    for c in cyclic:
        assert( len(c) == 5 ) # molecule, conc, start, stop, numCycles
        t = parseCycle( stimVec, c, t )

    for d in doser:
        assert( len(d) == 3 ) # molecule, midconc, settleTime
        t = parseDoser( stimVec, d, t )

    for s in stimArg:
        assert( len( s ) >= 3 and len(s) % 2 == 1 )
        for i in range( 1, len( s ), 2 ):
            stimVec.append( Stim( s[0], float( s[i] ), float(s[i+1]) ) )
    return sorted( stimVec, key = lambda x: x.time )

def oldparseStims( stimArg, builtin, cyclic, doser ):
    stimVec = []
    t = 0.0
    for b in builtin:
        # We do the dose-response, then high-freq, then mid-freq.
        # The dose response internally also does low-freq.
        assert( len(b) == 3 ) # molecule, midconc, midtime
        mol, midconc, midtime = b
        midconc = float( midconc )
        midtime = float( midtime )
        #print("'{}'     '{}'        '{}'".format( mol, midconc, midtime) )
        settleTime = midtime * settleTimeScale
        # Build dose=response
        stimVec.append( Stim( mol, 0.0, t ) )
        t += settleTime
        for x in stimRange: 
            stimVec.append( Stim( mol, midconc * x, t ) )
            t += settleTime
        # Use -ve conc to tell it to look up initial conc.
        stimVec.append( Stim( mol, -1.0, t ) ) 
        t += settleTime

        # duration, hence optimization weight, of each stim should match.
        for x in range( int( 0.25 * settleTimeScale * len( stimRange ) / stimRange[0] ) ):
            t += midtime * stimRange[0]
            stimVec.append( Stim( mol, midconc, t ) )
            t += midtime * stimRange[0]
            stimVec.append( Stim( mol, 0, t ) )

        for x in range( int( 0.25 * settleTimeScale * len( stimRange ) ) ):
            t += midtime
            stimVec.append( Stim( mol, midconc, t ) )
            t += midtime
            stimVec.append( Stim( mol, 0, t ) )
        t += midtime
        # Use -ve conc to tell it to look up initial conc.
        stimVec.append( Stim( mol, -1.0, t ) )

    for c in cyclic:
        for i in range( c[4] ):
            t += float( c[2] )
            stimVec.append( Stim( c[0], float( c[1] ), t ) )
            t += float( c[3] )
            stimVec.append( Stim( c[0], 0, t ) )
        t += float( c[2] ) # final zero level stim
        stimVec.append( Stim( c[0], float( c[1] ), t ) )

    for s in stimArg:
        assert( len( s ) >= 3 and len(s) % 2 == 1 )
        for i in range( 1, len( s ), 2 ):
            stimVec.append( Stim( s[0], float( s[i] ), float(s[i+1]) ) )
    return sorted( stimVec, key = lambda x: x.time )

def getMooseName( name ):
    sp = name.split( ':' )
    return sp[0]

def getHillTauName( name ):
    return name.split( ':' )[-1]

def runMishOptimization( mish, args, t1, t0 ):
    initParams = np.ones( len( mish.params ) )
    initRet = mish.doRun( initParams )
    if args.checkInit:
        print( "Initial MOOSE run took {:.3f} seconds".format( time.time() - t0 ) )
        print( "Number of Parameters= {}".format( len( mish.params )) )
        if len( mish.params ) > 15:
            print( "More than 15 Parameters is a Bad Idea. Suggest you remove some." )
        for i in mish.params:
            k = i.rsplit( "/" )[-1]
            print( "{:25s}".format( k ) )
        return initRet, initRet

    bounds = [(0.01, 100.0)] * len( mish.params )
    x0 = initParams

    ret = minimize( mish.doEval, x0, method = "L-BFGS-B", tol = args.tolerance, bounds = bounds, callback = iterPrint )

    finalRet = mish.doRun( ret.x )
    print( "\n{:25s}  {}".format( "Object.field", "Scale factor" ) )
    for i, j in zip( mish.params, ret.x ):
        k = i.rsplit( "/" )[-1]
        print( "{:25s}  {:4f}".format( k, j ) )

    print( "Timings: reference= {:.4f}s, optimization= {:.2f}s, MOOSE Cumulative = {:.2f}s \nNumber of evaluations = {}, number of optimization iterations = {}, \nInitial score = {:3g}, Final score = {:3g}".format( t1 - t0, time.time() - t1, mish.simt, mish.numIter, ret.nit,  mish.doScore( initRet ), ret.fun ) )

    if len( args.optfile ) > 0:
        mish.dumpScaledFile( ret.x, args.optfile )

    return initRet, finalRet

def generateStimEntries( stimVec ):
    ret = []
    assert( len( stimVec ) > 0 )
    # Split stimVec into subsets, one for each molecule.
    stimMolDict = {}
    for i in stimVec:
        if i.mooseMol in stimMolDict:
            stimMolDict[i.mooseMol].append(i)
        else:
            stimMolDict[i.mooseMol] = [i]

    for name, val in stimMolDict.items():
        ret.append( {"timeUnits": "sec", "quantityUnits": "mM", "entity": name, "field": "conc", "data": generateStimData(val) } )
    return ret

def generateStimData( stimVec ):
    ret = []
    for i in stimVec:
        ret.append( [i.time, i.conc] )
    return ret

def generateReadoutData( plotDt, refOutput, stimVec ):
    temp = []
    ret = []
    for i in stimVec:
        temp.append( i.time )
    st = np.concatenate( ( temp, np.arange( 0.0, plotDt * len(refOutput), plotDt * 100 ) ) )
    times = np.unique( np.round(st, 2) )
    indices = np.round( times/plotDt ).astype( int )
    if indices[-1] >= len( refOutput ):
        times = times[:-1]
        indices = indices[:-1]
    ret = [ [t,  refOutput[i], 0] for t, i in zip( times, indices )]
    return ret

def generateExperiment( args, stimVec, refOutput ):
    transcriber = getpass.getuser()
    mooseMolName = getMooseName( args.monitor[0] )
    htMolName = getHillTauName( args.monitor[0] )
    assert( htMolName in refOutput )
    jsonDict = { 
            "FileType": "FindSim",
            "Version": "1.0",
            "Metadata": 
            {"transcriber": transcriber, "organization": "OpenSource", 
                "source": {"sourceType": "other", "doi": "dummy", "year":datetime.datetime.now().year}
            },
            "Experiment": { "design": "TimeSeries", "species":"", "cellType": "", "notes": "Generated from mish.py" },
            "Stimuli": generateStimEntries( stimVec ),
            "Readouts": { "timeUnits": "sec", "quantityUnits":"uM",
                "entities": mooseMolName,
                "field": "conc",
                "data": generateReadoutData( plotDt, refOutput[htMolName], stimVec )
                }
            }
    with open( args.generateExperiment, "w" ) as fd:
        json.dump( jsonDict, fd, indent = 4 )
            

def main():
    global plotDt
    stimDict = {}
    parser = argparse.ArgumentParser( description = "Optimizes chemical kinetic (mass action and Michaelis-Menten) models of chemical signalling to fit a HillTau model." )
    parser.add_argument( "chemModel", type = str, help = "Required: Filepath for chemical kinetic model" )
    parser.add_argument( "HillTauModel", type=str, help = "Required: Filepath for HillTau model" )
    parser.add_argument( "-m", "--monitor", type = str, nargs = '+', metavar = "molName", help = "Optional: Molecules to monitor, as a list of space-separated names. If names differ between chemical and HillTau models, both can be specified, separated by a colon. Example: Ca:Calcium.", default = ["output"] )
    parser.add_argument( '-b', '--builtin', nargs = 3, metavar = ('molecule', 'midconc', 'midtime'), action='append', help='Optional: Deliver builtin stimulus. This is a dose-response centered around midconc, with a settling time of midtime * 10. This is followed by a timeseries of square-wave pulses from 0 to midconc, with on-time of midtime/10 followed by another with on-time of midtime. The first runs for 170 cycles and the second for 17. If multiple builtin stimuli are specified, they will be executed in order, without overlap. If molecule names are different between chem and HillTau models, they should be separated by a colon.', default = [] )
    parser.add_argument( '-c', '--cyclic', nargs = 5, metavar = ('molecule', 'conc', 'onTime', 'offTime', 'num_cycles'), action='append', help='Optional: Deliver cyclic stimulus. This is a timeseries of rectangular pulses from 0 to conc, with an onTime and offTime as specified, repeated for num_cycles. Before the first cycle, and after the last cycle it runs for another "offTime" seconds at conc = 0. If molecule names are different between chem and HillTau models, they should be separated by a colon.', default = [] )
    parser.add_argument( '-d', '--dose_response', nargs = 3, metavar = ('molecule', 'midconc', 'settle_time'), action='append', help='Optional: Deliver dose-response stimulus centered around midconc, with a settling time of settle_time. If other builtin, cyclic or dose_response stimuli are specified, they will be executed in order, without overlap. If molecule names are different between chem and HillTau models, they should be separated by a colon.', default = [] )
    parser.add_argument( "-a", "--addParams", type = str, nargs = "+", metavar = "obj.field", help = "Optional: Add parameter list. This will remove all the automatic ones obtained by scanning through the model, and only use the added ones from this list. Each parameter is of the form object.field. Any number of parameters can be added, separated by spaces. If molecule names are different between chem and HillTau models, they should be separated by a colon", default = [] ) 
    parser.add_argument( "-r", "--removeParams", type = str, nargs = "+", metavar = "param", help= "Optional: Remove parameters from the default ones which were generated automatically by scanning all the reactions in the model. Each parameter is of the form object.field. Any number of parameters can be specified, separated by spaces.", default = [] )
    parser.add_argument( '-s', '--stimulus', type = str, nargs = '+', metavar = 'args', action='append', help='Optional: Deliver stimulus as follows: --stimulus molecule conc time [conc time]... Each stimulus molecule may be followed by one or more [conc time] pairs. Any number of stimuli may be given, each indicated by --stimulus. Stimuli can overlap with the builtin stimuli, the values will apply from the time they are given till the builtin protocol delivers its own stimulus to override them.', default = [] )
    parser.add_argument( '-ci', '--checkInit', action='store_true', help='Flag: when set, only do the initial run for the MOOSE model, to see if the stimuli are doing what we expect.')
    parser.add_argument( '-p', '--plot', action='store_true', help='Flag: when set, it plots the chem output, the original HillTau output, and the optimized HillTau output')
    parser.add_argument( "-t", "--tolerance", type = float, help = "Optional: tolerance for convergence of optimization.", default = 1.0e-6 )
    parser.add_argument( '-o', '--optfile', type = str, help='Optional: File name for saving optimized SBML model. If not set, no file is saved.', default = "" )
    parser.add_argument( '-g', '--generateExperiment', type = str, help='Optional: Generate a FindSim experiment file with the specified stimulus protocol and simulated HillTau output as the target values.', default = "", metavar = "experiment_file_name" )
    args = parser.parse_args()

    if len( args.builtin ) > 0:
        plotDt = min( plotDt, float( args.builtin[0][2] ) * stimRange[0] * 0.2 )
    stimVec = parseStims( args.stimulus, args.builtin, args.cyclic, args.dose_response )
    t0 = time.time()

    htMonitor = [ getHillTauName(h) for h in args.monitor ]
    referenceOutputs = runHillTau( args.HillTauModel, stimVec, htMonitor )
    t1 = time.time()
    print( "Completed reference run of '{}' in {:.5f}s".format( args.HillTauModel, t1 -t0 ) )

    if args.generateExperiment:
        generateExperiment( args, stimVec, referenceOutputs )
        quit()

    mish = Mish( args.chemModel, referenceOutputs, args, stimVec )
    initRet, finalRet = runMishOptimization( mish, args, t1, t0 )

    if args.plot:
        for htname, ref in referenceOutputs.items():
            mooseName = mish.molMap[ htname ]
            fig = plt.figure( figsize = (6,6), facecolor='white' )
            ax = plotBoilerplate( xlabel = "Time (s)", title = mooseName )
            x = np.array( range( len( ref ) ) ) * plotDt
            ax.plot( x , 1000.0 * ref, label = "HillTau" )
            x = np.array( range( len( initRet[mooseName] ) ) ) * plotDt
            ax.plot( x, 1000.0 * initRet[mooseName], label = "MOOSE orig" )
            ax.plot( x, 1000.0 * np.array( finalRet[mooseName] ), label = "MOOSE opt" )
            ax.legend()
        plt.show()

if __name__ == '__main__':
    main()
