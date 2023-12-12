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
 * File:            exptSynth.py
 * Description:     Model Optimizer Using Synthetic signaling Experiments
 *                  Previously known as mouse.py
 * Author:          Upinder S. Bhalla
 * E-mail:          bhalla@ncbs.res.in
 ********************************************************************/

/**********************************************************************
** This program generates synthetic experiments in FindSim format, 
** optionally using a HillTau model to predict outcome of the experiment.
**           copyright (C) 2023 Upinder S. Bhalla. and NCBS
**********************************************************************/
'''
import datetime
import getpass
import sys
import os
from scipy.optimize import minimize
import json
import time
import argparse
import numpy as np
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
stimRange = [ 0.05, 0.1, 0.2, 0.4, 0.8, 1, 1.25, 2.5, 5.0, 10.0, 20.0 ]
settleTimeScale = stimRange[-1]  # How much longer is settleTime than midTime?

class Stim:
    ### Advance to specified time, and then set the conc to the stim value.
    def __init__( self, mol, conc, time, doSettle = False ):
        self.molName = mol
        self.mooseMol = getMooseName( mol )
        self.hillTauMol = getHillTauName( mol )
        self.conc = conc
        self.time = time
        self.doSettle = doSettle
        self.molIndex = 0


def runHillTau( model, stimVec, outMols, bufList ):

    for i in stimVec:
        mi = model.molInfo.get( i.hillTauMol )
        if mi:
            inputMolIndex = mi.index
            i.molIndex = inputMolIndex
            if i.conc < 0:  # Hack to specify use of initial conc
                i.conc = mi.concInit
        else:
            raise ValueError( "Nonexistent stimulus molecule: ", i.hillTauMol )
        si = model.reacInfo.get( i.hillTauMol )
        if si:
            si.isBuffered = 1

    outMolIndex = {}
    ret = {}
    for i in outMols:
        mi = model.molInfo.get( i )
        if mi:
            outMolIndex[i] = mi.index
            ret[i] = []
        else:
            raise ValueError( "Nonexistent output molecule: ", i )
    model.reinit()
    for ii in range( len( bufList ) // 2) :
        bb = bufList[2*ii+1]
        ri = model.reacInfo.get( bb["entity"] )
        if ri:
            ri.isBuffered = 1
            model.conc[ ri.prdIndex ] = bb["value"]
            #print( "buffering {} {} to {}".format( bb["entity"], ri.prdIndex,  bb["value"] ))
        else:
            print( "Error: buffer entity {} not found".format( bb["entity"] ) )
            quit()
    lastt = 0.0
    mi = model.molInfo[outMols[0]]
    for stim in stimVec:
        model.conc[ stim.molIndex ] = stim.conc
        model.concInit[ stim.molIndex ] = stim.conc
        model.advance( stim.time - lastt, stim.doSettle )
        #print( "ADVANCE ", stim.time - lastt, stim.conc, stim.doSettle )
        lastt = stim.time
        #print( "[{}].conc = {}".format( mi.name, model.conc[mi.index] ) )
        if stim.doSettle:   # Handling dose-response.
            for key, value in outMolIndex.items():
                ret[key].append( model.conc[value] )
    #nt = np.transpose( np.array( self.model.plotvec ) )
    #ret = { name:nt[index] for name, index in self.plotnum.items() }
    # OK< getting the conc vec is failing.
    if not stim.doSettle:
        ret = { name:np.array(model.getConcVec( index )) for name, index in outMolIndex.items() }

    # Unbuffer the reacs
    for i in stimVec:
        si = model.reacInfo.get( i.hillTauMol )
        if si:
            si.isBuffered = 0

    for ii in range( len( bufList ) // 2) :
        bb = bufList[2*ii+1]
        ri = model.reacInfo.get( bb["entity"] )
        if ri:
            ri.isBuffered = 0

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
        stimVec.append( Stim( mol, midconc * x, t, doSettle = True ) )
        t += settleTime
    stimVec.append( Stim( mol, 0.0, t ) ) 
    t += settleTime
    return t


def getMooseName( name ):
    sp = name.split( ':' )
    return sp[0]

def getHillTauName( name ):
    return name.split( ':' )[-1]

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
        ret.append( {"timeUnits": "sec", "quantityUnits": "uM", "entity": name, "field": "conc", "data": generateStimData(val) } )
    return ret

def estimateStimConc( htmodel, stimMol ):
    ret = -1.0
    mi = htmodel.molInfo.get( stimMol )
    stimMolIdx = mi.index
    if mi:
        ret = mi.concInit
        if ret == 0:
            # Try basename if it starts with an 'a' or a 'p'
            tryname = stimMol[1:]
            mi = htmodel.molInfo.get( tryname )
            if mi:
                ret = mi.concInit
    else:
        raise ValueError( "Nonexistent stimulus molecule: ", stimMol )

    for name, rr in htmodel.reacInfo.items():
        # Order of subs is [Reag, [modifier], ligand], or [Reag, [Reag...]]
        # In either a ligand or a single reag case, we use KA.
        ligandIndex = htmodel.molInfo[ rr.subs[-1] ].index
        if ligandIndex == stimMolIdx:
            if len( rr.subs ) == 1: # For single reagent, KA has no units
                if ret <= 0: # Try to use conc of input molecule, else 1 uM.
                    ret = 1.0e-3
            else:
                ret = max( ret, rr.KA )
        elif len( rr.subs ) > 2 and rr.subs[1] == stimMol:
            # If that didn't work, we could have second arg as a modifier.
            # In this case we use Kmod
            #print( "Using Kmod for ", stimMol, ", = ", rr.Kmod, ret )
            ret = max( ret, rr.Kmod )

    if ret <= 0:
        print( "Warning, failed to find estimate for stimulus max for {}, using 1.0".format( stimMol ) )
        return 1.0e-3   #   1 uM.
    return ret * 5.0

def estimateTau( htmodel, reacStims ):
    if htmodel:
        for name, readouts in reacStims.items():
            ri = htmodel.reacInfo.get( readouts[0] )
            break
        if ri:
            return ri.tau * 2.0 # Give the reac enough time to settle.
        else:
            raise ValueError( "Nonexistent reactant: ", readoutMol )
    return 300.0


def generateStimData( stimVec ):
    # runHillTau specifies the time of the end of the stimulus, but here
    # we have to specify the time of the start. So we subtract out the
    # first time entry as they are all offset by this.
    startt = stimVec[0].time
    ret = []
    for i in stimVec:
        ret.append( [np.round( i.time - startt, decimals = 3 ), i.conc * 1000] )
    return ret

def generateReadoutData( plotDt, refVals ):
    #ret = [ [ np.round( ss.time, decimals = 3 ), rr*1000, 0.0] for ss, rr in zip( stimVec, refVals ) ]
    ret = [ [ np.round( plotDt * ii, decimals = 3 ), rr*1000, 0.0] for ii, rr in enumerate( refVals ) ]
    return ret

def generateInputBaseline( model, stimName, reacName ):
    # Assumes HT model. Returns dict of nonzero baseline mol names, vals.
    # Returns string with inputBaseline entry.
    if model == None:
        return []
    reac = model.reacInfo.get( reacName )
    assert( reac )
    # I need to check if the stimulus is one of the substrates. If not,
    # then I can't assign a buffered baseline to other inputs, since the 
    # input may come through the same other inputs.
    if not stimName in reac.subs:
        return []

    ret = []
    # The set conversion gives me unique substrates.
    for sub in [ ss for ss in set(reac.subs) if ss != stimName ]:
        val = model.reacInfo.get( sub )
        if val:
            if val.baseline > 0.0:
                ret.extend( [
                        {
                            "entity": sub,
                            "field": "isBuffered",
                            "value": 1,
                            "units": "none"
                        },
                        {
                            "entity": sub,
                            "field": "concInit",
                            "value": val.baseline,
                            "units": "mM"
                        }
                    ]
                )
    '''
    if len( ret ) > 0:
        print( "Ret = \n", ret, "\n#############################" )
    '''
    return ret

def generateBufferedMolList( buffer ):
    ret = []
    if buffer:
        if len( buffer ) % 2 != 0:
            print( "Usage: -b mol conc [mol conc] .... Mol and conc must come in pairs." )
            quit()
        for ii in range( 0, len( buffer ), 2 ):
            name = buffer[ii]
            val = float( buffer[ii+1] )
            ret.extend( [
                    {
                        "entity": name,
                        "field": "isBuffered",
                        "value": 1,
                        "units": "none"
                    },
                    {
                        "entity": name,
                        "field": "concInit",
                        "value": val,
                        "units": "mM"
                    }
                ]
            )
    return ret

def generateTimeExperiment( fname, stimVec, refMol, refVals, baselineList ):
    fname = "{}_TS_{}_vs_{}.json".format(fname, refMol, stimVec[0].molName)
    transcriber = getpass.getuser()
    jsonDict = { 
            "FileType": "FindSim",
            "Version": "1.0",
            "Metadata": 
            {"transcriber": transcriber, "organization": "OpenSource", 
                "source": {"sourceType": "other", "doi": "dummy", "year":datetime.datetime.now().year}
            },
            "Experiment": { "design": "TimeSeries", "species":"", "cellType": "", "notes": "Generated from mousse.py" },
            "Stimuli": generateStimEntries( stimVec ),
            "Readouts": { "timeUnits": "sec", "quantityUnits":"uM",
                "entities": [refMol],
                "field": "conc",
                "data": generateReadoutData( plotDt, refVals )
            },
            "Modifications": {
                "parameterChange": [
                    {
                        "entity": stimVec[0].molName,
                        "field": "isBuffered",
                        "value": 1,
                        "units": "none"
                    }
                ]
            }
        }
    jsonDict["Modifications"]["parameterChange"].extend( baselineList )
    with open( fname, "w" ) as fd:
        json.dump( jsonDict, fd, indent = 4 )

def generateDoseExperiment( fname, stimVec, refMol, refVals, settleTime, baselineList ):
    fname = "{}_DR_{}_vs_{}.json".format(fname, refMol, stimVec[0].molName)
    transcriber = getpass.getuser()
    jsonDict = { 
            "FileType": "FindSim",
            "Version": "1.0",
            "Metadata": 
            {"transcriber": transcriber, "organization": "OpenSource", 
                "source": {"sourceType": "other", "doi": "dummy", "year":datetime.datetime.now().year}
            },
            "Experiment": { "design": "DoseResponse", "species":"", "cellType": "", "notes": "Generated from mousse.py" },
            "Stimuli": [{ "timeUnits": "sec", "quantityUnits": "uM", 
                "entity": stimVec[0].molName, "field": "conc"
                #, "isBuffered": 1
            },],
            "Readouts": { "timeUnits": "sec", "quantityUnits":"uM",
                "settleTime": np.round( settleTime, decimals = 2 ),
                "entities": [refMol],
                "field": "conc",
                "data": [ [ss.conc * 1000, rr*1000, 0.0] for ss, rr in zip( stimVec, refVals ) ]
            },
            "Modifications": {
                "parameterChange": [
                    {
                        "entity": stimVec[0].molName,
                        "field": "isBuffered",
                        "value": 1,
                        "units": "none"
                    }
                ]
            }
        }
    jsonDict["Modifications"]["parameterChange"].extend( baselineList )
    with open( fname, "w" ) as fd:
        json.dump( jsonDict, fd, indent = 4 )
            

def main():
    global plotDt
    stimDict = {}
    parser = argparse.ArgumentParser( description = "MOUSE: Model Optimizer Using Synthetic signaling Experiments. Generates FindSim format experiment definitions for time-series and dose-responses for each input/output combination, and optionally pairwise multi-input combinations." )
    parser.add_argument( '-a', '--allReacs', action='store_true', help='Flag: when set, generate all possible 1-step stimulus-readout pairs by scanning through all reactions.')
    parser.add_argument( "-s", "--stimuli", type = str, nargs = '+', metavar = "molName", help = "Optional: Molecules to stimulate, as a list of space-separated names.", default = [])
    parser.add_argument( "-sr", "--stimulusRange", nargs = 4, metavar = "molName low high duration", help = "Optional: Molecule lowVal highVal duration. Generates a step pulse from low to high with settle, stimulus, and post-stimulus times each equal to _duration_.")
    parser.add_argument( "-b", "--buffer", nargs = '+', metavar = "molName conc", help = "Optional: mol conc [mol conc]... List of buffered molecules with their concentration.")
    parser.add_argument( "-r", "--readouts", type = str, nargs = '+', metavar = "molName", help = "Optional: Readout molecules to monitor, as a list of space-separated names.", default = [] )
    parser.add_argument( "-m", "--model", type = str, help = "Optional: Filepath for chemical kinetic model in HillTau or SBML format. If model is not provided the synthetic file just has zeros for predicted output." )
    parser.add_argument( "-t", "--tau", type = float, help = "Optional: tau for reaction settling, overrides estimate from model if available. Default = 300 seconds." )
    parser.add_argument( '-f', '--findSimFile', type = str, help='Optional: Base name of FindSim output file, which will be of form <file>_TS_<output>_vs__<input>.json for TimeSeries outputs, and <file>_DR_<output>_vs_<input>.json for the dose-responses. Default = "synth"', default = "synth", metavar = "experiment_file_name" )
    parser.add_argument( '-d', '--dir', type = str, help='Optional: Directory in which to put the output files. If it does not exist it is created. Default is current directory', metavar = "output_directory" )
    parser.add_argument( '-p', '--pairwise', action='store_true', help='Flag: when set, generate all pairwise Input combinations as well for TS and DR')
    args = parser.parse_args()

    if ((args.stimulusRange != None) + (len( args.stimuli ) > 0) + args.allReacs) > 1:
        print( "Error: Can only specify one of 'allReacs', 'stimuli' or 'stimulusRange' ")
        quit()

    reacStims = {}
    slist = args.stimuli
    if args.stimulusRange:
        slist = [ args.stimulusRange[0] ]
    for ss in slist:
        rs = reacStims.get( ss )
        if rs:
            rs.extend( args.readouts )
        elif len( args.readouts ) > 0:
            reacStims[ss] = args.readouts

    bufList = generateBufferedMolList( args.buffer )

    if args.model:
        jsonDict = hillTau.loadHillTau( args.model )
        hillTau.scaleDict( jsonDict, hillTau.getQuantityScale( jsonDict ) )
        htmodel = hillTau.parseModel( jsonDict )
    else:
        if args.allReacs:
            print( "Error: Must define model to use --allReacs option" )
            quit()
        htmodel = None

    if args.allReacs:
        for rr, val in htmodel.reacInfo.items():
            subs = np.unique(val.subs )
            for ss in subs:
                rs = reacStims.get( ss )
                if rs:
                    rs.append( rr )
                else:
                    reacStims[ss] = [rr]

    if len(reacStims) == 0:
        print( "Error: No stimulus-readout pairs found. Terminating" )
        quit()


    if args.tau == None: # Use the model tau estimate
        tau = estimateTau( htmodel, reacStims )
    else:
        tau = args.tau

    if args.stimulusRange:
        tau = float(args.stimulusRange[3])
    plotDt = tau * 3 / 24
    settleTime = tau * 2

    if htmodel:
        htmodel.dt = plotDt

    if args.dir == None:
        fname = args.findSimFile
    else:
        if not os.path.exists(args.dir):
            os.makedirs(args.dir)
        elif not os.path.isdir(args.dir):
            print( "Error: Specified path is not a dir. Quitting." )
            quit()
        fname = args.dir + "/" + args.findSimFile
    if args.buffer:
        fname += "_B_" + args.buffer[0]

    msr = stimRange[-2]

    # Replace this with the reacStims dict
    for ss, rlist in reacStims.items():
        # Note ss is a name, and rlist is a list of names.
        # Build the timeseries first
        if args.stimulusRange:
            minConc = float( args.stimulusRange[1] )
            maxConc = float( args.stimulusRange[2] )
        else:
            maxConc = estimateStimConc(htmodel, ss )
            minConc = 0
        stimVec = [ 
                Stim( ss, minConc, tau),
                Stim( ss, maxConc, tau * 2),
                Stim( ss, minConc, tau * 3)
                ]
        #stimVec.extend( [ Stim( ii, maxConc, tau + (1+jj)*tau/12 ) for jj in range(12) ]  )
        #stimVec.append( Stim( ii, 0, tau * 3 ) )
        doseVec = [ Stim( ss, maxConc * conc/msr, (settleTime + 1) * (1+jj), doSettle = True ) for jj, conc in enumerate( stimRange ) ]
        if args.model != None:
            htmodel.dt = plotDt
            referenceOutputs = runHillTau( htmodel, stimVec, rlist, bufList)
            htmodel.dt = settleTime
            doserOutputs = runHillTau( htmodel, doseVec, rlist, bufList )
        else: 
            referenceOutputs = { rr:np.zeros(1+int(tau*3/plotDt)) for rr in rlist }
            doserOutputs = { rr:np.zeros(len(stimRange)) for rr in rlist }

        for key, val in referenceOutputs.items():
            baselineList = generateInputBaseline( htmodel, ss, key ) + bufList
            generateTimeExperiment( fname, stimVec, key, val, baselineList )
        for key, val in doserOutputs.items():
            baselineList = generateInputBaseline( htmodel, ss, key ) + bufList
            generateDoseExperiment( fname, doseVec, key, val, settleTime, baselineList )

if __name__ == '__main__':
    main()
