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
 * File:            mousse.py
 * Description:     Model Optimizer Using Synthetic Signaling Experiments
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
stimRange = [ 0.1, 0.2, 0.5, 1, 2.0, 5.0, 10.0 ]
settleTimeScale = stimRange[-1]  # How much longer is settleTime than midTime?

class Stim:
    ### Advance to specified time, and then set the conc to the stim value.
    def __init__( self, mol, conc, time ):
        self.molName = mol
        self.mooseMol = getMooseName( mol )
        self.hillTauMol = getHillTauName( mol )
        self.conc = conc
        self.time = time
        self.molIndex = 0


def runHillTau( model, stimVec, outMols ):
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

def findMaxConc( htmodel, stimMol ):
    if htmodel:
        mi = htmodel.molInfo.get( stimMol )
        if mi:
            conc = mi.concInit
            if conc == 0:
                # Try basename if it starts with an 'a' or a 'p'
                tryname = stimMol[1:]
                mi = htmodel.molInfo.get( tryname )
                if mi:
                    return mi.concInit
            else:
                return mi.concInit
        else:
            raise ValueError( "Nonexistent stimulus molecule: ", stimMol )
    return 1.0

def estimateTau( htmodel, readoutMol ):
    if htmodel:
        ri = htmodel.reacInfo.get( readoutMol )
        if ri:
            return ri.tau * 2.0 # Give the reac enough time to settle.
        else:
            raise ValueError( "Nonexistent reactant: ", readoutMol )
    return 300.0


def generateStimData( stimVec ):
    ret = []
    for i in stimVec:
        ret.append( [i.time, i.conc * 1000] )
    return ret

def generateReadoutData( plotDt, refVals ):
    ret = [ [ ii*plotDt, rr*1000, 0.0] for ii, rr in enumerate( refVals ) ]
    return ret

def generateExperiment( fname, stimVec, refMol, refVals ):
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
                }
            }
    with open( fname, "w" ) as fd:
        json.dump( jsonDict, fd, indent = 4 )
            

def main():
    global plotDt
    stimDict = {}
    parser = argparse.ArgumentParser( description = "MOUSSE: Model Optimizer Using Synthetic Signaling Experiments. Generates FindSim format experiment definitions for time-series and dose-responses for each input/output combination, and optionally pairwise multi-input combinations." )
    parser.add_argument( "-s", "--stimuli", type = str, nargs = '+', metavar = "molName", help = "Required: Molecules to stimulate, as a list of space-separated names.")
    parser.add_argument( "-r", "--readouts", type = str, nargs = '+', metavar = "molName", help = "Required: Readout molecules to monitor, as a list of space-separated names." )
    parser.add_argument( "-m", "--model", type = str, help = "Optional: Filepath for chemical kinetic model in HillTau or SBML format. If model is not provided the synthetic file just has zeros for predicted output." )
    parser.add_argument( "-t", "--tau", type = float, help = "Optional: tau for reaction settling, overrides estimate from model if available. Default = 300 seconds." )
    parser.add_argument( '-f', '--findSimFile', type = str, help='Optional: Base name of FindSim output file, which will be of form <file>_TS_<output>_vs__<input>.json for TimeSeries outputs, and <file>_DR_<output>_vs_<input>.json for the dose-responses. Default = "synth"', default = "synth", metavar = "experiment_file_name" )
    parser.add_argument( '-p', '--pairwise', action='store_true', help='Flag: when set, generate all pairwise Input combinations as well for TS and DR')
    args = parser.parse_args()

    if args.stimuli == None:
        print( "Error: At least one stimulus molecule must be defined." )
        quit()
    if args.readouts == None:
        print( "Error: At least one readout molecule must be defined." )
        quit()

    if args.model:
        jsonDict = hillTau.loadHillTau( args.model )
        hillTau.scaleDict( jsonDict, hillTau.getQuantityScale( jsonDict ) )
        htmodel = hillTau.parseModel( jsonDict )
    else:
        htmodel = None

    if args.tau == None: # Use the model tau estimate
        tau = estimateTau( htmodel, args.readouts[0] )
    else:
        tau = args.tau
    plotDt = tau * 3 / 24

    if htmodel:
        htmodel.dt = plotDt


    for ii in args.stimuli:
        # Build the timeseries first
        stimVec = [ Stim( ii, 0, 0.0 ), 
                Stim( ii, findMaxConc(htmodel, ii), tau ),
                Stim( ii, 0, tau * 2 ),
                Stim( ii, 0, tau * 3 )
                ]
        if args.model != None:
            referenceOutputs = runHillTau( htmodel, stimVec, args.readouts )
        else: 

            referenceOutputs = { rr:np.zeros(1+int(tau*3/plotDt)) for rr in args.readouts }
        for key, val in referenceOutputs.items():
            generateExperiment( args.findSimFile, stimVec, key, val )

if __name__ == '__main__':
    main()
