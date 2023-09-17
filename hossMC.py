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
# 

'''
*******************************************************************
 * File:            hoss.py
 * Description:
 * Author:          Upinder S. Bhalla
 * E-mail:          bhalla@ncbs.res.in
 ********************************************************************/

/**********************************************************************
** This program is part of 'HOSS', the
** Hierarchical Optimization for Systems Simulations
**           copyright (C) 2019-2020 Upinder S. Bhalla. and NCBS
**********************************************************************/

This script orchestrates multilevel optimization, first over many 
individual pathways, then over the cross-talk between them, and possibly
further levels. 
It uses a JSON file for configuring the optimiztaion.
Since the individual optimizations can go in parallel, the system spreads
out the load into many individual multi-parameter optimization runs,
which can be run in parallel. For starters doing this with multiprocessing,
but for clusters it may be necessary to also have a route through mpi4py.
'''

from __future__ import print_function
import sys
import argparse
import json
import jsonschema
import os
import shutil
import heapq
import time
import math
import numpy as np
import findSim
import multiprocessing
import multi_param_minimization
import scramParam

ScorePow = 2.0

HOSS_SCHEMA = "hossSchema.json"

class Monte:
    def __init__(self, pathway, fname, imm, ii, level, score, cumulativeScore ):
        self.pathway = pathway
        self.fname = fname
        self.startModelIdx = imm
        self.scramModelIdx = ii
        self.level = level - 1  # self.level is indexed starting at 0.
        self.score = score      # Score for current level only
        self.cumulativeScore = cumulativeScore
        self.rankScore = (self.score + self.cumulativeScore * self.level) / (self.level + 1)

    def __lt__( self, other ):
        return self.rankScore < other.rankScore

class Row():
    data = []
    names = []
    def __init__( self, idx ):
        self.idx = list(idx)
        self.tot = 0.0
        for dd, ii in zip( self.data, idx ):
            self.tot += dd[ ii ]

    def print( self ):
        for ii, nn, dd in zip( self.idx, self.names, self.data ):
            print( "{}  {:<8.1f}".format( nn, dd[ii] ), end = "" )
        print( "{:8.1f}".format( self.tot ) )

    def __lt__( self, other ):
        if self.tot < other.tot:
            return True
        elif self.tot == other.tot:
            return self.idx < other.idx
        return False

    def __eq__( self, other ):
        return self.idx == other.idx


def buildTopNList( pathwayScores, num ):
    '''
    Generates a sorted list of dicts: ret[rank][pathway_name]
        Each item is of class Monte.
        This sorted list is used to build a composite model from
        model files for each [pathway][rank], extracting the parameter set 
        corresponding to each pathway from the source file.
    Takes a dict of pathwayScores[pathway_name][trial#]

    '''
    ret = []
    sortedMonte = {}
    # first get top N = num for each pathway.
    for key, val in pathwayScores.items():
        sortedMonte[key] = sorted( val )[:num]
    # Then do algorithm for best num.

    topNames = sorted( sortedMonte )
    topData = [ [ss.rankScore for ss in sortedMonte[nn]] for nn in topNames]

    if len( topNames ) == 1:
        name = topNames[0]
        svec = [ {name:mm} for mm in sortedMonte[name] ]
    else:
        vec = [Row( [0] * len( topNames ) )]
        heapq.heapify( vec )
        svec = []

        while( len( svec ) < num ):
            #print( len( vec ), len( svec ), len( topData ), len( topData[0]) )
            rr = heapq.heappop( vec )
            #print( "RR = ", rr )
            if len( vec ) > 0 and not rr == vec[0]:
                entry = { nn:sortedMonte[nn][rr.idx[ii]] for ii, nn in enumerate( topNames ) }   
                #print( "ENTRY = {}, {}  {}".format( nn, ii, rr.idx[ii] ) )
                svec.append( entry )
            for ii in range( len( topData ) ):
                idx2 = list(rr.idx)
                idx2[ii] += 1
                #print( " pushing in idx2", idx2 )
                heapq.heappush( vec, Row( idx2 ) )

    return svec

########################################################################

def runJson( key, val, baseargs ):
    ''' a dummy function for testing '''
    for i, j in baseargs.items():
        print( "{}: {}".format( i, j ), end = "\t\t"  )
    print( "\n KV = {}: {}".format( key, val ) )
    return "foo"


def combineScores( eret ):
    initSum = 0.0
    finalSum = 0.0
    numSum = 0.0
    for ee in eret:
        eis = ee["initScore"]
        fis = ee["score"]
        #print( "eis = ", eis, "     fis = ", fis, "   weight = ", ee["weight"] )
        if eis >= 0:
            initSum += pow( eis, ScorePow ) * ee["weight"]
            finalSum += pow( fis, ScorePow ) * ee["weight"]
            numSum += ee["weight"]
    if numSum == 0:
        return 0,0, 0.0
    else:
        """
        print( "----> IS = {:.3f}   FS = {:.3f}, score = {:.3f}, ".format( 
            pow( initSum / numSum, 1.0/ScorePow ), 
            pow( finalSum / numSum, 1.0/ScorePow ), 
            fis, numSum ) )
        """
        return pow( initSum / numSum, 1.0/ScorePow ), pow( finalSum / numSum, 1.0/ScorePow )


def processIntermediateResults( retvec, baseargs, levelIdx, t0 ):
    prefix = baseargs["filePrefix"]
    fp = open( prefix + baseargs["resultFile"], "w" )
    optfile = prefix + baseargs["optfile"]
    #levelIdx = int(optfile[-6:-5]) # Assume we have only levels 0 to 9.
    totScore = 0.0
    totInitScore = 0.0
    totAltScore = 0.0
    for (results, eret, optTime, paramArgs) in retvec:
        multi_param_minimization.analyzeResults( fp, True, results, paramArgs, eret, optTime, baseargs["scoreFunc"], verbose = False ) # Last arg is 'verbose'
        totScore += results.fun
        #print( "in processIntermediateResults" )
        initScore, altScore = combineScores( eret )
        totAltScore += altScore
        totInitScore += initScore
    if not math.isclose( totScore, totAltScore, rel_tol = 1e-3, abs_tol = 1e-6 ):
        print( "Warning: Score mismatch in processIntermediateResults: ", totScore, totAltScore )
    if __name__ == "__main__":
        print( "Level {} ------- Init Score: {:.3f}   OptimizedScore {:.3f}       Time: {:.3f} s\n".format( levelIdx, totInitScore / len( retvec ), totAltScore / len( retvec ), t0 ) )

    fnames = { "model": baseargs["model"], "optfile": optfile, "map": baseargs["map"], "resultFile": prefix + baseargs["resultFile"] }
    pargs = []
    rargs = []
    # Catenate all the changed params and values.
    for (results, eret, optTime, paramArgs) in retvec:
        rargs.extend( results.x )
        pargs.extend( paramArgs )
    multi_param_minimization.saveTweakedModelFile( {}, pargs, rargs, fnames )
    return [ totInitScore, totAltScore, len( retvec ) ]

def processFinalResults( results, baseargs, intermed, t0 ):

    totScore = 0.0
    numScore = 0.0
    for retVec in results:
        for rr in retVec:
            totScore += rr[0].fun
        numScore += len( retVec )
    totInitScore = sum( [ ii[0] for ii in intermed ] )
    totAltScore = sum ( [ ii[1] for ii in intermed ] )
    numRet = sum ( [ ii[2] for ii in intermed ] )
    #print( "Multilevel optimization complete in {:.3f} s--- Mean Score = {:.3f} ".format( t0, totScore/numScore ) )
    #if __name__ == "__main__":
    print( "{}: HOSS opt: Init Score = {:.3f}, Final = {:.3f} {:.3f}, Time={:.3f}s".format( 
        baseargs['optfile'], 
        totInitScore/numRet, 
        totAltScore/numRet, 
        totScore/numRet, 
        t0 ))

#######################################################################

def runOptSerial( optBlock, baseargs ):
    score = []
    for name, ob in optBlock.items():
        #print( "\n", name, ob, "\n###########################################" )
        score.append( multi_param_minimization.runJson(name, ob, baseargs ) )
        #print( "in runOptSerial" )
        initScore, optScore = combineScores (score[-1][1] )
        #print( "OptSerial {:20s} Init={:.3f}     Opt={:.3f}     Time={:.3f}s".format(name, initScore, optScore, score[-1][2] ) )
        # optScore == score[-1][0].fun

    #print( "Serial SCORE = ", [ss[0].fun for ss in score] )
    return score

def ticker( arg ):
    return

def threadProc( name, ob, baseargs ):
    currProc = multiprocessing.current_process()
    currProc.daemon = False  # This allows nested multiprocessing.
    #print( "\n", name, ob, "\n###########################################" )
    return multi_param_minimization.runJson(name, ob, baseargs )

def runOptThreads( optBlock, baseargs ):
    numProcesses = baseargs["numProcesses"]
    if numProcesses == 0:
        numProcesses = multiprocessing.cpu_count() // 8 # Assume 8 expts per opt
    if numProcesses == 0:
        numProcesses = 1
    numProcesses = min( numProcesses, len(optBlock) )

    pool = multiprocessing.Pool( processes = numProcesses )
    score = []
    ret = []
    for name, ob in optBlock.items():
        ret.append( pool.apply_async( threadProc, args = ( name, ob, baseargs ), callback = ticker ) )
    score = [rr.get() for rr in ret ]

    #print( "Thread SCORE = ", [ss[0].fun for ss in score] )
    return score

def runOptMPI( optBlock, baseargs ):
    print( "MPI version not yet implemented, using serial version")
    return runOptSerial( optBlock, baseargs )
        
######################################

def loadConfig( args ):
    # Load and validate the config file
    try:
        with open( args.config ) as json_file:
            config = json.load( json_file )
            relpath = os.path.dirname( __file__ )
            if relpath != '': # outside local directory
                #relpath = relpath + '/'
                fs = relpath + '/' + HOSS_SCHEMA
            else:
                fs = HOSS_SCHEMA

            with open( fs ) as _schema:
                schema = json.load( _schema )
                jsonschema.validate( config, schema )
    except IOError:
        print( "Error: Unable to find HOSS config file: " + args.config )
        quit()

    # We have a number of necessary option values.
    # The order of priority is: command line, config file, pgm default
    # requiredDefaultArgs is set here in case neither the config file nor 
    # the command line has the argument. We can't use default args in 
    # argparser as this would override the config file.
    requiredDefaultArgs = { 
            "tolerance": 0.001, 
            "scoreFunc": "NRMS", 
            "algorithm": "SLSQP", 
            "solver": "LSODA", 
            "exptDir": "./Expts" ,
            "model": "./Models/model.json",
            "map": "./Maps/map.json",
            "filePrefix": "",
            "scramRange": 2,
            "numScram": 0,
            "numTopModels": 0
        } 
    baseargs = vars( args )
    for key, val in requiredDefaultArgs.items():
        if baseargs[key]:   # command line arg given
            continue
        elif key in config: # Specified in Config file
            baseargs[key] = config[key]
        else:               # Fall back to default.
            baseargs[key] = val

    # Build up optimization blocks. Within a block we assume that the
    # individual optimizations do not interact and hence can run in 
    # parallel. Once a block is done its values are consolidated and used
    # for the next block.
    assert( 'model' in baseargs )
    assert( 'resultFile' in baseargs )

    return baseargs, config

#######################################################################

def runOptimizer( blocks, baseargs, parallelMode, blocksToRun, t0 ):
    origModel = baseargs['model'] # Use for loading model
    optModel = baseargs['optfile'] # Use for final model save
    optResults = baseargs['resultFile'] # Use for final results
    modelFileSuffix = origModel.split( "." )[-1]
    results = []
    intermed = []

    for hossLevel in blocks: # Assume blocks are in order of execution.
        optBlock = {}
        hl = hossLevel["hierarchyLevel"]
        # Specify intermediate model and result files
        baseargs['optfile'] = "./_optModel{}.{}".format(hl, modelFileSuffix)
        baseargs['resultFile'] = "./_optResults{}.txt".format( hl )
        for key, val in hossLevel.items():
            if key == "name" or key == "hierarchyLevel":
                continue
            if "optModelFile" in val:
                baseargs['optfile'] = val["optModelFile"]
            if "resultFile" in val:
                baseargs['resultFile'] = val["resultFile"]
            # Either run all items or run named items in block.
            #print( "Args.blocks = ", args.blocks, "     Key = ", key )
            if blocksToRun == [] or key in blocksToRun:
                optBlock[ key] = val

        if len( optBlock ) == 0:
            continue        # Nothing to see here, move along to next level
        # Now we have a block to optimize, use suitable method to run it.
        # We can run items in a block in any order, but the whole block
        # must be wrapped up before we go to the next level of heirarchy.
        t1 = time.time()
        if parallelMode == "serial":
            score = runOptSerial( optBlock, baseargs )
        elif parallelMode == "threads":
            score = runOptThreads( optBlock, baseargs )
        elif parallelMode == "MPI":
            score = runOptMPI( optBlock, baseargs )
        t2 = time.time()
        # This saves the scores and the intermediate opt file, to use for
        # next level of optimization.
        ret = processIntermediateResults( score, baseargs, hl, t2 - t1 )
        t1 = t2
        intermed.append( ret )
        baseargs["model"] = baseargs["filePrefix"] + baseargs["optfile"] # Apply heirarchy to opt
        results.append( score )
    print( "MODELFILE = ", baseargs["model"] )
    computeModelScores( blocks, baseargs, time.time() - t0 )
    #processFinalResults( results, baseargs, intermed, time.time() - t0  )

def worker( baseargs, exptFile ):
    score, elapsedTime, diagnostics = findSim.innerMain( exptFile, 
            scoreFunc = baseargs["scoreFunc"],
            modelFile = baseargs["model"], mapFile = baseargs["map"], 
            hidePlot = True, ignoreMissingObj = True, silent = True,
            solver = "LSODA", plots = False )
    return score

def computeModelScores( blocks, baseargs, runtime ):
    MIN_BOUND = 1e-10
    MAX_BOUND = 1e6
    ret = 0.0
    nret = 0
    levelScores = []
    totScore = 0.0
    ed = baseargs["exptDir"]
    if len( ed ) > 0 and ed[-1] != "/":
        ed = ed + "/"
    sf = baseargs["scoreFunc"]
    pool = multiprocessing.Pool( processes = multiprocessing.cpu_count() )
    for hossLevel in blocks: # Assume blocks are in order of execution.
        lret = 0.0
        print( "L{} ".format( hossLevel["hierarchyLevel"] ), end = "" )
        meanPathwayScore = 0.0
        numPathways = 0
        scoreDict = {}
        for pathway, val in hossLevel.items():
            if pathway in ["name", "hierarchyLevel"]:
                continue
            expt = val.get( "expt" )
            if expt:
                exptList = sorted( [ ee for ee in expt ] )
            else:
                raise( "Missing Expt list in pathway " + expt )
            if len( exptList ) == 1:
                pathwayScore = worker( baseargs, hossLevel[exptList[0]] )
                meanPathwayScore += pathwayScore
                numPathways += 1
            elif len( exptList ) > 1:
                ret = []
                for ee in exptList:
                    ret.append( pool.apply_async(worker, args = (baseargs, ed + ee) ) )
                sumScore = 0.0
                sumWts = 0.0
                for rr, ee in zip( ret, exptList ):
                    score = rr.get()
                    #print( score )
                    wt = expt[ee]["weight"]
                    sumScore += score * score * wt
                    sumWts += wt
                pathwayScore = np.sqrt( sumScore / sumWts )
                meanPathwayScore += pathwayScore
                numPathways += 1
            else:
                pathwayScore = -1.0
            print( "{:12s}{:.3f}   ".format(pathway, pathwayScore), end="" )
            scoreDict[pathway] = pathwayScore
        print()
        mean = meanPathwayScore/numPathways if numPathways > 0 else -1.0
        totScore += mean
        levelScores.append( scoreDict )
    print( "Final Score for {} levels in {} = {:.3f}, Time={:.2f}s".format( len( levelScores ), baseargs["model"], totScore / len( levelScores ), runtime) )
    return levelScores



def runMCoptimizer(blocks, baseargs, parallelMode, blocksToRun, t0):
    origModel = baseargs['model'] # Use for loading model
    mapFile = baseargs['map'] # Use for loading model
    scramRange = baseargs["scramRange"]
    numScram = baseargs["numScram"]
    numTopModels = baseargs["numTopModels"]
    modelFileSuffix = origModel.split( "." )[-1]
    scramModelName = "scram"
    prefix = "MONTE/"
    intermed = []
    results = []
    t0 = time.time()

    # First compute the original score. It may well be better.
    origScores = computeModelScores( blocks, baseargs, 0 )
    #quit()
    startModelList = [(origModel, 0.0)] # Model name and cumulative score

    for idx, hossLevel in enumerate(blocks): # Assume blocks are in order of execution.
        hierarchyLevel = idx + 1
        pathwayScores = {}        
        optBlock = {}
        numScramPerModel = numScram // len(startModelList)
        #print( "startModelList =  ", startModelList )
        # Can't use the internal hierarchyLevel because it might not be
        # indexed from zero.
        for name, ob in hossLevel.items():
            if name == "name" or name == "hierarchyLevel":
                continue
            if "optModelFile" in ob:
                baseargs['optfile'] = ob["optModelFile"]
            if "resultFile" in ob:
                baseargs['resultFile'] = ob["resultFile"]
            if blocksToRun == [] or name in blocksToRun:
                optBlock[name] = ob
            else:
                continue
            paramList = ob["params"]
            #print( paramList )
            pathwayScores[name] = []

            for imm, (mm, cumulativeScore) in enumerate( startModelList ):
                sname = "{}{}_{}_{}.{}".format( prefix, scramModelName, name, imm, modelFileSuffix )
                scramParam.generateScrambled( mm, mapFile, sname, numScramPerModel, paramList, scramRange )
                # Here we put in the starting model as it may be best
                if imm == 0:
                    ss = origScores[idx][name]
                    print( "ORIG SCORE {} = {:.3f}".format( name, ss ) )
                else:
                    ss = cumulativeScore
                pathwayScores[name].append( 
                    Monte( name, mm, imm, 0, 
                    hierarchyLevel, 
                    ss, ss )
                )
                for ii in range( numScramPerModel ):
                    baseargs["model"] = "{}{}_{}_{}_{:03d}.{}".format( 
                            prefix, scramModelName, name, 
                            imm, ii, modelFileSuffix )
                    #print( "BASEARGS = ", baseargs["model"] )
                    #newScore = runJson( name, ob, baseargs )
                    score = multi_param_minimization.runJson( name, ob, baseargs )
                    initScore, newScore = combineScores( score[1] )
                    print( ".", end = "", flush=True)
                    pathwayScores[name].append( 
                            Monte( name, baseargs["model"], imm, ii, 
                            hierarchyLevel, 
                            newScore, cumulativeScore ) )

        if len( optBlock ) == 0:
            continue        # Nothing to see here, move along to next level

        topN = buildTopNList( pathwayScores, numTopModels )
        startModelList = []
        if baseargs["verbose"]:
            for idx, tt in enumerate( topN ):
                for name, monte in tt.items():
                    print( "\nL{}.{}: {} scores={:.3f} {:.3f}   fname= {}".format(
                        hierarchyLevel, idx, name, 
                        monte.score, monte.rankScore, monte.fname ) )

        # Build merged model.
        for idx, tt in enumerate( topN ):
            rmsScore = 0.0
            firstBlock = True
            outputModel = "topN_{}_{:03d}.{}".format( hierarchyLevel, idx, modelFileSuffix )
            for name, ob in optBlock.items():
                monte = tt[name]
                rmsScore += monte.score * monte.score
                startModel = monte.fname
                if firstBlock:
                    shutil.copyfile( startModel, outputModel )
                    #print( "Copyfile ", startModel, "       ", outputModel )
                else:
                    scramParam.mergeModels( startModel, mapFile, monte.fname, outputModel, ob["params"] )
                firstBlock = False
                #print( "CumulativeScore for {} = {:.3f}".format(outputModel, monte.cumulativeScore ) )

            rmsScore = np.sqrt( rmsScore / len( optBlock )  )
            newScore = (rmsScore + monte.cumulativeScore * monte.level)/(monte.level + 1 )
            startModelList.append( (outputModel, newScore  ) )

    # Finally compute the end score. It should be a lot better.
    baseargs["model"] = "topN_{}_{:03d}.{}".format( hierarchyLevel, 0, modelFileSuffix )
    print()
    finalScores = computeModelScores( blocks, baseargs, time.time() - t0 )

#######################################################################

def main( args ):
    t0 = time.time()
    parser = argparse.ArgumentParser( description = 
            'This script orchestrates multilevel optimization, first over many individual pathways, then over the cross-talk between them, and possibly further levels.  It uses a JSON file for configuring the optimization.  Since the individual optimizations can go in parallel, the system spreads out the load into many individual multi-parameter optimization runs, which can be run in parallel.' )
    parser.add_argument( 'config', type = str, help='Required: JSON configuration file for doing the optimization.')
    parser.add_argument( '-t', '--tolerance', type = float, help='Optional: Tolerance criterion for completion of minimization' )
    parser.add_argument( '-a', '--algorithm', type = str, help='Optional: Algorithm name to use, from the set available to scipy.optimize.minimize. Options are CG, Nelder-Mead, Powell, BFGS, COBYLA, SLSQP, trust-constr. The library has other algorithms but they either require Jacobians or they fail outright. There is also L-BFGS-B which handles bounded solutions, but this is not needed here because we already take care of bounds. SLSQP works well and is the default.' )
    parser.add_argument( '-b', '--blocks', nargs='*', default=[],  help='Blocks to execute within the JSON file. Defaults to empty, in which case all of them are executed. Each block is the string identifier for the block in the JSON file.' )
    parser.add_argument( '-m', '--model', type = str, help='Optional: Composite model definition file. First searched in directory "location", then in current directory.' )
    parser.add_argument( '-map', '--map', type = str, help='Model entity mapping file. This is a JSON file.' )
    parser.add_argument( '-e', '--exptDir', type = str, help='Optional: Location of experiment files.' )
    parser.add_argument( '-o', '--optfile', type = str, help='Optional: File name for saving optimized model', default = "" )
    parser.add_argument( '-fp', '--filePrefix', type = str, help='Optional: Prefix to add to names of optfile and resultFile. Can also be a directory path.', default = "" )
    parser.add_argument( '-p', '--parallel', type = str, help='Optional: Define parallelization model. Options: serial, MPI, threads. Defaults to serial. MPI not yet implemented', default = "serial" )
    parser.add_argument( '-n', '--numProcesses', type = int, help='Optional: Number of blocks to run in parallel, when we are not in serial mode. Note that each block may have multiple experiments also running in parallel. Default is to take numCores/8.', default = 0 )
    parser.add_argument( '-ns', '--numScram', type = int, help='Optional: Number of Monte Carlo samples to take by scrambling files. By default no Monte Carlo sampling will be done', default = 0 )
    parser.add_argument( '-nt', '--numTopModels', type = int, help='Optional: For Monte Carlo, number of models out of score-sorted set, to take to next stage of optimization to use as starting points for further scrambling. By default no Monte Carlo sampling will be done.', default = 0 )
    parser.add_argument( '-r', '--resultFile', type = str, help='Optional: File name for saving results of optimizations as a table of scale factors and scores.', default = "" )
    parser.add_argument( '-sf', '--scoreFunc', type = str, help='Optional: Function to use for scoring output of simulation. Default: NRMS' )
    parser.add_argument( '-scr', '--scramRange', type = float, help='Optional, used only when doing Monte Carlo sampling: Range for scrambling model parameters. Default: 2.0', default = 2.0 )
    parser.add_argument( '--solver', type = str, help='Optional: Numerical method to use for ODE solver. Ignored for HillTau models. Default = "LSODA".')
    parser.add_argument( '-v', '--verbose', action="store_true", help="Flag: default False. When set, prints all sorts of warnings and diagnostics.")
    parser.add_argument( '-st', '--show_ticker', action="store_true", help="Flag: default False. Prints out ticker as optimization progresses.")
    args = parser.parse_args( args )

    baseargs, config = loadConfig( args )
    blocks = config["HOSS"]

    if baseargs["numScram"] == 0 or baseargs["numTopModels"] == 0:
        runOptimizer( blocks, baseargs, args.parallel, args.blocks, t0 )
    else:
        runMCoptimizer( blocks, baseargs, args.parallel, args.blocks, t0 )


#######################################################################
# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
    main( sys.argv[1:] )
