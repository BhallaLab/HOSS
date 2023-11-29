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
OptModelFname = "OPTI"

runStatus = []

class Monte:
    def __init__(self, pathway, fname, imm, ii, level, score, cumulativeScore ):
        self.pathway = pathway
        self.fname = fname
        self.startModelIdx = imm
        self.scramModelIdx = ii
        self.level = level - 1  # self.level is indexed starting at 0.
        self.score = score      # Score for current level only
        self.cumulativeScore = cumulativeScore
        self.rankScore = self.score
        #self.rankScore = (self.score + self.cumulativeScore * self.level) / (self.level + 1)
        #print( "making Monte {}.{}.{}.{}@{:.3f}".format( pathway, fname, imm, ii, score))

    def __lt__( self, other ):
        return self.score < other.score

    def __eq__( self, other ):
        return self.fname == other.fname

    def __hash__( self ):
        return hash( self.fname + str( self.startModelIdx ) + str( self.scramModelIdx ) + str( self.level ) )

#########################################################################
class Row():
    data = []
    names = []
    def __init__( self, idx ):
        self.idx = list(idx)
        self.tot = 0.0
        for dd, ii in zip( self.data, idx ):
            if len( dd ) <= ii:
                raise ValueError( "Row init: len(dd)=={} < ii == {}".format( len(dd), ii) )
            self.tot += dd[ ii ]
        #print( "ROW: ", idx, self.tot )

    def print( self ):
        print( "RR= ", self.idx, self.tot )
        '''
        print( "Lidx = {}, Lnames = {}, Ldata = {}".format( len(self.idx), len( self.names ), len( self.data ) ) )
        for ii, nn, dd in zip( self.idx, self.names, self.data ):
            print( "{}  {:<8.1f}".format( nn, dd[ii] ), end = "" )
        print( "{:8.1f}".format( self.tot ) )
        '''

    def __lt__( self, other ):
        if self.tot < other.tot:
            return True
        elif self.tot == other.tot:
            return self.idx < other.idx
        return False

    def __eq__( self, other ):
        return self.idx == other.idx

#########################################################################
class RunStatus():
    def __init__( self, idx ):
        self.finished = False
        self.justFinished = False
        self.startTime = 0.0
        self.endTime = 0.0
        self.timedOut = False
        self.scores = None
        self.idx = idx

#########################################################################
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
        sortedMonte[key] = sorted( val )[:num*2]
    # Then do algorithm for best num.

    topNames = sorted( sortedMonte )
    topData = [ [ss.rankScore for ss in sortedMonte[nn]] for nn in topNames]
    Row.data = topData
    Row.names = topNames

    if len( topNames ) == 1:
        name = topNames[0]
        svec = [ {name:mm} for mm in sortedMonte[name] ][:num]
    else:
        vec = [Row( [0] * len( topNames ) )]
        heapq.heapify( vec )
        svec = []

        while( len( svec ) < num ):
            #print( len( vec ), len(svec), len(topData), len( topData[0]) )
            rr = heapq.heappop( vec )
            #rr.print()
            if ( len(vec) == 0 ) or ( (len(vec) > 0) and not rr == vec[0] ):
                '''
                print( "LENS = ", len( rr.idx ), len( sortedMonte ) )
                for ii, nn in enumerate( topNames ):
                    print( "ENTRY = {}, {}  {}, lsm = {}, {}".format( nn, ii, rr.idx[ii], len( sortedMonte ), len( sortedMonte[nn] ) ) )
                '''
                entry = { nn:sortedMonte[nn][rr.idx[ii]] for ii, nn in enumerate( topNames ) }   
                svec.append( entry )
            for ii in range( len( topData ) ):
                idx2 = list(rr.idx)
                idx2[ii] += 1
                heapq.heappush( vec, Row( idx2 ) )

    '''
    for idx, ss in enumerate( svec ):
        stot = 0.0
        for nn, val in ss.items():
            print( "{}: {} = {:.3f}".format( idx, nn, val.score ), end = "    " )
            stot += val.score
        print( "Tot = {:.3f}".format( stot ) )
    '''
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
        return 0.0, 0.0
    else:
        """
        print( "----> IS={:.3f}   FS={:.3f}, finalSum={:.3f}, numSum={:.3f}".format( 
            pow( initSum / numSum, 1.0/ScorePow ), 
            pow( finalSum / numSum, 1.0/ScorePow ), 
            finalSum, numSum ), flush = True )
        """
        return pow( initSum / numSum, 1.0/ScorePow ), pow( finalSum / numSum, 1.0/ScorePow )


def processIntermediateResults( retvec, baseargs, levelIdx, t0, idx=None ):
    prefix = baseargs["outputDir"]
    if len( prefix ) > 0 and prefix[-1] != '/':
        prefix += "/"
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
        lstr = "" if idx == None else "{:03d}.".format( idx )
        print( "Level {}{} --- Init Score: {:.3f}   OptimizedScore {:.3f}       Time: {:.3f} s".format( lstr, levelIdx, totInitScore / len( retvec ), totAltScore / len( retvec ), t0 ) )

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
    #print( "Model = {}, optfile = {}".format( baseargs['model'], baseargs['optfile'] ) )
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

def runOptThreads( optBlock, baseargs ):
    numProcesses = baseargs["numProcesses"]
    if numProcesses == 0:
        numProcesses = max( multiprocessing.cpu_count()//8, 1)
    numProcesses = min( numProcesses, len(optBlock) )

    pool = multiprocessing.Pool( processes = numProcesses )
    ret = []
    for name, ob in optBlock.items():
        ret.append( pool.apply_async( wrapMultiParamMinimizer, args = ( name, ob, baseargs ), callback = ticker ) )
    score = []
    timeout = baseargs["timeout"]
    for rr in ret:
        try:
            score.append(rr.get() )
        except multiprocessing.TimeoutError:
            if baseargs["verbose"]:
                print( "Warning: Timeout in runOptThreads. Skipping pathway")

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
    # We can't use default args in hossSchema because this doesn't create
    # any absent key:val, it is just an annotation.
    requiredDefaultArgs = { 
            "tolerance": 0.001, 
            "scoreFunc": "NRMS", 
            "algorithm": "SLSQP", 
            "solver": "LSODA", 
            "exptDir": "./Expts" ,
            "outputDir": "./OPTIMIZED" ,
            "scramDir": "./SCRAM" ,
            "resultFile": "result.txt",
            "optfile": "",
            "model": "./Models/model.json",
            "map": "./Maps/map.json",
            "freezeParams": None,
            "method": "hoss",
            "scrambleRange": 2,
            "numScramble": 0,
            "numInitScramble": 0,
            "numTopModels": 0,
            "numProcesses": 0,
            "timeout": None
        } 
    baseargs = vars( args )
    for key, val in requiredDefaultArgs.items():
        if baseargs[key]:   # command line arg given
            continue
        elif key in config: # Specified in Config file
            baseargs[key] = config[key]
        elif key in config["hossMethod"]:
            baseargs[key] = config["hossMethod"][key]
            #print( "basargs[key]=", key, baseargs[key] )
        else:               # Fall back to default.
            baseargs[key] = val

    # Build up optimization blocks. Within a block we assume that the
    # individual optimizations do not interact and hence can run in 
    # parallel. Once a block is done its values are consolidated and used
    # for the next block.
    assert( 'model' in baseargs )
    assert( 'resultFile' in baseargs )
    modelFileSuffix = baseargs["model"].split( "." )[-1]
    if baseargs["optfile"] == "":    # Still empty
        baseargs["optfile"] = "opt." + modelFileSuffix

    return baseargs, config

#######################################################################
def runInitScramThenOptimize( blocks, baseargs, t0 ):
    global runStatus
    origModel = baseargs['model'] # Use for loading model
    modelFileSuffix = origModel.split( "." )[-1]
    scramModelName = "scram"
    prefix = baseargs["scramDir"]
    if prefix and (prefix[-1] != "/"):
        prefix = prefix + "/"
    outputDir = baseargs["outputDir"]
    if len( outputDir ) > 0 and outputDir[-1] != '/':
        outputDir += '/'
    numProcesses = baseargs["numProcesses"]
    numInitScramble = baseargs["numInitScramble"]
    NumTopN = min( 10, numInitScramble )
    runStatus = [ RunStatus( ii ) for ii in range( numInitScramble ) ]
    if numProcesses == 0:
        numProcesses = max( multiprocessing.cpu_count()//8, 1)

    origScores, initScore = computeModelScores( blocks, baseargs, 0 )

    pool = multiprocessing.Pool( processes = numProcesses )
    scramRange = baseargs["scrambleRange"]
    sname = "{}{}.{}".format( prefix, scramModelName, modelFileSuffix )
    #print( "PREFIX = ", prefix, "   SNAME = ", sname )
    scramParam.generateScrambled( origModel, baseargs["map"], sname, 
            numInitScramble, None, scramRange, 
            freezeParams = baseargs["freezeParams"],
            ignoreMissingFreeze = True )

    pool = multiprocessing.Pool( processes = numProcesses )
    ret = []
    #print( "Starting to farm out scrambles: ", time.time() - t0 )
    startTimes = [t0] * numInitScramble
    for ii in range( numInitScramble ):
        # Need to redo each iter because model is updated
        newargs = dict( baseargs ) 
        newargs["model"] = "{}{}_{:03d}.{}".format( prefix, 
            scramModelName, ii, modelFileSuffix )
        runStatus[ii].startTime = time.time()
        ret.append( pool.apply_async( wrapRunOptimizer, args = ( blocks, newargs, ii, t0 ), callback = ticker ) )
        #print( "Farmed", ii, " at time ", time.time() - t0 )
    score = []
    #pollHoss( ret, runStatus, baseargs )


    for idx, rr in enumerate( ret ):
        try:
            #print( "to get {} at time {:.3f}; startTime = {:.3f}".format( idx, time.time() - t0, time.time() - startTimes[idx] ), flush = True )
            score.append( rr.get() )
            #print( "got {} at time {:.3f}; startTime = {:.3f}".format( idx, time.time() - t0, time.time() - startTimes[idx] ), flush = True )
        except multiprocessing.TimeoutError:
            if baseargs["verbose"]:
                print( "Warning: Timeout in runInitScramThenOptimize. Skipping number ", idx, flush = True )
            score.append( None )

    # Do something with lots of scores.
    scoreList = [] # We'll have a list of tuples of (scramIdx, score)
    for scramIdx, scramVal in enumerate( score ):    
        totScore = 0.0
        if scramVal == None:
            print( "final Score for scramIdx {:03d} in {} = None".format(
                scramIdx, baseargs["model"] ) )
            continue
        for level, ss in enumerate( scramVal ):
            # Each level contains a scoreDict of { pathway: score }
            pathwayScore = 0.0
            for pp, val in ss.items():
                print( "{:12s}{:.3f}   ".format(pp, val), end="" )
                pathwayScore += val
            print()
            totScore += (pathwayScore / len( ss ) if len( ss ) > 0 else 0.0)
        print( "Final Score for scramIdx {:03d} in {} = {:.3f}".format(
            scramIdx, baseargs["model"], totScore / len( scramVal ) ) )
        scoreList.append( (scramIdx, totScore / len( scramVal ) ) )

    sortedScoreList = sorted( scoreList, key = lambda x: x[1] )
    sortedResults = "{}sortedResults.txt".format( outputDir )
    with open( sortedResults, "w" ) as fp:
        for ss in sortedScoreList:
            fp.write( "{:03d}   {:.4f}\n".format( ss[0], ss[1] ) )
    for idx, ss in enumerate( sortedScoreList[:NumTopN] ):
        srcfile = "{}{}_{:03d}.{}".format( outputDir, OptModelFname,
            ss[0], modelFileSuffix )
        destfile = "{}topN_{:03d}.{}".format( outputDir, idx, modelFileSuffix )
        shutil.copyfile( srcfile, destfile )

    return initScore, sortedScoreList[0][1]

def wrapRunOptimizer( blocks, baseargs, idx, t0 ):
    print( "Launching wrapRunOptimizer {} at time = {:.3f} for {} ".format( idx, time.time() - t0, baseargs["model"] ) )
    currProc = multiprocessing.current_process()
    currProc.daemon = False  # This allows nested multiprocessing.
    levelScores, initScore, optScore = runHossOptimizer( blocks, baseargs, 
            "serial", [], time.time(), idx )
    return levelScores

#######################################################################

def insertFileIdx( fname, idx ):
    [fpre, fext] = os.path.splitext( fname )
    return "{}_{:03d}{}".format( fpre, idx, fext )

def runFlatOptimizer( blocks, baseargs, parallelMode, blocksToRun, t0, 
        idx = None ):
    origModel = baseargs['model'] # Use for loading model
    outputDir = baseargs["outputDir"]
    if len( outputDir ) > 0 and outputDir[-1] != '/':
        outputDir += '/'
    modelFileSuffix = origModel.split( "." )[-1]
    results = []
    intermed = []
    origScores, initScore = computeModelScores( blocks, baseargs, 0 )

    flatPathway = {
        "comment": "This is the flattened pathway with all expts", 
        "resultFile": baseargs['resultFile'],
        "optModelFile": baseargs['optfile'],
        "expt": {},
        "params": [],
        "paramBounds": {}
    }

    #baseargs['optfile'] = flatPathway['optModelFile']
    #baseargs['resultFile'] = flatPathway['resultFile']
    #print( "OUTPUT IN ", baseargs["optfile"], baseargs["resultFile"] )

    for hossLevel in blocks: # Assume blocks are in order of execution.
        optBlock = {}
        hl = hossLevel["hierarchyLevel"]
        # Specify intermediate model and result files in case they are here
        #baseargs['optfile'] = "{}/_optModel{}.{}".format(outputDir, hl, modelFileSuffix)
        #baseargs['resultFile'] = "{}/_optResults{}.txt".format( outputDir, hl )
        for key, val in hossLevel.items():
            if key == "name" or key == "hierarchyLevel":
                continue
            # Either run all items or run named items in block.
            #print( "Args.blocks = ", args.blocks, "     Key = ", key )
            if blocksToRun == [] or key in blocksToRun:
                flatPathway["expt"].update( val["expt"] )
                flatPathway["params"].extend( val["params"] )
                pb = val.get("paramBounds")
                if pb:
                    flatPathway["paramBounds"].update( pb )

    flatBlock = {
        #"name": "all", 
        #"hierarchyLevel": 1, 
        "flatPathway": flatPathway 
    }


    # Now we have a block to optimize, use suitable method to run it.
    # We can run items in a block in any order, but the whole block
    # must be wrapped up before we go to the next level of heirarchy.
    t1 = time.time()
    if parallelMode == "serial":
        score = runOptSerial( flatBlock, baseargs )
    elif parallelMode == "threads":
        score = runOptThreads( flatBlock, baseargs )
    elif parallelMode == "MPI":
        score = runOptMPI( flatBlock, baseargs )
    t2 = time.time()
    # This saves the scores and the intermediate opt file, to use for
    # next level of optimization.
    ret = processIntermediateResults( score, baseargs, 1, t2 - t1 )
    t1 = t2
    intermed.append( ret )
    baseargs["model"] = outputDir + baseargs["optfile"] # Apply heirarchy to opt
    results.append( score )
    flatBlock[ "name" ] =  "all"
    flatBlock[ "hierarchyLevel" ] =  1
    newblocks = [flatBlock]
    levelScores, totScore = computeModelScores( newblocks, baseargs, time.time() - t0 )
    return ret, initScore, totScore

def runHossOptimizer( blocks, baseargs, parallelMode, blocksToRun, t0, 
        idx = None ):
    origModel = baseargs['model'] # Use for loading model
    modelFileSuffix = origModel.split( "." )[-1]
    outputDir = baseargs["outputDir"]
    if len( outputDir ) > 0 and outputDir[-1] != '/':
        outputDir += "/"
    results = []
    intermed = []
    origScores, initScore = computeModelScores( blocks, baseargs, 0 )

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
                #baseargs['optfile'] = outputDir + "/" + val["optModelFile"]
            if "resultFile" in val:
                baseargs['resultFile'] = val["resultFile"]
                #baseargs['resultFile'] = outputDir + "/" + val["resultFile"]

            if idx != None: # Assign index to the optimization file names.
                baseargs['optfile'] = insertFileIdx( baseargs['optfile'], idx )
                baseargs['resultFile'] = insertFileIdx( baseargs['resultFile'], idx )

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
        ret = processIntermediateResults( score, baseargs, hl, t2 - t1, idx)
        t1 = t2
        intermed.append( ret )
        baseargs["model"] = outputDir + baseargs["optfile"] # Apply heirarchy to opt
        results.append( score )
    levelScores, totScore = computeModelScores( blocks, baseargs, time.time() - t0 )
    destfile = "{}_{:03d}.{}".format( OptModelFname, idx, modelFileSuffix ) if idx != None else "{}_000.{}".format( OptModelFname, modelFileSuffix )
    shutil.copyfile( baseargs["model"], outputDir + destfile )
    return levelScores, initScore, totScore

def worker( baseargs, exptFile ):
    score, elapsedTime, diagnostics = findSim.innerMain( exptFile, 
            scoreFunc = baseargs["scoreFunc"],
            modelFile = baseargs["model"], mapFile = baseargs["map"], 
            hidePlot = True, ignoreMissingObj = True, silent = True,
            solver = "LSODA", plots = False )
    return score

def computeModelScores( blocks, baseargs, runtime, doPrint = False ):
    MIN_BOUND = 1e-10
    MAX_BOUND = 1e6
    ret = 0.0
    nret = 0
    levelScores = []
    totScore = 0.0
    flatScore = 0.0
    flatWt = 0.0
    ed = baseargs["exptDir"]
    if len( ed ) > 0 and ed[-1] != "/":
        ed = ed + "/"
    sf = baseargs["scoreFunc"]
    pool = multiprocessing.Pool( processes = multiprocessing.cpu_count() )
    for hossLevel in blocks: # Assume blocks are in order of execution.
        lret = 0.0
        if doPrint:
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
                raise KeyError( "Missing Expt list in pathway " + expt )
            if len( exptList ) == 1:
                pathwayScore = worker( baseargs, ed + exptList[0] )
                meanPathwayScore += pathwayScore
                wt = expt[exptList[0]]["weight"]
                flatScore += pathwayScore * pathwayScore * wt
                flatWt += wt
                numPathways += 1
            elif len( exptList ) > 1:
                ret = []
                for ee in exptList:
                    ret.append( pool.apply_async(worker, args = (baseargs, ed + ee) ) )
                sumScore = 0.0
                sumWts = 0.0
                for rr, ee in zip( ret, exptList ):
                    try:
                        score = rr.get()
                        #print( score )
                    except multiprocessing.TimeoutError:
                        if baseargs["verbose"]:
                            print( "computeModelScores: timeoutError. Skipping: ", ee, flush = True )
                    else:
                        wt = expt[ee]["weight"]
                        sumScore += score * score * wt
                        sumWts += wt
                        flatScore += score * score * wt
                        flatWt += wt
                pathwayScore = np.sqrt( sumScore / sumWts )
                meanPathwayScore += pathwayScore
                numPathways += 1
            else:
                pathwayScore = -1.0
            if doPrint:
                print( "{:12s}{:.3f}   ".format(pathway, pathwayScore), end="" )
            scoreDict[pathway] = pathwayScore
        if doPrint:
            print()
        mean = meanPathwayScore/numPathways if numPathways > 0 else -1.0
        totScore += mean
        levelScores.append( scoreDict )

    if baseargs["method"] == "flat":
        # Hack to replace topScore by the equivalent from flatScore.
        totScore = np.sqrt( flatScore /flatWt ) * len( levelScores )

    if doPrint:
        print( "Final Score for {} levels in {} = {:.3f}, Time={:.2f}s".format( len( levelScores ), baseargs["model"], totScore / len( levelScores ), runtime) )
    return levelScores, totScore / len( levelScores )


#######################################################################
## Stuff for MC optimizer

def analyzeMCthreads( name, modelNum, hierarchyLevel, threadRet, numProcesses, pathwayScores, baseargs ):
    mapFile = baseargs["map"]
    scramDir = baseargs["scramDir"]
    outputDir = baseargs["outputDir"]
    for ii, rr in enumerate( threadRet ):
        # rr[0] is the return scores array, rr[1] is baseargs["model"]
        # scores array is ( results, eret, time.time, paramArgs )
        # results.x is the solution vector
        # eret =[{ "expt":e[0], "weight":1, "score": ret, "initScore": 0} for e in ev.expts ]
        optfile = rr[1].replace( scramDir, outputDir )
        fnames = { "model": rr[1], "optfile": optfile, "map": mapFile, "resultFile": "resultFile" }
        if numProcesses == 1:
            score = rr[0]
            initScore, newScore = combineScores( score[1] )
            #print( "single proc" )
        else:
            #print( "num proc = ", numProcesses )
            try:
                score = rr[0].get()
            except multiprocessing.TimeoutError:
                if baseargs["verbose"]:
                    print( "analyzeMCthreads: timeoutError. Skipping: ",ii, rr[1], flush = True )
                newScore = 1.0e9    # Ensure it is never used for next set
                # Copy the source file to the output file as a dummy.
                shutil.copyfile( rr[1], optfile )
            else:
                initScore, newScore = combineScores( score[1] )
                multi_param_minimization.saveTweakedModelFile( {}, score[3], score[0]["x"], fnames )

        print( ".", end = "", flush=True)
        pathwayScores[name].append( 
            Monte( name, rr[1], modelNum, ii, 
            hierarchyLevel, 
            newScore, 0.0 ) )
        


def wrapMultiParamMinimizer( name, optBlock, baseargs ):
    currProc = multiprocessing.current_process()
    currProc.daemon = False  # This allows nested multiprocessing.
    return multi_param_minimization.runJson( name, optBlock, baseargs )

def runMCoptimizer(blocks, baseargs, parallelMode, blocksToRun, t0):
    origModel = baseargs['model'] # Use for loading model
    mapFile = baseargs['map'] # Use for loading model
    scramRange = baseargs["scrambleRange"]
    numScram = baseargs["numScramble"]
    numTopModels = baseargs["numTopModels"]
    modelFileSuffix = origModel.split( "." )[-1]
    scramModelName = "scram"
    prefix = baseargs['scramDir']
    if prefix and (prefix[-1] != "/"):
        prefix = prefix + "/"
    outputDir = baseargs["outputDir"]
    if len( outputDir ) > 0 and outputDir[-1] != '/':
        outputDir += '/'
    intermed = []
    results = []
    numProcesses = baseargs["numProcesses"]
    if numProcesses == 0:
        numProcesses = max( multiprocessing.cpu_count()//8, 1)
    pool = multiprocessing.Pool( processes = numProcesses )
    t0 = time.time()

    # First compute the original score. It may well be better.
    origScores, initScore = computeModelScores( blocks, baseargs, 0 )
    #quit()
    startModelList = [(origModel, 0.0)] # Model name and score

    for idx, hossLevel in enumerate(blocks): # Assume blocks are in order of execution.
        hierarchyLevel = idx + 1
        pathwayScores = {}        
        optBlock = {}
        if idx == 0 and baseargs["numInitScramble"] > 0: 
            # Starting level, we have a single model to scramble
            numScramPerModel = baseargs["numInitScramble"]
        else:
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

            for imm, (mm, score) in enumerate( startModelList ):
                sname = "{}{}_{}_{}.{}".format( prefix, scramModelName, name, imm, modelFileSuffix )
                # print ( "mm = ", mm, "      SNAME = ", sname, "  Num=", numScramPerModel )
                scramParam.generateScrambled( mm, mapFile, sname, 
                    numScramPerModel, paramList, scramRange, 
                    freezeParams = baseargs['freezeParams'],
                    ignoreMissingFreeze = True )
                # Here we put in the starting model as it may be best
                if imm == 0:
                    ss = origScores[idx][name]
                    #print( "ORIG SCORE {} = {:.3f}".format( name, ss ) )
                    pathwayScores[name].append( 
                        Monte( name, mm, imm, 0, 
                        hierarchyLevel, 
                        ss, ss )
                    )
                # We do NOT try to reuse topN model, it would need 
                # calculation of expts and is too messy.
                threadRet = []
                for ii in range( numScramPerModel ):
                    baseargs["model"] = "{}{}_{}_{}_{:03d}.{}".format( 
                            prefix, scramModelName, name, 
                            imm, ii, modelFileSuffix )
                    #print( "BASEARGS = ", baseargs["model"] )
                    #newScore = runJson( name, ob, baseargs )
                    if numProcesses <= 1:
                        #print( "start single proc, t={:.3f} ".format( time.time() - t0  ) )
                        threadRet.append( ( multi_param_minimization.runJson( name, ob, baseargs ), baseargs["model"] ) )
                    else:
                        #print( "start multi proc. num proc = {}, t = {:.3f}".format( numProcesses, time.time() - t0) )
                        threadRet.append( ( pool.apply_async( wrapMultiParamMinimizer, args = ( name, ob, dict( baseargs ) ), callback = ticker ), baseargs["model"] ) )
                analyzeMCthreads( name, imm, hierarchyLevel, threadRet, numProcesses, pathwayScores, baseargs )

        if len( optBlock ) == 0:
            continue        # Nothing to see here, move along to next level

        topN = buildTopNList( pathwayScores, numTopModels )
        startModelList = []
        if baseargs["verbose"]:
            for idx, tt in enumerate( topN ):
                for name, monte in tt.items():
                    print( "\nL{}.{}: {} scores={:.5f} {:.5f}   fname= {}".format(
                        hierarchyLevel, idx, name, 
                        monte.score, monte.rankScore, monte.fname ), flush=True )

        # Build merged model.
        for idx, tt in enumerate( topN ):
            rmsScore = 0.0
            firstBlock = True
            outputModel = "{}topN_{}_{:03d}.{}".format( outputDir, hierarchyLevel, idx, modelFileSuffix )
            for name, ob in optBlock.items():
                monte = tt[name]
                rmsScore += monte.score * monte.score
                if firstBlock:
                    startModel = monte.fname.replace( baseargs["scramDir"], baseargs["outputDir"] )
                    shutil.copyfile( startModel, outputModel )
                    #print( "Copyfile ", startModel, "       ", outputModel )
                else:
                    scramParam.mergeModels( startModel, mapFile, monte.fname.replace( baseargs["scramDir"], baseargs["outputDir"] ), outputModel, ob["params"] )
                firstBlock = False
                startModel = monte.fname.replace ( baseargs["scramDir"], baseargs["outputDir"] )

            rmsScore = np.sqrt( rmsScore / len( optBlock )  )
            startModelList.append( (outputModel, rmsScore  ) )

    # Finally compute the end score. It should be a lot better.
    baseargs["model"] = "{}topN_{}_{:03d}.{}".format( outputDir, hierarchyLevel, 0, modelFileSuffix )
    print()
    finalScores, finalTotScore = computeModelScores( blocks, baseargs, time.time() - t0, doPrint = False )
    print( "{}: hossMC opt: Init Score {:.3f}, Final = {:.3f}, Time = {:.3f}s".format(
        baseargs['optfile'], initScore, finalTotScore, time.time() - t0 ) )
    for tt in range( numTopN ):
        baseargs["model"] = "{}topN_{}_{:03d}.{}".format( outputDir, hierarchyLevel, tt, modelFileSuffix )
        finalScores, finalTotScore = computeModelScores( blocks, baseargs, time.time() - t0, doPrint = False )
        print( "final Score for topN{:03d} = {:.3f}".format( 
            tt, finalTotScore ) )

#######################################################################

def main( args ):
    t0 = time.time()
    parser = argparse.ArgumentParser( description = 
            'This script orchestrates multilevel optimization, first over many individual pathways, then over the cross-talk between them, and possibly further levels.  It uses a JSON file for configuring the optimization.  Since the individual optimizations can go in parallel, the system spreads out the load into many individual multi-parameter optimization runs, which can be run in parallel.' )
    parser.add_argument( 'config', type = str, help='Required: JSON configuration file for doing the optimization.')
    parser.add_argument( '-t', '--tolerance', type = float, help='Optional: Tolerance criterion for completion of minimization' )
    parser.add_argument( '-a', '--algorithm', type = str, help='Optional: Algorithm name to use, from the set available to scipy.optimize.minimize. Options are CG, Nelder-Mead, Powell, BFGS, COBYLA, SLSQP, trust-constr. The library has other algorithms but they either require Jacobians or they fail outright. There is also L-BFGS-B which handles bounded solutions, but this is not needed here because we already take care of bounds. SLSQP works well and is the default.' )
    parser.add_argument( '-b', '--blocks', nargs='*', default=[],  help='Blocks to execute within the JSON file. Defaults to empty, in which case all of them are executed. Each block is the string identifier for the block in the JSON file.' )
    parser.add_argument( '-freeze', '--freezeParams', nargs='*', help='Space-separated list of parameters (specified as obj.field) to be frozen when doing the parameter scrambling.' )
    parser.add_argument( '-m', '--model', type = str, help='Optional: Composite model definition file. First searched in directory "location", then in current directory.' )
    parser.add_argument( '-map', '--map', type = str, help='Model entity mapping file. This is a JSON file.' )
    parser.add_argument( '-e', '--exptDir', type = str, help='Optional: Location of experiment files. Default = "./Expts"' )
    parser.add_argument( '-od', '--outputDir', type = str, help='Optional: Location of output/optimized files. Default = "./OPTIMIZED"' )
    parser.add_argument( '-sd', '--scramDir', type = str, help='Optional: Location of scrambled model files. Default = "./SCRAM"' )
    parser.add_argument( '-o', '--optfile', type = str, help='Optional: File name for saving optimized model', default = "" )
    parser.add_argument( '-meth', '--method', type = str, help='Optimization method: one of hoss, flat, initScram or hossMC. Default = hoss' )
    parser.add_argument( '-p', '--parallel', type = str, help='Optional: Define parallelization model. Options: serial, MPI, threads. Defaults to serial. MPI not yet implemented', default = "serial" )
    parser.add_argument( '-n', '--numProcesses', type = int, help='Optional: Number of blocks to run in parallel, when we are not in serial mode. Note that each block may have multiple experiments also running in parallel. Default is to take numCores/8.', default = 0 )
    parser.add_argument( '-ns', '--numScramble', type = int, help='Optional: Number of Monte Carlo samples to take by scrambling files. By default no Monte Carlo sampling will be done', default = 0 )
    parser.add_argument( '-nt', '--numTopModels', type = int, help='Optional: For Monte Carlo, number of models out of score-sorted set, to take to next stage of optimization to use as starting points for further scrambling. By default no Monte Carlo sampling will be done.', default = 0 )
    parser.add_argument( '-ni', '--numInitScramble', type = int, help='Optional: Number of initial models to generate by scrambling all parameters of initial model file. Each one of these scrambled models will then be used as the start point for a full, non-MC optimization. By default no scrambling will be done. Each optimization will be run on a different thread, up to the limit set by numProcesses. If set, this overrides numScram and numTopModels options.', default = 0 )
    parser.add_argument( '-r', '--resultFile', type = str, help='Optional: File name for saving results of optimizations as a table of scale factors and scores.', default = "" )
    parser.add_argument( '-sf', '--scoreFunc', type = str, help='Optional: Function to use for scoring output of simulation. Default: NRMS' )
    parser.add_argument( '-scr', '--scrambleRange', type = float, help='Optional, used only when doing Monte Carlo sampling: Range for scrambling model parameters. Default: 2.0', default = 2.0 )
    parser.add_argument( '--solver', type = str, help='Optional: Numerical method to use for ODE solver. Ignored for HillTau models. Default = "LSODA".')
    parser.add_argument( '-to', '--timeout', type = float, help='Optional. TimeOut in seconds to stop optimization run which is taking too long. Default:No timeout' )
    parser.add_argument( '-v', '--verbose', action="store_true", help="Flag: default False. When set, prints all sorts of warnings and diagnostics.")
    parser.add_argument( '-st', '--show_ticker', action="store_true", help="Flag: default False. Prints out ticker as optimization progresses.")
    args = parser.parse_args( args )

    baseargs, config = loadConfig( args )
    blocks = config["HOSS"]

    if baseargs["method"] == "hoss":
        ret, initScore, finalScore = runHossOptimizer( blocks, baseargs, args.parallel, args.blocks, t0 )
        print( "{}: hoss: Init Score {:.3f}, Final = {:.3f}, Time = {:.3f}s".format( baseargs["model"], initScore, finalScore, time.time() - t0 ) )
    elif baseargs["method"] == "flat":
        ret, initScore, finalScore = runFlatOptimizer( blocks, baseargs, args.parallel, args.blocks, t0 )
        print( "{}: flat: Init Score {:.3f}, Final = {:.3f}, Time = {:.3f}s".format( baseargs["model"], initScore, finalScore, time.time() - t0 ) )
    elif baseargs["method"] == "initScram":
        if baseargs["numInitScramble"] >= 5:
            initScore, finalScore = runInitScramThenOptimize( blocks, baseargs, t0 )
            print( "{}: initScramble: Init Score {:.3f}, Final = {:.3f}, Time = {:.3f}s".format( baseargs["model"], initScore, finalScore, time.time() - t0 ) )
        else:
            raise ValueError( "numInitScramble must be >= 5 in initScram optimization" )
    elif baseargs["method"] == "hossMC":
        if baseargs["numScramble"] >= 5 and baseargs["numTopModels"] > 0 and baseargs["numInitScramble"] >= 5:
            runMCoptimizer(blocks, baseargs, args.parallel, args.blocks, t0)
        else:
            raise ValueError( "numInitScramble and numScramble  should be >= 5 and numTopModels > 0 in hossMC optimization" )
    else:
        assert( 0 )


#######################################################################
# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
    main( sys.argv[1:] )
