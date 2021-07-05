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
import argparse
import json
import time
from multiprocessing.pool import ThreadPool
from mpi4py import MPI
import multi_param_minimization

# This is used to monitor the progress of optimization blocks
class BlockStatus:
    def __init__( self, numExpts, rank ):
        self.myBlock = rank
        self.numExpts = numExpts
        self.isDone = False
        self.numCalls = 0
        self.numIter = 0
        self.exptStatus = [0] * numExpts

    def nodeTicker( self, result ):
        self.isDone = True

    def procTicker( self, procNumber ):
        assert( procNumber < self.numExpts )
        self.numCalls += 1
        self.exptStatus[ procNumber ] += 1
        if self.numCalls % self.numExpts == 0:
            self.numIter += 1
            #print( "rank{}, nexpt{}, {}".format( self.myBlock, self.numExpts, self.exptStatus ) )
            assert( self.exptStatus == [1] * self.numExpts )
            self.exptStatus = [0] * self.numExpts

def baseMain():
    parser = argparse.ArgumentParser( description = 
            'This script orchestrates multilevel optimization, first over many individual pathways, then over the cross-talk between them, and possibly further levels.  It uses a JSON file for configuring the optimiztaion.  Since the individual optimizations can go in parallel, the system spreads out the load into many individual multi-parameter optimization runs, which can be run in parallel. For now it requres that the # of nodes assigned == # of parameter blocks to optimize, and it sets them all off at the outset.' )
    parser.add_argument( 'config', type = str, help='Required: JSON configuration file for doing the optimization.')
    parser.add_argument( '-t', '--tolerance', type = float, help='Optional: Tolerance criterion for completion of minimization', default = 1e-4 )
    parser.add_argument( '-l', '--level', type = int, help='Optional: Level of HOSS hierarchy to optimize. Default = 1', default = 1 )
    parser.add_argument( '-b', '--blocks', nargs='*', default=[],  help='Blocks to execute within the JSON file. Defaults to empty, in which case all of them are executed. Each block is the string identifier for the block in the JSON file.' )
    parser.add_argument( '-o', '--optfile', type = str, help='Optional: File name for saving optimized model', default = "allOpt.g" )
    parser.add_argument( '-r', '--resultfile', type = str, help='Optional: File name for saving results of optimizations as a table of scale factors and scores.', default = "allOpt.txt" )
    parser.add_argument( '-v', '--verbose', action="store_true", help="Flag: default False. When set, prints all sorts of warnings and diagnostics.")
    args = parser.parse_args()

    try:
        with open( args.config ) as json_file:
            config = json.load( json_file )
    except IOError:
        print( "Error: Unable to find HOSS config file: " + args.config )
        quit()

    blocks = config["HOSS"]
    basekeys = ["model", "map", "exptDir", "scoreFunc", "tolerance"]
    baseargs = {"exptDir": "./", "tolerance": 1e-4}
    for key, val in config.items():
        if key in basekeys:
            baseargs[key] = val
    return blocks, baseargs, args


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    t0 = time.time()
    if rank == 0:
        blocks, baseargs, args = baseMain()
        argvec = []
        for hossLevel in blocks:
            if hossLevel["hierarchyLevel"] == args.level:
                for key, val in hossLevel.items():
                    if key == "name" or key == "hierarchyLevel":
                        continue
                    if args.blocks == [] or key in args.blocks:
                        argvec.append( (key, val ) )
                        #print( "AV key = {}".format( key ) )
        baseargs["numBlocks"] = len( argvec )
        print( "Num Nodes = {}, numBlocks = {}".format( size, len( argvec)))
    else:
        baseargs = None
        argvec = None

    baseargs = comm.bcast( baseargs, root = 0 )

    #print( "Rank {} in size {}, numBlocks = {}".format( rank, size, len( argvec ) ) )
    argitem = comm.scatter( argvec, root = 0 )
    blockStatus = BlockStatus( len( argitem[1]["expt"] ), rank )
    print( "running MINIM on rank {} with {}".format( rank, argitem[0] ) )
    pool = ThreadPool( processes = 1 )
    ret = pool.apply_async(optRun, args = ( argitem, baseargs), callback = blockStatus.nodeTicker)
    #ret.get()
    isComplete = False
    numPoll = 0
    while not isComplete:
        numCalls, numIter, numExpts = multi_param_minimization.extractStatus()
        if numExpts == 0:
            time.sleep(1)
            continue
        #print( "baseargs[numBlocks] = {},  blockStatus.numExpts = {}, numExpts = {}".format ( baseargs["numBlocks"], blockStatus.numExpts, numExpts )) 
        assert( blockStatus.numExpts == numExpts )
        blockStatus.numCalls = numCalls
        blockStatus.numIter = numIter
        retvec = comm.gather( blockStatus, root = 0 )
        # Analyse retvec on node 0 to check for completeness
        if comm.rank == 0:
            isComplete = testCompleteness( retvec )
            printStatus( retvec, numPoll )
            numPoll += 1
        isComplete = comm.bcast( isComplete, root = 0 )
        time.sleep( 1 )
    minimResult = ret.get()
    #processLocalResult( minimResult, argitem, baseargs )
    retvec = comm.gather( minimResult, root = 0 )
    if rank == 0:
        processResults( retvec, args, baseargs, t0 )
    return 0

def optRun( argitem, baseargs ): # Optimizer call. 
    minimResult = multi_param_minimization.runJson(argitem[0], argitem[1], baseargs )
    processLocalResults( minimResult, argitem, baseargs )
    return minimResult

def testCompleteness( retvec ):
    return ( sum( [ i.isDone for i in retvec ] ) == len( retvec ) )

def printStatus( retvec, numPoll ):
    if numPoll % 25 == 0:
        print("\nNode:", end = "")
        for i in range( len( retvec ) ):
            print( "{:2d}".format( i ), end = " " )
        print( " ## ", end = "" )
        for i in range( len( retvec ) ):
            print( "{:2d}".format( i ), end = " " )
    print("\n{:3d}:".format( numPoll ), end = " ")
    for i in retvec:
        print( "{:2d}".format( i.numIter ), end = " " )
    print( " ## ", end = "" )
    for i in retvec:
        if i.isDone:
            print( " .", end = " " )
        else:
            #print( "{:2d}".format( i.numExpts - i.numCalls % i.numExpts ), end = " " )
            print( "{:2d}".format( i.numCalls % 100 ), end = " " )


class Targs:
    def __init__( self, model, optmodel, objMap ):
        self.model = model
        self.optfile = optmodel
        self.map = objMap

def processResults( retvec, args, baseargs, t0 ):
    fp = ""
    dumpData = False
    if len( args.resultfile ) > 0:
        fp = open( args.resultfile, "w" )
        dumpData = True

    print( "\n----------- Completed entire HOSS run in {:.3f} sec ---------- ".format(time.time() - t0 ) ) 
    for (results, eret, optTime, paramArgs) in retvec:
        multi_param_minimization.analyzeResults( fp, dumpData, results, paramArgs, eret, optTime )
    if len( args.optfile ) > 2: # at least foo.g
        targs = Targs( baseargs["model"], args.optfile, baseargs["map"] )
        pargs = []
        rargs = []
        # Catenate all the changed params and values.
        for (results, eret, optTime, paramArgs) in retvec:
            rargs.extend( results.x )
            pargs.extend( paramArgs )
        multi_param_minimization.saveTweakedModelFile( targs, pargs, rargs )

def processLocalResults( ret, argitem, baseargs ):
    fp = open( "checkpoint_" + argitem[0] + ".txt", "a+" )
    results, eret, optTime, paramArgs = ret
    print( "\n----------- Completed local HOSS run in {:.3f} sec ---------- ".format( optTime ) ) 
    multi_param_minimization.analyzeResults( fp, True, results, paramArgs, eret, optTime )
    optfile = "checkpoint_" + argitem[0] + ".g"
    targs = Targs( baseargs["model"], optfile, baseargs["map"] )
    pargs = []
    rargs = []
    rargs.extend( results.x )
    pargs.extend( paramArgs )
    multi_param_minimization.saveTweakedModelFile( targs, pargs, rargs )


def runJson( key, val, baseargs ):
    for i, j in baseargs.items():
        print( "{}: {}".format( i, j ), end = "\t\t"  )
    print( "\n KV = {}: {}".format( key, val ) )
    return "foo"

        
# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
    main()
