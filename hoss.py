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
import multi_param_minimization


def runJson( key, val, baseargs ):
    ''' a dummy function for testing '''
    for i, j in baseargs.items():
        print( "{}: {}".format( i, j ), end = "\t\t"  )
    print( "\n KV = {}: {}".format( key, val ) )
    return "foo"


def processIntermediateResults( retvec, baseargs, t0 ):
    fp = open( baseargs["resultfile"], "w" )
    optfile = baseargs["optfile"]
    levelIdx = int(optfile[-6:-5]) # Assume we have only levels 0 to 9.
    totScore = 0.0
    for (results, eret, optTime, paramArgs) in retvec:
        multi_param_minimization.analyzeResults( fp, True, results, paramArgs, eret, optTime )
        totScore += results.fun
    print( "\n-Level {} -- Mean Score = {:.3f} -- Total Time: {:.3f} s -------- ".format( levelIdx, totScore / len( retvec ), t0 ) )

    fnames = { "model": baseargs["model"], "optfile": optfile, "map": baseargs["map"], "resultfile": baseargs["resultfile"] }
    pargs = []
    rargs = []
    # Catenate all the changed params and values.
    for (results, eret, optTime, paramArgs) in retvec:
        rargs.extend( results.x )
        pargs.extend( paramArgs )
    multi_param_minimization.saveTweakedModelFile( {}, pargs, rargs, fnames )

def processFinalResults( results, baseargs, t0 ):
    print( "\n-Completed mulitlevel optimzation -- Total Time: {:.3f} s -------- ".format( t0 ) )

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser( description = 
            'This script orchestrates multilevel optimization, first over many individual pathways, then over the cross-talk between them, and possibly further levels.  It uses a JSON file for configuring the optimization.  Since the individual optimizations can go in parallel, the system spreads out the load into many individual multi-parameter optimization runs, which can be run in parallel.' )
    parser.add_argument( 'config', type = str, help='Required: JSON configuration file for doing the optimization.')
    parser.add_argument( '-t', '--tolerance', type = float, help='Optional: Tolerance criterion for completion of minimization', default = 1e-4 )
    parser.add_argument( '-a', '--algorithm', type = str, help='Optional: Algorithm name to use, from the set available to scipy.optimize.minimize. Options are CG, Nelder-Mead, Powell, BFGS, COBYLA, SLSQP, trust-constr. The library has other algorithms but they either require Jacobians or they fail outright. There is also L-BFGS-B which handles bounded solutions, but this is not needed here because we already take care of bounds. SLSQP works well and is the default.', default = "SLSQP" )
    parser.add_argument( '-b', '--blocks', nargs='*', default=[],  help='Blocks to execute within the JSON file. Defaults to empty, in which case all of them are executed. Each block is the string identifier for the block in the JSON file.' )
    parser.add_argument( '-m', '--model', type = str, help='Optional: Composite model definition file. First searched in directory "location", then in current directory.' )
    parser.add_argument( '-map', '--map', type = str, help='Model entity mapping file. This is a JSON file.' )
    parser.add_argument( '-e', '--exptDir', type = str, help='Optional: Location of experiment files.', default = "./Expts" )
    parser.add_argument( '-o', '--optfile', type = str, help='Optional: File name for saving optimized model', default = "" )
    parser.add_argument( '-p', '--parallel', type = str, help='Optional: Define parallelization model. Options: serial, MPI, threads. Defaults to serial. MPI not yet implemented', default = "serial" )
    parser.add_argument( '-r', '--resultfile', type = str, help='Optional: File name for saving results of optimizations as a table of scale factors and scores.', default = "" )
    parser.add_argument( '-sf', '--scoreFunc', type = str, help='Optional: Function to use for scoring output of simulation.', default = "NRMS" )
    parser.add_argument( '-v', '--verbose', action="store_true", help="Flag: default False. When set, prints all sorts of warnings and diagnostics.")
    parser.add_argument( '-st', '--show_ticker', action="store_true", help="Flag: default False. Prints out ticker as optimization progresses.")
    args = parser.parse_args()

    try:
        with open( args.config ) as json_file:
            config = json.load( json_file )
    except IOError:
        print( "Error: Unable to find HOSS config file: " + args.config )
        quit()

    blocks = config["HOSS"]
    #basekeys = ["model", "map", "exptDir", "scoreFunc", "tolerance"]
    basekeys = vars( args ).keys()
    baseargs = vars( args )
    for key, val in config.items():
        if key in basekeys:
            # Use config file setting if not passed in on command line.
            if baseargs[key] == "" or baseargs[key] == None:
                baseargs[key] = val

    # Build up optimization blocks. Within a block we assume that the
    # individual optimizations do not interact and hence can run in 
    # parallel. Once a block is done its values are consolidated and used
    # for the next block.
    assert( 'model' in baseargs )
    assert( 'resultfile' in baseargs )
    origModel = baseargs['model'] # Use for loading model
    optModel = baseargs['optfile'] # Use for final model save
    optResults = baseargs['resultfile'] # Use for final results
    results = []
    for hossLevel in blocks: # Assume blocks are in order of execution.
        optBlock = {}
        hl = hossLevel["hierarchyLevel"]
        # Specify intermediate model and result files
        baseargs['optfile'] = "./_optModel{}.json".format( hl )
        baseargs['resultfile'] = "./_optResults{}.txt".format( hl )
        for key, val in hossLevel.items():
            if key == "name" or key == "hierarchyLevel":
                continue
            # Either run all items or run named items in block.
            if args.blocks == [] or key in args.blocks:
                optBlock[ key] = val

        # Now we have a block to optimize, use suitable method to run it.
        # We can run items in a block in any order, but the whole block
        # must be wrapped up before we go to the next level of heirarchy.
        t1 = time.time()
        if args.parallel == "serial":
            score = runOptSerial( optBlock, baseargs )
        elif args.parallel == "threads":
            score = runOptThreads( optBlock, baseargs )
        elif args.parallel == "MPI":
            score = runOptMPI( optBlock, baseargs )
        t2 = time.time()
        # This saves the scores and the intermediate opt file, to use for
        # next level of optimization.
        processIntermediateResults( score, baseargs, t2 - t1 )
        t1 = t2
        baseargs["model"] = baseargs["optfile"] # Apply heirarchy to opt
        results.append( score )
    processFinalResults( results, baseargs, time.time() - t0  )

def runOptSerial( optBlock, baseargs ):
    score = []
    for name, pathway in optBlock.items():
        score.append( multi_param_minimization.runJson( name, pathway, baseargs ) )
        print( "Run OptSerial for {} gave {}".format(name, score[-1][0].fun) )

    return score

def runOptThreads( optModel, itemName, item, baseargs ):
    score = []
    pool = ThreadPool( processes = len( optBlocks ) )
    ret = [pool.apply_async( multi_param_minimization.runJson, args=( key, val, baseargs ) ) for key, val in  optBlocks.items() ]
    score.append( multi_param_minimization.runJson( itemName, item, baseargs ) )

    return score, optModel

def runOptMPI( optModel, itemName, item, baseargs ):
    score = []
    score.append( multi_param_minimization.runJson( itemName, item, baseargs ) )

    return score, optModel
        
# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
    main()
