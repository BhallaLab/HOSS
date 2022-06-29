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


def processResults( retvec, args, baseargs, t0 ):
    fp = ""
    dumpData = False
    if len( args.resultfile ) > 0:
        fp = open( args.resultfile, "w" )
        dumpData = True

    print( "\n----------- Completed entire HOSS run in {:.3f} sec ---------- ".format(time.time() - t0 ) ) 
    totScore = 0.0
    for (results, eret, optTime, paramArgs) in retvec:
        multi_param_minimization.analyzeResults( fp, dumpData, results, paramArgs, eret, optTime )
        totScore += results.fun
    print( "\n----------- Mean Score = {:.3f} -- Total Time: {:.3f} s -------- ".format( totScore / len( retvec ), time.time() - t0 ) ) 

    if len( args.optfile ) > 2: # at least foo.g
        fnames = { "model": baseargs["model"], "optfile": args.optfile, "map": baseargs["map"], "resultfile": args.resultfile }
        pargs = []
        rargs = []
        # Catenate all the changed params and values.
        for (results, eret, optTime, paramArgs) in retvec:
            rargs.extend( results.x )
            pargs.extend( paramArgs )
        multi_param_minimization.saveTweakedModelFile( args, pargs, rargs, fnames )


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
            if baseargs[key] == "" or baseargs[key] == None:
                baseargs[key] = val

    optBlocks = {}
    for hossLevel in blocks:
        if hossLevel["hierarchyLevel"] == 1:
            for key, val in hossLevel.items():
                if key == "name" or key == "hierarchyLevel":
                    continue
                if args.blocks == [] or key in args.blocks:
                    optBlocks[key] = val
    if len( optBlocks ) == 0:
        print( "Error: No matching optimization blocks found" )
        quit()

    if args.parallel == "serial":
        score = []
        for key, val in optBlocks.items():
            print( "\n===================== Serial run for block: ", key , "=========================" )
            score.append( multi_param_minimization.runJson( key, val, baseargs ) )
        #score = [multi_param_minimization.runJson( key, val, baseargs ) for key, val in  optBlocks.items() ]
    elif args.parallel == "MPI":
        print( "MPI not yet running" )
        score = [0.0] * len( optBlocks )
    elif args.parallel == "threads":
        print ("RUNNNING THREADS" )
        pool = ThreadPool( processes = len( optBlocks ) )
        ret = [pool.apply_async( multi_param_minimization.runJson, args=( key, val, baseargs ) ) for key, val in  optBlocks.items() ]
        #ret = [pool.apply_async( runJson, args=( key, val, baseargs ) ) for key, val in  optBlocks.items() ]
        # return from runJson is (results, ev, initScore), paramArgs
        score = [ i.get() for i in ret ]

    processResults( score, args, baseargs, t0 )    
        
# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
    main()
