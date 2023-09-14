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
 * File:            dispAllExpts.py
 * Description:
 * Author:          Upinder S. Bhalla
 * E-mail:          bhalla@ncbs.res.in
 ********************************************************************/

/**********************************************************************
** This program is part of 'HOSS', the
** Hierarchical Optimization for Systems Simulations
**           copyright (C) 2019-2020 Upinder S. Bhalla. and NCBS
**********************************************************************/

This script displays simulation results compared with experiments for all
the experiments defined in the optimization configuration file 
(a json file). It runs in serial.
'''

from __future__ import print_function
import argparse
import findSim
#from multiprocessing import Pool
import multiprocessing
import json
import time
import numpy as np


results = []
def logResult(result):
    # This is called whenever pool(i) returns a result.
    # results is modified only by the main process, not the pool workers.
    results.append(result)

def worker( fname, returnDict, scoreFunc, modelFile, mapFile, silent, solver, plots, hidePlot, weight ):
    score, elapsedTime, diagnostics = findSim.innerMain( fname, scoreFunc = scoreFunc, modelFile = modelFile, mapFile = mapFile, hidePlot = hidePlot, ignoreMissingObj = True, silent = silent, solver = solver, plots = plots )
    returnDict[fname] = (score, weight)

def main():
    parser = argparse.ArgumentParser( description = 
            'This script displays simulation results compared with experiments for all the experiments defined in the optimization configuration file (a json file). It runs in serial.')
    parser.add_argument( 'config', type = str, help='Required: JSON configuration file, typically same file as used for doing the optimization.')
    parser.add_argument( '-b', '--blocks', nargs='*', default=[],  help='Blocks to display within the JSON file. Defaults to empty, in which case all of them are display. Each block is the string identifier for the block in the JSON file.' )
    parser.add_argument( '-m', '--model', type = str, help='Optional: File name for alternative model to run.', default = "" )
    parser.add_argument( '-map', '--map', type = str, help='Optional: File name for alternative model mapfile.', default = "" )
    parser.add_argument( '-e', '--exptDir', type = str, help='Optional: Location of experiment files.' )
    parser.add_argument( '-sf', '--scoreFunc', type = str, help='Optional: Function to use for scoring output of simulation.', default = "" ) 
    parser.add_argument( '--solver', type = str, help='Optional: Numerical method to use for ODE solver. Ignored for HillTau models. Default = "lsoda".', default = "lsoda" )
    parser.add_argument( '-p', '--plot', type = str, nargs = '*', help='Optional: Plot specified fields as time-series', default = "" )
    parser.add_argument( '-hp', '--hidePlot', action="store_true", help="Flag: default False. When set, turns off plots. Useful to view weighted scores.")
    parser.add_argument( '-v', '--verbose', action="store_true", help="Flag: default False. When set, prints all sorts of warnings and diagnostics.")
    args = parser.parse_args()
    totScore = innerMain( args )
    if args.verbose:
        print( "Final Score = {:.3f}".format( totScore ) )


def innerMain( args ):
    try:
        with open( args.config ) as json_file:
            config = json.load( json_file )
    except IOError:
        print( "Error: Unable to find HOSS config file: " + args.config )
        quit()

    requiredDefaultArgs = {
            "scoreFunc": "NRMS",
            "solver": "LSODA",
            "exptDir": "./Expts"
        }
    baseargs = vars( args )
    for key, val in requiredDefaultArgs.items():
        if baseargs[key]:   # Command line given
            continue
        elif key in config: # Set up in config file
            baseargs[key] = config[key]
        else:
            baseargs[key] = val

    model = config["model"]
    if args.model != "":
        model = args.model
    mapfile = config["map"]
    if args.map != "":
        mapfile = args.map
    b = args.blocks
    edict = {}
    blocks = config["HOSS"]
    if len( b ) == 0:
        for bl in blocks:
            for key, val in bl.items():
                if key not in ("name", "hierarchyLevel" ):
                    expt = val.get( "expt" )
                    if expt:
                        #edict[key] = [(e, expt[e]["weight"]) for e in expt]
                        edict[key] = expt
    else: 
        for bl in blocks:
            for i in b:
                val = bl.get( i )
                if val:
                    expt = val.get( "expt" )
                    if expt:
                        #edict[i] = [(e, expt[e]["weight"] ) for e in expt]
                        edict[i] = expt

    #pool = Pool( processes = len( edict ) )
    ret = []
    manager = multiprocessing.Manager()
    totScore = 0.0
    numTot = 0
    for blockName, val in edict.items(): # Iterate through blocks
        jobs = []
        returnDict = manager.dict()
        for f in val: # Iterate through each expt (tsv or json) fname
            fname = baseargs["exptDir"] + "/" + f
            p = multiprocessing.Process( target = worker, args = ( fname, returnDict, ), kwargs = dict( scoreFunc = baseargs["scoreFunc"], modelFile = model, mapFile = mapfile, silent = not args.verbose, solver = baseargs["solver"], plots = args.plot, hidePlot = args.hidePlot, weight = val[f]["weight"] ) )
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        sumScore = 0.0
        sumWts = 0.0
        for key, (score, wt) in returnDict.items():
            #print( "{:50s}{:.4f}".format( key, score ) )
            sumScore += score * score * wt
            sumWts += wt
        pathwayScore = np.sqrt( sumScore / sumWts )
        totScore += pathwayScore
        numTot += 1
        if args.verbose or not args.hidePlot:
            print( "{:20s} Score = {:.3f}".format( blockName, pathwayScore ) )
    return totScore / numTot if numTot > 0 else -1.0

            #ret.append( pool.apply_async( findSim.innerMain, (fname,), dict( modelFile = model, mapFile = mapfile, hidePlot = False, silent = not args.verbose  ), callback = logResult ) )

    #time.sleep(5)
    #pool.close()
    #pool.join()
    #print(results)
    #finished = [i.get() for i in ret]


    #return model, mapfile, edict
        
# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
    multiprocessing.set_start_method( 'spawn' )
    main()
