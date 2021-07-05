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


results = []
def logResult(result):
    # This is called whenever pool(i) returns a result.
    # results is modified only by the main process, not the pool workers.
    results.append(result)

def main():
    parser = argparse.ArgumentParser( description = 
            'This script displays simulation results compared with experiments for all the experiments defined in the optimization configuration file (a json file). It runs in serial.' )
    parser.add_argument( 'config', type = str, help='Required: JSON configuration file, typically same file as used for doing the optimization.')
    parser.add_argument( '-b', '--blocks', nargs='*', default=[],  help='Blocks to display within the JSON file. Defaults to empty, in which case all of them are display. Each block is the string identifier for the block in the JSON file.' )
    parser.add_argument( '-m', '--model', type = str, help='Optional: File name for alternative model to run.', default = "" )
    parser.add_argument( '-map', '--map', type = str, help='Optional: File name for alternative model mapfile.', default = "" )
    parser.add_argument( '-v', '--verbose', action="store_true", help="Flag: default False. When set, prints all sorts of warnings and diagnostics.")
    args = parser.parse_args()

    try:
        with open( args.config ) as json_file:
            config = json.load( json_file )
    except IOError:
        print( "Error: Unable to find HOSS config file: " + args.config )
        quit()

    model = config["model"]
    if args.model != "":
        model = args.model
    mapfile = config["map"]
    if args.map != "":
        mapfile = args.map
    scoreFunc = config["scoreFunc"]
    b = args.blocks
    edict = {}
    blocks = config["HOSS"]
    if len( b ) == 0:
        for bl in blocks:
            for key, val in bl.items():
                if key not in ("name", "hierarchyLevel" ):
                    expt = val.get( "expt" )
                    if expt:
                        edict[key] = [e for e in expt]
    else: 
        for bl in blocks:
            for i in b:
                val = bl.get( i )
                if val:
                    expt = val.get( "expt" )
                    if expt:
                        edict[i] = [e for e in expt]

    #pool = Pool( processes = len( edict ) )
    ret = []
    exptDir = config["exptDir"]
    for key, val in edict.items(): # Iterate through blocks
        for f in val: # Iterate through each expt (tsv or json) fname
            fname = exptDir + "/" + f
            p = multiprocessing.Process( target = findSim.innerMain, args = ( fname,), kwargs = dict( scoreFunc = scoreFunc, modelFile = model, mapFile = mapfile, hidePlot = False, ignoreMissingObj = True, silent = not args.verbose  ) )
            p.start()
            #ret.append( pool.apply_async( findSim.innerMain, (fname,), dict( modelFile = model, mapFile = mapfile, hidePlot = False, silent = not args.verbose  ), callback = logResult ) )

    #time.sleep(5)
    #pool.close()
    #pool.join()
    #print(results)
    #finished = [i.get() for i in ret]


    #return model, mapfile, edict
        
# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
    main()
