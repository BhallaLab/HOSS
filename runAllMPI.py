
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
 * File:            runAllMPI.py
 * Description:     Run a set of expt.json files in parallel using MPI.
 * Author:          Upinder S. Bhalla
 * E-mail:          bhalla@ncbs.res.in
 ********************************************************************/

/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
**           copyright (C) 2003-2018 Upinder S. Bhalla. and NCBS
**********************************************************************/

This script runs the findSim program on all expt.json files in the specified
directory, computes their scores, and prints out basic stats of the scores.
It can do this in parallel using Python's mpi4py library.
'''

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import time
import findSim
from mpi4py import MPI

MY_MPI_TAG = 21298

class Job:
    def __init__( self, rank, resultIndex, fname ):
        self.rank = rank
        self.resultIndex = resultIndex
        self.fname = fname
        self.complete = False
        self.req = ''
        self.buf = np.empty(4, dtype=np.float64)


class Result:
    def __init__( self, fname, weight ):
        self.fname = fname
        self.weight = weight
        self.score = 0.0
        self.runtime = 0.0


def enumerateFindSimFiles( exptlist ):
    if os.path.isdir( exptlist ):
        if exptlist[-1] != '/':
            exptlist += '/'
        fnames = [ (exptlist + i) for i in os.listdir( exptlist ) if i.endswith( ".json" )]
        return fnames, [1.0] * len( fnames )
    elif os.path.isfile( exptlist ):
        fnames = []
        weights = []
        with open( exptlist, "r" ) as fp:
            for line in fp:
                if len( line ) <= 2:
                    continue
                if line[0] == '#':
                    continue
                f,w = line.split()
                fnames.append( f )
                weights.append( float( w ) )
        return fnames, weights
    else:
        print( "Error: Unable to find file or directory at " + exptlist )
        quit()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if ( rank == 0 ):
        innerMain( comm )
    else:
        worker( comm )

def launchNewJob( j, comm, args, results, currResult, fnames ):
    # If runs still pending, launch another job.
    if currResult < len( fnames ):
        j.resultIndex = currResult
        mpiargs = [results[currResult].fname, args.model, args.map, args.verbose]
        comm.send( mpiargs, dest = j.rank, tag = MY_MPI_TAG )
        j.req = comm.Irecv( j.buf, source=j.rank, tag = MY_MPI_TAG )
        j.startTime = time.time()
        j.fname = results[currResult].fname
        currResult += 1
    else:
        if not j.complete:
            # Tell worker to quit.
            comm.send( [], dest = j.rank, tag = MY_MPI_TAG )
        j.complete = True
        j.req = ""
    return currResult

def innerMain( comm ):
    t0 = time.time()
    parser = argparse.ArgumentParser( description = 'Wrapper script to run a lot of FindSim evaluations in parallel.' )

    parser.add_argument( 'exptlist', type = str, help='Required: Files to run. Can be a directory or a filename. If directory then run all the files (in FindSim json format) in it. If filename, then each line is pair of "<fname>.json weight". Preceding # says to ignore line.')
    parser.add_argument( '-m', '--model', type = str, help='Optional: Composite model definition file. First searched in directory "exptlist", then in current directory.', default = "" )
    parser.add_argument( '-map', '--map', type = str, help='Model entity mapping file. This is a JSON file.', default = "" )
    parser.add_argument( '-s', '--scale_param', nargs=3, default=[],  help='Scale specified object.field by ratio.' )
    parser.add_argument( '-v', '--verbose', action="store_true", help="Flag: default False. When set, prints all sorts of warnings and diagnostics.")
    parser.add_argument( '-to', '--timeout', type = float, help="Maximum time in seconds to permit for any evaluation. Default 300 sec. Sluggards will get a score of -1. Current implementation is faulty. It does indeed wrap up, but needs a control-C to kill the MPI process.", default = 300.0 )
    args = parser.parse_args()
    exptlist = args.exptlist
    if exptlist[-1] != '/':
        exptlist += '/'
    if os.path.isfile( exptlist + args.model ):
        modelFile = exptlist + args.model
    elif args.model == "" or os.path.isfile( args.model ):
        modelFile = args.model
    else:
        print( "Error: runAllParallel: Unable to find model file '{}'".format( args.model ) )
        quit()

    fnames, weights = enumerateFindSimFiles( args.exptlist )

    num = min( len(fnames), comm.size - 1 )
    results = [ Result( f, w ) for f, w in zip ( fnames, weights ) ]
    # Job( rank, resultIndex, fname )
    jobs = [ Job( i+1, i, fnames[i] ) for i in range( num ) ]

    # Set off the first lot.
    for j in jobs:
        mpiargs = [j.fname, modelFile, args.map, args.verbose]
        comm.send( mpiargs, dest = j.rank, tag = MY_MPI_TAG )
        j.req = comm.Irecv( j.buf, source = j.rank, tag = MY_MPI_TAG )
        j.startTime = time.time()

    for k in range( 1+len( jobs ), comm.size ):
        mpiargs = []
        comm.send( mpiargs, dest = k, tag = MY_MPI_TAG )

    # Now tick along monitoring responses and launching new jobs when done
    currResult = num
    numDone = 0
    if True:
        # CurrTime  currentResult   numRunning
        print( "CurrT\tnumDone\t\tnumRunning" )
    while True:
        numComplete = 0
        for j in jobs:
            if j.complete:
                numComplete += 1
                continue
            if j.req.Test():
                numDone += 1
                j.req.Wait()
                r = results[j.resultIndex]
                r.score = j.buf[0]
                r.runtime = j.buf[1]
                # If runs still pending, launch another job.
                currResult = launchNewJob( j, comm, args, results, currResult, fnames )
            else: # Handle timeouts. This gets the main job to wrap up but the nodes keep on working, so MPI doesn't terminate properly. Leave for now.
                if time.time() - j.startTime > args.timeout:
                    j.req.Cancel() 
                    r = results[j.resultIndex]
                    r.score = -1
                    r.runtime = args.timeout
                    currResult = launchNewJob( j, comm, args, results, currResult, fnames )
        if int( 10 * (time.time() - t0) ) % 10 == 0:
            # CurrTime  numDone   numRunning    JSON
            expt = ""
            if len( results ) - numDone <= 2:
                for k in jobs:
                    if not k.complete:
                        expt += k.fname + "\t"
            print( "{:.2f}\t{}\t\t{}\t{}".format(time.time() - t0, numDone, len( jobs ) - numComplete, expt ) )
        if numComplete == len( jobs ):
            break
        time.sleep( 0.2 )

    numGood = 0
    sumScore = 0.0
    sumWts = 0.0
    print( "\n---------Completed---------" )
    print( "{:40s}  {:6s}  {:6s}".format( "Expt Name", "score", "runtime" ))
    for i, j, w in zip( fnames, results, weights ):
        print( "{:40s}  {:.4f}  {:.4f}".format( i, j.score, j.runtime ) )
        if j.score >= 0:
            numGood += 1
            sumScore += j.score * w
            sumWts += w
    if sumWts <= 0.0:
        sumWts = 1.0
    print( "Weighted Score of {:.0f} good of {:.0f} runs = {:.3f}. Runtime = {:.3f} sec".format( numGood, len( fnames ), sumScore / sumWts, time.time() - t0 ) )

def worker( comm ):
    sendbuf = np.empty(4, dtype=np.float64)
    while True:
        mpiargs = comm.recv( source = 0, tag = MY_MPI_TAG )
        if ( len( mpiargs ) != 4 ):
            break;
        exptFile = mpiargs[0]
        modelFile = mpiargs[1]
        mapFile = mpiargs[2]
        verbose = mpiargs[3]

        ret = findSim.innerMain( exptFile, modelFile = modelFile, hidePlot = True, silent = not verbose, optimizeElec = False, mapFile = mapFile)
        sendbuf[0], sendbuf[1] = ret
        comm.Send( sendbuf, dest = 0, tag = MY_MPI_TAG )
        
# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
    main()
