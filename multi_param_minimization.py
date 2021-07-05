
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
 * File:            multi_param_minimization.py
 * Description:
 * Author:          Upinder S. Bhalla
 * E-mail:          bhalla@ncbs.res.in
 ********************************************************************/

/**********************************************************************
** This program is part of 'FindSim', the
** Framework for Integrating Neuronal Data and SIgnaling Models
**           copyright (C) 2003-2018 Upinder S. Bhalla. and NCBS
**********************************************************************/

This script does a multi-dimensional minimization on the model. It runs the 
findSim program on all expt.json files in the specified directory with 
modifications of the selected parameters. It computes the weighted score 
for each run as the return value for the minimization function. While the
BGFR algorithm is serial, there are lots of indvidual expt calculations 
for each step that can be done in parallel.
'''

from __future__ import print_function
import numpy as np
from scipy import optimize
import argparse
import errno
import os
import sys
import time
import json
import findSim
from multiprocessing import Pool

defaultScoreFunc = "(expt-sim)*(expt-sim) / (datarange*datarange+1e-9)"
ev = ""

class DummyResult:
    def __init__(self, num):
        self.x = [0.0] * num
        self.fun = -1.0

def enumerateFindSimFiles( location ):
    if os.path.isdir( location ):
        if location[-1] != '/':
            location += '/'
        fnames = [ (location + i) for i in os.listdir( location ) if i.endswith( ".json" )]
        return fnames, [1.0] * len( fnames )
    elif os.path.isfile( location ):
        fnames = []
        weights = []
        with open( location, "r" ) as fp:
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
        print( "Error: Unable to find file or directory at " + location )
        quit()

def dumbTicker( result ):
    global ev
    ev.procTicker( result )
    #print( ".", end = '' )
    #sys.stdout.flush()

class EvalFunc:
    def __init__( self, params, expts, pool, modelFile, mapFile, verbose, showTicker = True ):
        self.params = params
        self.expts = expts # Each expt is ( exptFile, weight, scoreFunc )
        self.pool = pool # pool of available CPUs
        self.modelFile = modelFile
        self.mapFile = mapFile
        self.verbose = verbose # bool
        self.showTicker = showTicker
        self.numCalls = 0
        self.numIter = 0
        self.score = []

    def procTicker( self, result ):
        if self.showTicker:
            print( ".", end = '' )
            sys.stdout.flush()
        self.numCalls += 1
        if self.showTicker:
            if self.numCalls % 50 == 0:
                print( " {}".format( self.numCalls ) )

    def doEval( self, x ):
        ret = []
        assert( len(x) == len( self.params) )
        paramList = []

        for i, j in zip( self.params, x ):
            spl = i.rsplit( '.' ,1)
            assert( len(spl) == 2 )
            obj, field = spl
            paramList.append( obj )
            paramList.append( str(field) )
            #paramList.append( field.encode( "ascii") )
            paramList.append( j )
        #print( "{}".format( paramList ) )

        for k in sorted(self.expts):
            ret.append( self.pool.apply_async( findSim.innerMain, (k[0],), dict(scoreFunc = k[2], modelFile = self.modelFile, mapFile = self.mapFile, hidePlot=True, scaleParam=paramList, tabulateOutput = False, ignoreMissingObj = True, silent = not self.verbose ), callback = dumbTicker ) )
        #print( "  = {} {}".format( len( ret ), ret ) )
        #self.ret = [ i.get() for i in ret ]
        #self.ret = { e[0]:i.get() for i, e in zip( ret, self.expts ) }
        self.ret = { e[0]:i.get() for i, e in zip( ret, sorted(self.expts) ) }
        #print( "{}".format( self.ret[0] ) )
        self.score = []
        numFailures = 0
        #for key, val in self.ret.items():
        for key in sorted( self.ret ):
            val = self.ret[key]
            self.score.append( val[0] )
            if val[0] < 0.0:
                print( "Error: EvalFunc: Negative score on expt '{}'".format( key ) )
                numFailures += 1
        if numFailures > 0:
            return -1.0
        sumScore = sum([ s*e[1] for s,e in zip(self.score, self.expts) if s>=0.0])
        sumWts = sum( [ e[1] for s,e in zip(self.score, self.expts) if s>=0.0 ] )
        return sumScore/sumWts

def optCallback( x ):
    global ev
    ev.numIter += 1
    if ev.showTicker == False:
        return
    print ("\nIter {}: [".format( ev.numIter ), end = "" )
    for i in x:
        print ("{:.3f}  ".format( i ), end = "" )
    print( "]" )

def runOptFromCommandLine( args ):
    location = args.location
    if location[-1] != '/':
        location += '/'
    if os.path.isfile( location + args.model ):
        modelFile = location + args.model
    elif os.path.isfile( args.model ):
        modelFile = args.model
    else:
        print( "Error: Unable to find model file {}".format( args.model ) )
        quit()

    #fnames = [ (location + i) for i in os.listdir( args.location ) if i.endswith( ".json" )]
    fnames, weights = enumerateFindSimFiles( args.location )
    expts = zip( fnames, weights, [ defaultScoreFunc ] * len( fnames ) )
    ret = innerMain( args.parameters, expts, modelFile, args.map, args.verbose, args.tolerance, showTicker = args.show_ticker )
    clfnames = { key: args[key] for key in ["model", "map", "resultfile", "optfile" ] }
    #return ret + ( args["model"], args["map"] )
    return ret + (clfnames,)

def checkdir( fname ):
    dirname = os.path.dirname( os.path.realpath( fname ) )
    if not os.path.exists( dirname ):
        raise FileNotFoundError( errno.ENOENT, os.strerror(errno.ENOENT), fname)
        

def fnames( baseargs, val, args ):
    ret = {"model": baseargs["model"], "map": baseargs["map"], "resultfile": args.resultfile, "optfile": args.optfile}
    if len( args.resultfile ) == 0 and "resultFile" in val:
        ret["resultfile"] = val["resultFile"]
        checkdir( ret["resultfile"] )
    if len( args.optfile ) == 0 and "optModelFile" in val:
        ret["optfile"] = val["optModelFile"]
        checkdir( ret["optfile"] )
    return ( ret, )

def runOptFromJson( args ):
    with open( args.location ) as json_file:
        config = json.load( json_file )
    blocks = config["HOSS"]
    basekeys = ["model", "map", "exptDir", "scoreFunc", "tolerance"]
    baseargs = {"exptDir": "./", "tolerance": 1e-4, "show_ticker": args.show_ticker}
    v = vars( args )
    for key, val in config.items():
        if key in basekeys:
            baseargs[key] = val
            vk = v.get( key )
            if vk:
                #baseargs[key] = args[key]
                baseargs[key] = vk
    #if len( args.resultfile ) == 0 
    for hossLevel in blocks:
        if hossLevel["hierarchyLevel"] == 1:
            if args.optblock == "":
                for key, val in hossLevel.items():
                    if key != "name" and key != "hierarchyLevel":
                        fn = fnames( baseargs, val, args )
                        ret =  runJson( key, val, baseargs, args.verbose )
                        return ret + fn
                        #return ret + ( baseargs["model"], baseargs["map"] )
            elif args.optblock in hossLevel:
                val = hossLevel[ args.optblock ]
                ret = runJson( args.optblock, val, baseargs, args.verbose )
                return ret + fnames( baseargs, val, args )
                #return ret + ( baseargs["model"], baseargs["map"] )
            else:
                print( "runOptFromJson: Failed, specified opt block {} not found in Hoss config file {}".format( args.optblock, args.location ) )
                quit()
    

def runJson( optName, optDict, args, isVerbose = False ):
    # The optDict is the individual pathway opt spec from the HOSS Json file
    paramArgs = [ i for i in optDict["params"] ]
    #paramArgs = [ i.encode( "ascii") for i in optDict["params"] ]
    if "scoreFunc" in args:
        df = args["scoreFunc"]
    else:
        df = defaultScoreFunc
    ed = args["exptDir"] + "/"
    expts = []
    #print( "{}".format( optDict["expt"] ))
    for key, val in optDict["expt"].items():
        if "scoreFunc" in val:
            expts.append( (ed + key, val["weight"], val["scoreFunc"] ) )
        else:
            expts.append( (ed + key, val["weight"], df ) )
    expts = sorted(expts)
    ret = innerMain( paramArgs, expts, args["model"], args["map"], isVerbose, args["tolerance"], showTicker = args["show_ticker"] )
    return ret + ( paramArgs, )
    
def extractStatus():
    if ev == "":
        return ( 0, 0, 0 )
    return ( ev.numCalls, ev.numIter, len( ev.expts ) )

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser( description = 'Script to run a multi-parameter optimization in which each function evaluation is the weighted mean of a set of FindSim evaluations. These evaluations may be run in parallel. The optimiser uses the BGFR method with bounds. Since we are doing relative scaling the bounds are between 0.03 and 30 for Kd, tau, Km, Kmod and gain, and between 0 and 30 for other parameters' )
    parser.add_argument( 'location', type = str, help='Required: Directory in which the scripts (in json format) are all located. OR: File in which each line is the filename of a scripts.json file, followed by weight to assign for that file. OR: Json file in hoss format, specifying optimization to run. In case there are multiple optimization blocks, it will take the first by default, or the one specified by name using the --optblock argument')
    parser.add_argument( '-n', '--numProcesses', type = int, help='Optional: Number of processes to spawn', default = 2 )
    parser.add_argument( '-t', '--tolerance', type = float, help='Optional: Tolerance criterion for completion of minimization', default = 1e-4 )
    parser.add_argument( '-m', '--model', type = str, help='Optional: Composite model definition file. First searched in directory "location", then in current directory.' )
    parser.add_argument( '-map', '--map', type = str, help='Model entity mapping file. This is a JSON file.' )
    parser.add_argument( '-p', '--parameters', nargs='*', default=[],  help='Parameter to vary. Each is defined as an object.field pair. The object is defined as a unique MOOSE name, typically name or parent/name. The field is separated from the object by a period. The field may be concInit for molecules, Kf, Kb, Kd or tau for reactions, and Km or kcat for enzymes. It can additionally be tau2, baseline, gain or Kmod in HillTau. One can specify more than one parameter for a given reaction or enzyme. It is advisable to use Kd and tau for reactions unless you have a unidirectional reaction.' )
    parser.add_argument( '-o', '--optfile', type = str, help='Optional: File name for saving optimized model', default = "" )
    parser.add_argument( '-r', '--resultfile', type = str, help='Optional: File name for saving results of simulation as a table of scale factors and scores.', default = "" )
    parser.add_argument( '-b', '--optblock', type = str, help='Optional: Block name to optimize in case we have loaded a Hoss.json file with multiple optimization blocks.', default = "" )
    parser.add_argument( '-v', '--verbose', action="store_true", help="Flag: default False. When set, prints all sorts of warnings and diagnostics.")
    parser.add_argument( '-st', '--show_ticker', action="store_true", help="Flag: default False. Prints out ticker as optimization progresses.")
    args = parser.parse_args()
    if args.location[-4:] == 'json':
        results, eret, optTime, paramArgs, fnames = runOptFromJson( args )
    else:
        results, eret, optTime, fnames = runOptFromCommandLine( args )
        paramArgs = args.parameters

    print( "\n----------- Completed in {:.3f} sec ---------- ".format(time.time() - t0 ) )
    print( "\n----- Score= {:.4f} ------ ".format(results.fun ) )

    dumpData = False
    fp = ""
    if len( fnames["resultfile"] ) > 0:
        fp = open( fnames["resultfile"], "w" )
        dumpData = True
    analyzeResults( fp, dumpData, results, paramArgs, eret, optTime )
    if len( fnames["resultfile"] ) > 0:
        fp.close()
    if len( fnames["optfile"] ) > 0:
        saveTweakedModelFile( args, paramArgs, results.x, fnames )
        dumpData = False

def innerMain( paramArgs, expts, modelFile, mapFile, isVerbose, tolerance, showTicker = True ):
    global ev
    t0 = time.time()
    pool = Pool( processes = len( expts ) )

    params = []
    bounds = []
    for i in paramArgs:
        spl = i.rsplit( '.',1 )
        assert( len(spl) == 2 )
        params.append( i )
        if spl[1] in ['Kd', 'tau', 'tau2', 'Km', 'KA', 'Kmod', 'gain']:
            bounds.append( (0.03,30) )
        else:
            bounds.append( (0.01, 30 ) ) # Concs, Kfs and Kbs can be zero.
    ev = EvalFunc( params, expts, pool, modelFile, mapFile, isVerbose, showTicker = showTicker )
    # Generate the score for each expt for the initial condition
    ret = ev.doEval( [1.0]* len( params ) )
    if ret < -0.1: # Got a negative score, ie, run failed somewhere.
        eret = [ { "expt":e[0], "weight":1, "score": ret, "initScore": 0} for e in ev.expts ]
        return ( DummyResult(len(params) ), eret, time.time() - t0 )

    initScore = ev.score
    # Do the minimization
    results = optimize.minimize( ev.doEval, np.ones( len(params) ), method='L-BFGS-B', tol = tolerance, callback = optCallback, bounds = bounds )
    eret = [ { "expt":e[0], "weight":e[1], "score": s, "initScore": i} for e, s, i in zip( sorted(ev.expts), ev.score, initScore ) ]
    return (results, eret, time.time() - t0 )

def saveTweakedModelFile( args, params, x, fnames ):
    changes = []
    for s, scale in zip( params, x ):
        spl = s.rsplit( '.',1 )
        assert( len( spl ) == 2 )
        changes.append( (spl[0], spl[1], scale ) )
    # Here findSim takes over: Loads model, modifies, tweaks the params,
    # dumps the modified file.
    findSim.saveTweakedModel( fnames["model"], fnames["optfile"], fnames["map"], changes)

def analyzeResults(fp, dumpData, results, params, eret, optTime):
    assert( len(results.x) == len( params ) )
    out = [ "-------------------------------------------------------------"]
    out.append( "Minimization runtime = {:.3f} sec".format( optTime ) )
    for p,x, in zip(params, results.x):
        out.append( "Parameter = {:30s}scale = {:.3f}".format(p, x) )
    out.append( "\n{:40s}{:>12s}{:>12s}{:>12s}".format( "File", "initScore", "finalScore", "weight" ) )
    initSum = 0.0
    finalSum = 0.0
    numSum = 0.0
    for e in eret:
        out.append( "{:40s}{:12.5f}{:12.5f}{:12.3f}".format( e["expt"], e["initScore"], e["score"], e["weight"] ) )
        if e["initScore"] >= 0:
            initSum += e["initScore"] * e["weight"]
            finalSum += e["score"] * e["weight"]
            numSum += e["weight"]
    out.append( "\nInit score = {:.4f}, final = {:.4f}".format(initSum/numSum, results.fun ) )
    for i in out:
        print( i )
        if dumpData:
            fp.write( i + '\n' )
    #fp.close()    
# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
    main()
