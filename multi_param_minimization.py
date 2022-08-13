
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
import math
import time
import json
import findSim
from multiprocessing import Pool

defaultScoreFunc = "(expt-sim)*(expt-sim) / (datarange*datarange+1e-9)"
ev = ""
#algorithm = 'SLSQP'
ScorePow = 2.0
MINIMUM_CONC = 1e-10
MIDDLE_CONC = 1e-3
PENALTY_SLOPE = 1.0

class Bounds:
    '''
    This class maintains bounds for each parameter, and provides for conversions. All inputs are expected to be in the range 0..1. The bounds.func will take the input x in this range and return a value within the bounds. It can use either exponential or linear scaling within the range, exponential is the default. It also has a function to return the inverse transform value, which is useful to get a number between 0..1 for which the Bounds.func will return the starting value of the parameter. For symmetric exponential and symmetric linear ranges, this value is 0.5.
    '''
    def __init__( self, lo, hi, isLinear = False ):
        self.lo = lo
        self.hi = hi
        self.penaltyLowBound = 0.2
        self.penaltyHighBound = 1.0 - self.penaltyLowBound
        self.penaltySlope = 0.5 / self.penaltyLowBound
        self.name = ""
        if isLinear:
            self.range = self.hi - self.lo
            self.func = self.linBounds
        else:
            if self.lo < MINIMUM_CONC:
                self.lo = MINIMUM_CONC
            if self.hi < MINIMUM_CONC:
                self.hi = MIDDLE_CONC
            self.range = np.log( self.hi / self.lo )
            self.func = self.expBounds

    def linBounds( self, x ): # returns a value linearly between lo and hi
        #return x
        return self.smootherstep( x ) * self.range + self.lo

    def expBounds( self, x ): # value exponentially between lo and hi
        #return self.lo * np.exp( x * self.range )
        return self.lo * np.exp( self.smootherstep(x) * self.range )

    def boundsPenalty( self, x ): # Penalty to score for values outside bounds
        ret = (x - 0.5) * (x - 0.5 )
        ret = 0.0
        if x < self.penaltyLowBound:
            ret = self.penaltySlope * (self.penaltyLowBound - x)
        elif x > self.penaltyHighBound:
            ret = self.penaltySlope * (x - self.penaltyHighBound)
        return ret * ret

        '''
        for x in xvec:
            y = 0.0
            if x < self.penaltyLowBound:
                y = self.penaltySlope * (self.penaltyLowBound - x)
            elif x > self.penaltyHighBound:
                y =  self.penaltySlope * (x - self.penaltyHighBound)
            ret += y * y
        return np.sqrt( ret / len( xvec ) )
        '''

    def newInvFunc( self, p ): 
        if self.func == self.linBounds:
            return p
        return np.log( p/self.lo )/self.range

    def invFunc( self, p ): 
        # func( x in range 0..1 ) = paramValue
        # So, invFunc( paramValue ) = x in range 0..1
        if p <= self.lo:
            return 0.0
        elif p >= self.hi:
            return 1.0
        guess = 0.5
        lolim = 0.0
        hilim = 1.0
        for i in range(20):
            delta = self.func( guess ) -  p
            #print( "{}  {}   {}  {}  {} {}".format( i, lolim, hilim, guess, p, abs( delta ) * 1e8 ) )
            if abs( delta ) < 1e-6 * self.range:
                return guess
            if delta > 0:
                hilim = guess
                guess = (guess + lolim) / 2.0
            else:
                lolim = guess
                guess = (guess + hilim) / 2.0
        return guess
    
    # A smoother version based on smootherstep.
    def smootherstep( self, x ):
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        x = x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
        return x

defaultBounds = {
        "conc":Bounds(1e-9, 100.0), 
        "concInit":Bounds(1e-9, 100.0),
        "KA": Bounds(1e-9, 100.0), 
        "Kd": Bounds(1e-6, 100.0),  # Nanomolar.
        "Km": Bounds(1e-6, 1.0),  # nanomolar to millimolar affinity
        "kcat": Bounds(1e-3, 100.0), 
        "tau": Bounds( 0.1, 2000.0), 
        "tau2": Bounds( 0.2, 4000.0), 
        "Kmod": Bounds( 1e-9, 100.0), 
        "Amod": Bounds( 1e-6, 1e6), 
        "baseline": Bounds( 1e-9, 100.0),
        "gain": Bounds( 1e-4, 1e4) }

class DummyResult:
    def __init__(self, num):
        self.x = [0.0] * num
        self.initParams = [1.0] * num
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
    def __init__( self, params, bounds, expts, pool, modelFile, mapFile, verbose, showTicker = True, solver = "gsl" ):
        # params specified as list of strings of form object.field 
        self.params = params
        # paramBounds specified as list of Bounds objects
        self.paramBounds = bounds
        self.expts = expts # Each expt is ( exptFile, weight, scoreFunc )
        self.pool = pool # pool of available CPUs
        self.modelFile = modelFile
        self.mapFile = mapFile
        self.verbose = verbose # bool
        self.showTicker = showTicker
        self.solver = solver
        self.numCalls = 0
        self.numIter = 0
        self.score = []
        self.runtime = 0.0
        self.loadtime = 0.0
        self.paramAccessTime = 0.0

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
        paramList = []

        boundsPenalty = 0.0
        if len( x ) > 0:
            if len(x) != len( self.params ):
                print( "Warning: parameter vector length differs from # of params", len(x), "    ", len( self.params ), "    ", self.params )
            assert( len(x) == len( self.params) )
            # radial distance of param from origin, which centers at 0.5,0.5
            bpsq = 0.0
            for pb, param in zip( self.paramBounds, x ):
                bpsq += (0.5 - param) * (0.5 - param)
            r = np.sqrt( bpsq )
            boundsPenalty = PENALTY_SLOPE * max( r - 0.5, 0.0 )
            #boundsPenalty = self.paramBounds[0].boundsPenalty( rdist )
            pb = self.paramBounds[-1]
            #print( "boundsPenalty = ", boundsPenalty, x, pb.name, pb.lo, pb.hi )
            print( "boundsPenalty = ", boundsPenalty, x )

            for i, j, b in zip( self.params, x, self.paramBounds ):
                spl = i.rsplit( '.' ,1)
                assert( len(spl) == 2 )
                obj, field = spl
                paramList.append( obj )
                paramList.append( str(field) )
                #paramList.append( field.encode( "ascii") )
                #y = 0.01 + 99.99 / ( 1 + np.exp( -j ) )
                paramList.append( b.func(j) )
                #print( "{} = {:.3f}".format( i, b.func(j) ))
            #print( "{}".format( paramList ) )
        if boundsPenalty > 0.0:
            return boundsPenalty + 1.0

        if len( self.expts ) == 1:
            k = self.expts[0]
            ret.append( findSim.innerMain( k[0], scoreFunc = k[2], modelFile = self.modelFile, mapFile = self.mapFile, hidePlot=True, scaleParam=paramList, tabulateOutput = False, ignoreMissingObj = True, silent = not self.verbose, solver = self.solver ))
            #print( "RET ===========", ret )
            self.ret = { e[0]:i for i, e in zip( ret, sorted(self.expts) ) }
        else:
            for k in sorted(self.expts):
                #print ("ssssssssssssssscoreFunc = ", k[2] )
                ret.append( self.pool.apply_async( findSim.innerMain, (k[0],), dict(scoreFunc = k[2], modelFile = self.modelFile, mapFile = self.mapFile, hidePlot=True, scaleParam=paramList, tabulateOutput = False, ignoreMissingObj = True, silent = not self.verbose, solver = self.solver ), callback = dumbTicker ) )
            #print( "RET ===========", ret )
            self.ret = { e[0]:i.get() for i, e in zip( ret, sorted(self.expts) ) }
        #print( "  = {} {}".format( len( ret ), ret ) )
        #self.ret = [ i.get() for i in ret ]
        #self.ret = { e[0]:i.get() for i, e in zip( ret, self.expts ) }
        #print( "{}".format( self.ret[0] ) )
        self.score = []
        numFailures = 0
        #for key, val in self.ret.items():
        for key in sorted( self.ret ):
            val = self.ret[key]
            self.score.append( val[0] )
            if val[0] < 0.0:
                print( "Error: EvalFunc: Negative score {} on expt '{}'".format( val[0], key ) )
                numFailures += 1
            else:
                self.runtime += val[2]["runtime"]
                self.loadtime += val[2]["loadtime"]
                self.paramAccessTime += val[2]["paramAccessTime"]
        if numFailures > 0:
            return -1.0
        sumScore = sum([ pow( s, ScorePow )*e[1] for s, e in zip(self.score, self.expts) if s>=0.0])
        sumWts = sum( [ e[1] for s, e in zip(self.score, self.expts) if s>=0.0 ] )
        #print("RET = {:.3f}".format( pow( sumScore/sumWts, 1.0/ScorePow )))
        return pow( sumScore/sumWts, 1.0/ScorePow )

def optCallback( x ):
    global ev
    ev.numIter += 1
    if ev.showTicker == False:
        return
    print ("\nIter {}: [".format( ev.numIter ), end = "" )
    #sx = [ sigmoid( j ) for j in x ]
    sx = x
    for i in sx:
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
    ret = innerMain( args.parameters, expts, modelFile, args.map, args.verbose, args.tolerance, showTicker = args.show_ticker, algorithm = args.algorithm, solver = args.solver )
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
    baseargs = {"exptDir": "./", "tolerance": 1e-4, "show_ticker": args.show_ticker, "algorithm": args.algorithm, "solver": args.solver}
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
            if args.optblock == "": # Just do the first one.
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
    #print( "RJRJRJRJRJ..........", optName, "\n", "\n", args )
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
    if "paramBounds" in optDict:
        paramBounds = { key: Bounds(val[0], val[1], val[2]) for key, val in optDict["paramBounds"].items() }
    else:
        paramBounds = {}

    ret = innerMain( paramArgs, expts, args["model"], args["map"], isVerbose, args["tolerance"], showTicker = args["show_ticker"], algorithm = args["algorithm"], paramBounds = paramBounds, solver = args["solver"] )
    return ret + ( paramArgs, )
    
def extractStatus():
    if ev == "":
        return ( 0, 0, 0 )
    return ( ev.numCalls, ev.numIter, len( ev.expts ) )

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser( description = 'Script to run a multi-parameter optimization in which each function evaluation is the weighted mean of a set of FindSim evaluations. These evaluations may be run in parallel. The optimiser uses various algorithm available with scipy.optimize, default SLSQP. Since we are doing relative scaling the bounds are between 0.01 and 100 for all parameters' )
    parser.add_argument( 'location', type = str, help='Required: Directory in which the scripts (in json format) are all located. OR: File in which each line is the filename of a scripts.json file, followed by weight to assign for that file. OR: Json file in hoss format, specifying optimization to run. In case there are multiple optimization blocks, it will take the first by default, or the one specified by name using the --optblock argument')
    parser.add_argument( '-a', '--algorithm', type = str, help='Optional: Algorithm name to use, from the set available to scipy.optimize.minimize. Options are CG, Nelder-Mead, Powell, BFGS, COBYLA, SLSQP, trust-constr. The library has other algorithms but they either require Jacobians or they fail outright. There is also L-BFGS-B which handles bounded solutions, but this is not needed here because we already take care of bounds. SLSQP works well and is the default.', default = "SLSQP" )
    parser.add_argument( '-n', '--numProcesses', type = int, help='Optional: Number of processes to spawn', default = 2 )
    parser.add_argument( '-t', '--tolerance', type = float, help='Optional: Tolerance criterion for completion of minimization', default = 1e-4 )
    parser.add_argument( '-m', '--model', type = str, help='Optional: Composite model definition file. First searched in directory "location", then in current directory.' )
    parser.add_argument( '-map', '--map', type = str, help='Model entity mapping file. This is a JSON file.' )
    parser.add_argument( '-p', '--parameters', nargs='*', default=[],  help='Parameter to vary. Each is defined as an object.field pair. The object is defined as a unique MOOSE name, typically name or parent/name. The field is separated from the object by a period. The field may be concInit for molecules, Kf, Kb, Kd or tau for reactions, and Km or kcat for enzymes. It can additionally be tau2, baseline, gain or Kmod in HillTau. One can specify more than one parameter for a given reaction or enzyme. It is advisable to use Kd and tau for reactions unless you have a unidirectional reaction.' )
    parser.add_argument( '-pb', '--parameter_bounds', nargs=4, default=[],  help='Set bounds for a parameter. If the parameter is not already included in the list, put it in. The arguments are: object.field lower_bound upper_bound isLinear. [str, float, float, int]. In most cases, isLinear should be 0 to indicate that the system should scale the parameter exponentially. Default values for bounds are concs, baseline, KA and Kmod: 1e-9 to 100, tau: 0.1 to 2000, tau2: 0.2 to 4000, Amod: 1e-6 to 1e6, gain: 1e-4 to 1e4. All default to exponential scaling', metavar = "args" )
    parser.add_argument( '-nb', '--narrow_bounds', nargs=1, help='Set narrow bounds for all parameters to scale up and down by the specified factor' )
    parser.add_argument( '-o', '--optfile', type = str, help='Optional: File name for saving optimized model', default = "" )
    parser.add_argument( '-r', '--resultfile', type = str, help='Optional: File name for saving results of simulation as a table of scale factors and scores.', default = "" )
    parser.add_argument( '-b', '--optblock', type = str, help='Optional: Block name to optimize in case we have loaded a Hoss.json file with multiple optimization blocks.', default = "" )
    parser.add_argument( '--solver', type = str, help='Optional: Numerical method to use for ODE solver. Ignored for HillTau models. Default = "gsl".', default = "gsl" )
    parser.add_argument( '-sf', '--scoreFunc', type = str, help='Optional: Function to use for scoring output of simulation.', default = "NRMS" )
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
    print( "\nCalls={:5d}  Iter={:3d}  evalTime={:.3f}  loadTime={:.3f}  paramAccessTime={:.3f}".format(ev.numCalls, ev.numIter, ev.runtime, ev.loadtime, ev.paramAccessTime ) )

    dumpData = False
    fp = ""
    if len( fnames["resultfile"] ) > 0:
        fp = open( fnames["resultfile"], "w" )
        dumpData = True
    analyzeResults( fp, dumpData, results, paramArgs, eret, optTime, args.scoreFunc )
    if len( fnames["resultfile"] ) > 0:
        fp.close()
    if len( fnames["optfile"] ) > 0:
        saveTweakedModelFile( args, paramArgs, results.x, fnames )
        dumpData = False

def findInitialParams( expts, modelFile, mapFile, paramArgs, solver ):
    if len( paramArgs ) == 0:
        raise KeyError( "multi_param_minimization.py::findInitialParams: Quit because no paramArgs were given" )
    initParams = findSim.getInitParams( modelFile, mapFile, paramArgs )

    if initParams[0] == -1:
        raise KeyError( "multi_param_minimization.py::findInitialParams: Quit because initParams[0] == -1" )

    '''
    for idx, ip in enumerate( initParams ):
        if not math.isclose( ip, -2.0 ): # initParams has -2 if obj not found. This occurs sometimes when subsetting, so we have to skip this param.
            mergedInitParams[idx] = ip
    numBad = sum( [ math.isclose( mip, -2.0 ) for mip in mergedInitParams] )
    if numBad == 0:
        break;
    elif numBad == len( initParams ):
        raise KeyError( "No valid params in multi_param_minimization::innerMain for expt = " + ee[0] )

    for pa, mip in zip( paramArgs, mergedInitParams ):
        if math.isclose( mip, -2.0 ):
            print( "Error: findInitialParams: Parameter not found: ", pa )
    if numBad > 0:
        raise KeyError( "Invalid params in multi_param_minimization" )

    return mergedInitParams
    '''
    return initParams


def innerMain( paramArgs, expts, modelFile, mapFile, isVerbose, tolerance, showTicker = True, algorithm = "SLSQP", paramBounds = {}, solver = "gsl" ):
    global ev
    t0 = time.time()
    pool = Pool( processes = len( expts ) )

    # Some nasty stuff here to get the initial parameters from the model.
    # Ideally there should be a way to combine with the previous eval.
    initParams = findInitialParams( expts, modelFile, mapFile, paramArgs, solver )
    #print( "INIT PARAMS = ", initParams, "\n expt= ", expts[0][0] )

    # By default, set the bounds in the range of 0.01 to 100x original.
    params = []
    bounds = []
    inits = []
    for i, ip in zip( paramArgs, initParams ):
        if math.isclose( ip, -2.0 ):    # Skip missing params.
            continue
        inits.append( ip )
        spl = i.rsplit( '.',1 ) # i is of the form: object.field
        assert( len(spl) == 2 )
        params.append( i )
        pb = paramBounds.get( i )
        if pb:
            bounds.append( pb )
        else:
            if ip <= 0.0:
                bounds.append( Bounds( MINIMUM_CONC, MIDDLE_CONC ) )
            else:
                bounds.append( Bounds( ip * 0.01, ip * 100.0 ) )
        bounds[-1].name = i
        #print( "{} = {:.4g}, bounds = {:.4g}, {:.4g}".format( i, ip, bounds[-1].lo, bounds[-1].hi ) )
            #bounds.append( defaultBounds.get( spl[1] ) )
    #print( "PARAMS = ", params )
    #print( "INIT  = ", [i for i in initParams ])
    #print( "BOUNDS = ", [ (b.lo, b.hi) for b in bounds] )
    #print( "------------------------------------------------" )
    ev = EvalFunc( params, bounds, expts, pool, modelFile, mapFile, isVerbose, showTicker = showTicker, solver = solver )
    # Generate the score for each expt for the initial condition
    ret = ev.doEval( [] )
    if ret < -0.1: # Got a negative score, ie, run failed somewhere.
        eret = [ { "expt":e[0], "weight":1, "score": ret, "initScore": 0} for e in ev.expts ]
        return ( DummyResult(len(params) ), eret, time.time() - t0 )
    initScore = ev.score
    #print( "INIT SCORE = ", initScore )
    initVec = [ b.invFunc(p) for b, p in zip( bounds, inits ) ]
    for b, p in zip( bounds, inits ):
        lo = b.func(b.penaltyLowBound)
        hi = b.func(b.penaltyHighBound)
        if p <lo or p > hi:
            print( "Warning: Initial value {} of parameter {} is outside specified bounds {} to {}".format( p, b.name, lo, hi ) )
    #print( ["{:.3f} {:.3f}".format( p, b.invFunc(p) ) for b, p in zip( bounds, initParams)] )
    #print( "INITs = ", inits )
    #print( "INITVec = ", initVec )
    #print( "INITParms = ", initParams )
    

    # Do the minimization
    if algorithm in ['COBYLA']:
        callback = None
    else:
        callback = optCallback

    results = optimize.minimize( ev.doEval, initVec, method= algorithm, tol = tolerance, callback = callback )
    eret = [ { "expt":e[0], "weight":e[1], "score": s, "initScore": i} for e, s, i in zip( sorted(ev.expts), ev.score, initScore ) ]
    results.x = [ b.func( x ) for x, b in zip( results.x, ev.paramBounds ) ]
    results.initParams = initParams
    return (results, eret, time.time() - t0 )

def saveTweakedModelFile( args, params, x, fnames ):
    changes = []
    #sx = [ sigmoid( j ) for j in x ]
    sx = x
    for s, scale in zip( params, sx ):
        spl = s.rsplit( '.',1 )
        assert( len( spl ) == 2 )
        changes.append( (spl[0], spl[1], scale ) )
    # Here findSim takes over: Loads model, modifies, tweaks the params,
    # dumps the modified file.
    findSim.saveTweakedModel( fnames["model"], fnames["optfile"], fnames["map"], changes)

def analyzeResults(fp, dumpData, results, params, eret, optTime, scoreFunc, verbose=True):
    #print( "RES.x = ", results.x )
    #print( "RES.initParams = ", results.initParams )
    #print( "Params = ", params )

    assert( len(results.x) == len( params ) )
    assert( len(results.x) == len( results.initParams ) )
    sys.stdout.flush()
    out = [ "-------------------------------------------------------------"]
    out.append( "scoreFunc = {}, Minimization runtime = {:.3f} sec".format( scoreFunc, optTime ) )
    #sx = [ sigmoid( j ) for j in results.x ]
    sx = results.x
    out.append( "Parameter              Initial Value     Final Value          Ratio ")
    for p,x,y in zip(params, sx, results.initParams):
        if y <= 0.0:
            out.append( "{:20s}{:16.4g}{:16.4g}{:>16s}".format(p, y, x, "---") )
        else:
            out.append( "{:20s}{:16.4g}{:16.4g}{:16.4f}".format(p, y, x, x/y) )
    out.append( "\n{:40s}{:>12s}{:>12s}{:>12s}".format( "File", "initScore", "finalScore", "weight" ) )
    initSum = 0.0
    finalSum = 0.0
    numSum = 0.0
    for e in eret:
        exptFile = e["expt"].split('/')[-1]
        out.append( "{:40s}{:12.5f}{:12.5f}{:12.3f}".format( exptFile, e["initScore"], e["score"], e["weight"] ) )
        eis = e["initScore"]
        if eis >= 0:
            initSum += pow( eis, ScorePow) * e["weight"]
            finalSum += e["score"] * e["score"] * e["weight"]
            numSum += e["weight"]
    out.append( "\nInit score = {:.4f}, final = {:.4f}".format(pow(initSum/numSum, 1.0/ScorePow), results.fun ) )
    for i in out:
        if verbose:
            print( i )
        if dumpData:
            fp.write( i + '\n' )
    #fp.close()    
# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
    main()
