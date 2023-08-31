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
 * File:            MonteCarlo.py
 * Description:     Test program to do Monte Carlo optimization as an
 *                  inner loop for Hoss.py
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
import numpy as np
import math
import multiprocessing
import multi_param_minimization
import scramParam
import findSim
import moose

ScorePow = 2.0

HOSS_SCHEMA = "hossSchema.json"

def runJson( optName, optDict, args ):
    ''' Dummy replacement for multi_param_minimization.json. It just runs the findsim expts and returns score '''
    # The optDict is the individual pathway opt spec from the HOSS Json file
    paramArgs = [ i for i in optDict["params"] ]
    solver = "LSODA"
    tolerance = 0.001
    scoreFunc = "NRMS"
    modelFile = args["model"]
    mapFile = args["map"]
    ed = args["exptDir"] + "/"
    ''' #Threaded#
    expts = sorted( [(ed + key, 100, "NRMS") for key in optDict["expt"] ] )
    pool = Pool( processes = 12 )
    ret = []
    for k in expts:
        ret.append (pool.apply_async( findSim.innerMain, k[0], 
            dict( scorefunc = k[2], modelFile = modelFile, mapFile = mapFile, hidePlot = True, scaleParam = paramList, tabulateOutput = False, ignoreMissingObj = True, silent = true ) ) )
    scores = [ ii.get()[0] for ii in ret ]
    '''
    # Serial #
    initParams = findSim.getInitParams( modelFile, mapFile, paramArgs )
    paramList = []
    for item, paramValue in zip( paramArgs, initParams ):
        [obj, field]  = item.rsplit( "." )
        paramList.append( obj )
        paramList.append( field )
        paramList.append( paramValue )
    #print( paramList )

    scores = []
    scoreSum = 0.0
    wtSum = 0.0
    for exptName, val in optDict["expt"].items():
        weight = val["weight"]
        score = findSim.innerMain( ed + exptName, 
            scoreFunc = "NRMS", modelFile = modelFile, mapFile = mapFile, 
            hidePlot = True, scaleParam = paramList, 
            tabulateOutput = False, ignoreMissingObj = True, 
            silent = True, solver = solver )[0]
        scoreSum += score * score * weight
        wtSum += weight

    if wtSum == 0.0:
        return 1.0
    return np.sqrt( scoreSum / wtSum )



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
    print( "TOPDATA = ", topData )

    if len( topNames ) == 1:
        name = topNames[0]
        svec = [ {name:mm} for mm in sortedMonte[name] ]
    else:
        vec = [Row( [0] * len( topNames ) )]
        heapq.heapify( vec )
        svec = []

        while( len( svec ) < num ):
            print( len( vec ), len( svec ), len( topData ), len( topData[0]) )
            rr = heapq.heappop( vec )
            #print( "RR = ", rr )
            if len( vec ) > 0 and not rr == vec[0]:
                entry = { nn:sortedMonte[nn][rr.idx[ii]] for ii, nn in enumerate( topNames ) }   
                print( "ENTRY = {}, {}  {}".format( nn, ii, rr.idx[ii] ) )
                svec.append( entry )
            for ii in range( len( topData ) ):
                idx2 = list(rr.idx)
                idx2[ii] += 1
                print( " pushing in idx2", idx2 )
                heapq.heappush( vec, Row( idx2 ) )

    # Debug Printing stuff:
    for idx, rr in enumerate( svec ):
        for key, val in rr.items():
            print( "{} TOPN {}:{}   s = {:.3f}, {:.3f}".format(idx, key, val.fname, val.rankScore, val.score ) )


    return svec


    '''
    # Then generate the sorted list of dicts. Here we just do a dummy.
    numPathways = len( pathwayScores )
    for ii in range( num ):
        ret.append( { name:pp[ii//numPathways] for name, pp in sortedMonte.items() } )
        #print( "Extending by ", len( ret ) )
        
    for idx, rr in enumerate( ret ):
        for key, val in rr.items():
            print( "{} TOPN {}:{}   s = {:.3f}, {:.3f}".format(idx, key, val.fname, val.rankScore, val.score ) )

    return ret
    '''


def loadMap( fname ):
    objMap = ""
    with open( fname ) as fd:
        objMap = json.load( fd )
    return objMap


def findObj( name, rootpath ):
    try1 = moose.wildcardFind( rootpath+'/' + name )
    try2 = moose.wildcardFind( rootpath+'/##/' + name )
    try2 = [ i for i in try2 if not '/model[0]/plots[0]' in i.path ]
    if len( try1 ) + len( try2 ) > 1:
        raise Exception( "findObj: ambiguous name: '{}'".format(name) )
    if len( try1 ) + len( try2 ) == 0:
        raise Exception( "findObj: No object found on '{}' named: '{}'".format( rootpath, name) )
    if len( try1 ) == 1:
        return try1[0]
    else:
        return try2[0]


def buildModelLookup( model, mapfile ):
    fname, extn = os.path.splitext( model )
    objMap = loadMap( mapfile )
    rootpath = "/model"
    if extn == ".g":
        modelId = moose.loadModel( model, 'model', 'ee' )
    elif extn in [".xml", ".sbml"]:
        modelId, errormsg = moose.readSBML( model, 'model', 'ee' )
    elif extn == ".json":   # Presumed HillTau model, don't need any lookup
        return { key: path[0] for key, path in objMap.items() }

    ret = {}
    for key, paths in objMap.items():
        foundObj = findObj( paths[0], rootpath ).path
        #foundObj = [ findObj( p, rootpath ) for p in paths ]
        #foundObj = [ j.path for j in foundObj if j.name != '/' ]
        if foundObj != "/":
            ret[key] = foundObj
    if moose.exists( '/model' ):
        moose.delete( '/model' )
    if moose.exists( '/library' ):
        moose.delete( '/library' )

    return ret


def mapParamList( objLookup, paramList ):
    ret = []
    for pp in paramList:
        [ obj, field ] = pp.rsplit( "." )
        #print( "MAPPARAM  ", obj, field )
        #print( objLookup )
        #quit()
        fullpath = objLookup.get( obj )
        if fullpath:
            temp = fullpath.replace( "/model[0]/kinetics[0]", "" )
            temp = temp.replace( "[0]", "" )
            ret.append( temp + "." + field )
            #print( "got obj '{}' as '{}', now {}.{}".format( obj, fullpath, temp, field ) )
        else:
            print( "Warning: object ", obj, " not found. Skipping" )
    return ret

def computeModelScores( blocks, baseargs, modelLookup ):
    ret = 0.0
    nret = 0
    levelScores = []
    for hossLevel in blocks: # Assume blocks are in order of execution.
        lret = 0.0
        pathwayScores = {}        
        print( "L{} ".format( hossLevel["hierarchyLevel"] ), end = "" )
        for name, ob in hossLevel.items():
            if name == "name" or name == "hierarchyLevel":
                continue
            paramList = ob["params"]
            #print( paramList )
            #mappedParamList = mapParamList( objLookup, paramList )
            mappedParamList = mapParamList( modelLookup, paramList )
            #print( "MMMMMMMMMM = ", mappedParamList )
            score = runJson(name, ob, baseargs)
            pathwayScores[name] = score
            lret += score * score
            print( "{:12s}{:.3f}    ".format( name, score ), end = "" )
        ret += lret
        nret += 1
        levelScores.append( pathwayScores )
        print()
    print( "Final Score for {} = {:.3f}".format( baseargs["model"], np.sqrt( ret/nret ) ) )
    return levelScores

def runOneModel(blocks, args, baseargs, modelLookup, t0):
    origModel = baseargs['model'] # Use for loading model
    objLookup = loadMap( baseargs["map"] )
    scramRange = baseargs["scramRange"]
    numScram = baseargs["numScram"]
    numTopModels = baseargs["numTopModels"]
    modelFileSuffix = origModel.split( "." )[-1]
    scramModelName = "scram"
    prefix = "MONTE/"
    intermed = []
    results = []

    # First compute the original score. It may well be better.
    origScores = computeModelScores( blocks, baseargs, modelLookup )
    startModelList = [(origModel, 0.0)] # Model name and cumulative score

    # Here is pseudocode to generate many start models, optimize, and 
    # harvest.
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
            optBlock[ name ] = ob
            paramList = ob["params"]
            #print( paramList )
            #mappedParamList = mapParamList( objLookup, paramList )
            mappedParamList = mapParamList( modelLookup, paramList )
            #print( "MMMMMMMMMM = ", mappedParamList )
            pathwayScores[name] = []

            for imm, (mm, cumulativeScore) in enumerate( startModelList ):
                sname = "{}{}_{}_{}.{}".format( prefix, scramModelName, name, imm, modelFileSuffix )
                scramParam.generateScrambled( mm, sname, numScramPerModel, mappedParamList, scramRange )
                # Here we put in the starting model as it may be best
                if imm == 0:
                    ss = origScores[idx][name]
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
                    newScore = runJson( name, ob, baseargs )
                    print( ".", end = "", flush=True)
                    pathwayScores[name].append( 
                            Monte( name, baseargs["model"], imm, ii, 
                            hierarchyLevel, 
                            newScore, cumulativeScore ) )


        topN = buildTopNList( pathwayScores, numTopModels )
        startModelList = []
        for idx, tt in enumerate( topN ):
            for name, monte in tt.items():
                print( "L{}.{}: {} scores={:.3f} {:.3f}   fname= {}".format(
                    hierarchyLevel, idx, name, 
                    monte.score, monte.rankScore, monte.fname ) )

        #print( topN )

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
                    print( "Copyfile ", startModel, "       ", outputModel )
                else:
                    scramParam.mergeModels( startModel, monte.fname, outputModel, ob["params"] )
                firstBlock = False
                #print( "CumulativeScore for {} = {:.3f}".format(outputModel, monte.cumulativeScore ) )

            rmsScore = np.sqrt( rmsScore / len( optBlock )  )
            newScore = (rmsScore + monte.cumulativeScore * monte.level)/(monte.level + 1 )
            startModelList.append( (outputModel, newScore  ) )

    # Finally compute the end score. It should be a lot better.
    baseargs["model"] = "topN_{}_{:03d}.{}".format( hierarchyLevel, 0, modelFileSuffix )
    finalScores = computeModelScores( blocks, baseargs, modelLookup )
        
########################################################################

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
    parser.add_argument( '-ns', '--numScram', type = int, help='Optional: Number of Monte Carlo samples to take by scrambling files.', default = 20 )
    parser.add_argument( '-nt', '--numTopModels', type = int, help='Optional: Number of models out of score-sorted set, to take to next stage of optimization to use as starting points for further scrambling.', default = 5 )
    parser.add_argument( '-n', '--numProcesses', type = int, help='Optional: Number of blocks to run in parallel, when we are not in serial mode. Note that each block may have multiple experiments also running in parallel. Default is to take numCores/8.', default = 0 )
    parser.add_argument( '-r', '--resultFile', type = str, help='Optional: File name for saving results of optimizations as a table of scale factors and scores.', default = "" )
    parser.add_argument( '-sf', '--scoreFunc', type = str, help='Optional: Function to use for scoring output of simulation. Default: NRMS' )
    parser.add_argument( '-scr', '--scramRange', type = float, help='Optional: Factor over which scaling is permitted.', default = 2.0 )
    parser.add_argument( '--solver', type = str, help='Optional: Numerical method to use for ODE solver. Ignored for HillTau models. Default = "LSODA".')
    parser.add_argument( '-v', '--verbose', action="store_true", help="Flag: default False. When set, prints all sorts of warnings and diagnostics.")
    parser.add_argument( '-st', '--show_ticker', action="store_true", help="Flag: default False. Prints out ticker as optimization progresses.")
    args = parser.parse_args( args )

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

    blocks = config["HOSS"]

    # We have a number of necessary option values.
    # The order of priority is: command line, config file, pgm default
    # requiredDefaultArgs is set here in case neither the config file nor 
    # the command line has the argument. We can't use default args in 
    # argparser as this would override the config file.
    requiredDefaultArgs = { 
            "tolerance": 0.001, 
            "scoreFunc": "NRMS", 
            "algorithm": "COBYLA", 
            "solver": "LSODA", 
            "exptDir": "./Expts" ,
            "model": "./Models/model.json",
            "map": "./Maps/map.json",
            "filePrefix": "",
            "scramRange": 2,
            "numScram": 20,
            "numTopModels": 5
        }
    baseargs = vars( args )
    for key, val in requiredDefaultArgs.items():
        if baseargs.get( key ): # command line arg given
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
    modelLookup = buildModelLookup( baseargs["model"], baseargs["map"] )
    optResults = baseargs['resultFile'] # Use for final results
    optModel = baseargs['optfile'] # Use for final model save
    runOneModel( blocks, args, baseargs, modelLookup, t0 )
# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
    main( sys.argv[1:] )
