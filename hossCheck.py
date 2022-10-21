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
 * File:            hossCheck.py
 * Description:
 * Author:          Upinder S. Bhalla
 * E-mail:          bhalla@ncbs.res.in
 ********************************************************************/

/**********************************************************************
** This program is part of 'HOSS', the
** Hierarchical Optimization for Systems Simulations
**           copyright (C) 2019-2022 Upinder S. Bhalla. and NCBS
**********************************************************************/

This script does a sweep of tests to make it more likely that an 
optimization will succeed.
* Hoss config file checks so that specified hierarchy levels are in sequence
* File requirements tests to ensure that all the referred files are present
* Expt File correctness tests using Findsim-Schema.json
* Field existence tests to ensure that all altered entities/parameters exist
- Optimization sanity tests to ensure that we don't try to optimize too
    many parameters with too few experiments
'''

import argparse
import json
import jsonschema
import os
import time
import multi_param_minimization

HOSS_SCHEMA = "hossSchema.json"
FINDSIM_SCHEMA = "FindSim-Schema.json"
HTobjFields = ["concInit", "KA", "tau", "tau2", "Kmod", "Amod", "gain", "baseline" ]
ODEobjFields = ["concInit", "Kd", "tau", "Kf", "Kb", "Km", "kcat"]

######################################

def findHTmols( model ):
    ret = []
    for gg in model["Groups"].values():
        if "Eqns" in gg:
            ret.extend( [ee for ee in gg["Eqns"]] )
        if "Reacs" in gg:
            ret.extend( [rr for rr in gg["Reacs"]] )
        if "Species" in gg:
            ret.extend( [ss for ss in gg["Species"]] )
    if "Constants" in model: # We also permit scaling of named constts.
        ret.extend( [cc for cc in model["Constants"]] )
    return ret

def validateConfig( args ):
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

    try:
        if relpath != '': # Go look in Hoss directory
            fs = relpath + '/' + FINDSIM_SCHEMA
        else:
            fs = FINDSIM_SCHEMA
        with open( fs ) as fsSchemaFile:
            fsSchema = json.load( fsSchemaFile )
            #fsSchemaValidator = jsonschema.protocols.Validator( fsSchema )
    except IOError:
        print( "Error: Unable to find FindSim schema file: " + FINDSIM_SCHEMA )
        quit()
    print( "Validated config file: ", args.config )
    return config, fsSchema 

def validateMap( args, config ):
    if args.map == None:
        mapFileName = config["map"]
    else:
        mapFileName = args.map
    try:
        with open( mapFileName ) as map_file:
            try:
                modelmap = json.load( map_file )
            except json.decoder.JSONDecodeError as err:
                print( "Error: Load/syntax error in map file: '{}'".format( mapFileName ) )
                print( err )
                quit()
    except IOError:
        print( "Error: Unable to find map file: " + mapFileName )
        quit()
    print( "Validated map file: ", mapFileName )
    return modelmap

def validateModel( args, config ):
    if args.model == None:
        modelFileName = config["model"]
    else:
        modelFileName = args.model
    try:
        with open( modelFileName ) as model_file:
            modelSuffix = modelFileName.split( "." )[-1]
            if modelSuffix == "json":
                try:
                    model = json.load( model_file )
                    modelMols = findHTmols( model )
                except json.decoder.JSONDecodeError as err:
                    print( "Error: Load/syntax error in model file: '{}'".format( modelFileName ) )
                    print( err )
                    quit()
                objFields = HTobjFields
            else:
                objFields = ODEobjFields
                modelMols = []
    except IOError:
        print( "Error: Unable to find model file: " + modelFileName )
        quit()
    print( "Validated model file: ", modelFileName )
    return model, modelMols, objFields

def validateHierarchyOrdering( blocks ):
    fail = False
    for idx, bb in enumerate( blocks ):
        hL = bb['hierarchyLevel']
        if hL != idx + 1:
            print( "Error: expected block heirarchy {}, got {} for {}".format( idx+1, hL, bb['name'] ) )
            fail = True
    if fail: 
        quit()
    print( "Cleared test for sequential hierarchyLevel in blocks")

def checkForExperimentFiles( args, blocks ):
    fail = False
    numExpts = 0
    for idx, bb in enumerate( blocks ):
        for pname, pblock in bb.items():
            if not pname in ["name", "hierarchyLevel"]:
                expts = pblock['expt']
                for ee in expts:
                    epath = args.exptDir + "/" + ee
                    try:
                        with open( epath ) as exptFile:
                            print( ".", end = "", flush=True)
                            numExpts += 1
                    except IOError:
                        print( "\nError: Unable to find expt file: '{}' in pathway '{}' in block {} ".format( epath, pname, idx+1  ) )
                        fail = True
    if fail: 
        quit()
    print( "\nFound all {} named experiment files.".format( numExpts ) )

def validateExperimentFiles( args, blocks, fsSchema ):
    fail = False
    numExpts = 0
    exptDefnDict = {}
    for idx, bb in enumerate( blocks ):
        for pname, pblock in bb.items():
            if not pname in ["name", "hierarchyLevel"]:
                expts = pblock['expt']
                for ee in expts:
                    epath = args.exptDir + "/" + ee
                    try:
                        with open( epath ) as exptFile:
                            try:
                                exptDefn = json.load( exptFile )
                            except json.decoder.JSONDecodeError as err:
                                print( "\nError: Load/syntax error in expt file: '{}' in pathway '{}' in block {} ".format( epath, pname, idx+1  ) )
                                print( err )
                                fail = True
                                continue
                            #jsonschema.protocols.Validator.validate( exptDefn )
                            try:
                                jsonschema.validate( exptDefn, fsSchema )
                                print( ".", end = "", flush=True)
                                numExpts += 1
                            except jsonschema.exceptions.ValidationError as err:
                                print( "\nError: Failed to validate expt file: '{}' in pathway '{}' in block {} ".format( epath, pname, idx+1  ) )
                                print( err )
                                fail = True
                    except IOError:
                        print( "\nError: Unable to find expt file: '{}' in pathway '{}' in block {} ".format( epath, pname, idx+1  ) )
                        fail = True

    if fail: 
        print( "\n" )
        quit()
    print( "\nValidated all {} named experiment files.".format( numExpts ) )

def checkModelObjectsExist( blocks, modelmap, objFields, modelMols ):
    fail = False
    numParams = 0
    for idx, bb in enumerate( blocks ):
        for pname, pblock in bb.items():
            if not pname in ["name", "hierarchyLevel"]:
                params = pblock['params']
                for pp in params:
                    ppsp = pp.split(".")
                    if len( ppsp ) != 2:
                        print( "\nError: Bad param formatting for '{}' in pathway '{}' in block {}".format( pp, pname, idx+1 ) )
                        fail = True
                    obj = ppsp[0]
                    field = ppsp[1]
                    ok = True
                    if not field in objFields:
                        print( "\nError: Unknown field name: '{}' on obj '{}' in pathway '{}' in block {}".format( field, obj, pname, idx+1 ) )
                        fail = True
                        ok = False
                    if (obj in modelmap) or (obj in modelMols):
                        if ok:
                            print( ".", end = "", flush=True)
                            numParams += 1
                    else:
                        print( "\nError: Missing Obj name in mapfile: '{}' in pathway '{}' in block {}".format( obj, pname, idx+1 ) )
                        fail = True
    if fail: 
        print( "\n" )
        quit()
    print( "\nValidated all {} parameters.".format( numParams ) )

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser( description = 
            'This script does a sweep of tests to catch cases that may prevent an optimization from working. ')
    parser.add_argument( 'config', type = str, help='Required: JSON configuration file for doing the optimization.')
    parser.add_argument( '-m', '--model', type = str, help='Optional: Composite model definition file. First searched in directory "location", then in current directory.' )
    parser.add_argument( '-map', '--map', type = str, help='Model entity mapping file. This is a JSON file.' )
    parser.add_argument( '-e', '--exptDir', type = str, help='Optional: Location of experiment files.', default = "./Expts" )
    parser.add_argument( '-o', '--optfile', type = str, help='Optional: File name for saving optimized model', default = "" )
    parser.add_argument( '-r', '--resultfile', type = str, help='Optional: File name for saving results of optimizations as a table of scale factors and scores.', default = "" )
    args = parser.parse_args()

    ## Check 0: Load and validate the config file
    config, fsSchema = validateConfig( args )


    ## Check 0.1: Load and validate the map file. We'll use it later too.
    modelmap = validateMap( args, config )

    ## Check 0.2: Load and validate the model file. We'll use it later too.
    model, modelMols, objFields = validateModel( args, config )


    blocks = config["HOSS"]
    ## Check 1: All hierarchyLevels are present, sequentially
    validateHierarchyOrdering( blocks )


    ## Check 2: All expt files are present and readable
    checkForExperimentFiles( args, blocks )

    ## Check 3: All expt files are well-formed JSON and pass FindSim-Schema
    validateExperimentFiles( args, blocks, fsSchema )

    ## Check 4: All named objects exist in the map file or model.
    checkModelObjectsExist( blocks, modelmap, objFields, modelMols )


        
# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
    main()
