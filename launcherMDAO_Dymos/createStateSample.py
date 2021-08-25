# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:04:00 2021

@author: jorge
"""
from pathlib import Path
p = Path(__file__).parents[1]
import sys
sys.path.append( str(p) )

import os
dirname = os.path.dirname(__file__)

import openturns as ot
import numpy as np
import codecs
import json

def createOTMeshDymos(time, solutionType):
    # given the "intepolated_results.txt" containing all the trajectories
    # interpolated to an equispaced grid using the "interpolate_results" function
    # create the input and output samples necessary for the Active Learning technique
    
     
    # create OT mesh
    t_min    = 0.0
 
    t_max    = float(min(time[:,-1]))
        
    gridsize = time.shape[1]
    mesh = ot.IntervalMesher([gridsize-1]).build(ot.Interval(t_min, t_max))
    # obtain vertices
    # save mesh
    mesh = { 'simplices' : np.array(mesh.getSimplices()).tolist(),
             'vertices'  : np.array(mesh.getVertices()).tolist()}
    
    return mesh

def extractInput(state, readPath, solutionType):
    # read text file
    # # read data sets
    obj_text = codecs.open(readPath, 'r', encoding='utf-8').read()
    # load list with dictionaries containing interpolated results
    interp_results = json.loads(obj_text)['interp_results']
    # ============================================================
    # eliminate the results from Dymos that were identified to tap the 
    # lower bound of the mp_1 variable after a visual inspection of the 
    # opt_report.txt files. more info in the text file:
    # results_DYMOS-SPIRO/README_Important.txt
    removeDictsList = [135, 347, 525, 620, 634, 714, 813, 905, 1048, 1049]
    # the list must be in descending order to avoid changes in the indices
    removeDictsList.reverse()
    for idx in removeDictsList:
        interp_results.pop(idx)
    # =========================================================================
    # 2 types of interpolated Dymos solutions are available
    # 'interpolated_results':    is the interpolation using quadratic splines 
    #                            of the LGL state nodes. to be used with normalized time
    # 'interpolated_resultsSim': is the interpolation of the Runge-Kutta solution
    #                            of the Simulate method combined with a coasting phase
    #                            propagation.
    if solutionType == 'pseudospectral':
        dymosSolution = 'interpolated_results'
        time = 'time_norm'
    elif solutionType == 'simulate':
        dymosSolution = 'interpolated_resultsSim'
        time = 'time'
    # =========================================================================
    # intialize timeseries  and input lists
    timeseries_list = []
    u_list          = []
    time_list  = []
    
    # for Id in range(number_of_realizations):
    counter = 0
    for results in interp_results:
        try:
            timeseries_list.append( results[dymosSolution][state] )
            time_list.append(results[dymosSolution][time])
            u_list.append(  results['opt_info']['uncertainties'] )
        except:
            counter += 1
    print('''
          ===========================================================
          %i dictionaries containing Dymos trajectories failled. 
          Likely the trajectory did not converge and they are empty.
          ===========================================================
          '''%counter)
    
    return np.array(timeseries_list), np.array(u_list), np.array(time_list)


def saveTextFiles(state, readPath, inputPath, outputPath, indicesArr, solutionType):
    # save output file ==================================
    timeseries, u, time = extractInput(state, readPath, solutionType)
    # all trajectories have the same normalized time...pass only one traj
    mesh          = createOTMeshDymos(time, solutionType)
    # modify this code to generate diffferent training and valdiation samples
    timeseries = timeseries[indicesArr,:]
    u          = u[indicesArr,:]
    # write json file
    exDict = {'resultsDict': timeseries.tolist(),
              'mesh':mesh}
    # write text file
    with open(outputPath, 'w') as file:
         file.write(json.dumps(exDict))
         file.close()
    # save input file ===================================
    # convert np.arrays into lists
    u_list = u.tolist()
    # write json file
    exDict = {'resultsDict': u_list}
    # write text file
    with open(inputPath, 'w') as file:
         file.write(json.dumps(exDict))
         file.close()
         
def createStateSample(indicesArrTraining, indicesArrValidation, state, solutionType, resultsPath):
    readPath   = dirname + '/samples/intepolated_results.txt'
    inputPath  = resultsPath + '/inputTrainingSample.txt'
    outputPath = resultsPath + '/outputTrainingSample.txt'
    saveTextFiles(state, readPath, inputPath, outputPath, indicesArrTraining, solutionType)
    inputPath  = resultsPath + '/inputValidationSample.txt'
    outputPath = resultsPath + '/outputValidationSample.txt'
    saveTextFiles(state, readPath, inputPath, outputPath, indicesArrValidation, solutionType)

if __name__ == '__main__':
    createStateSample(np.array(range(0,230)), np.array(range(240,1130)), 'r', 'simulate')      

    