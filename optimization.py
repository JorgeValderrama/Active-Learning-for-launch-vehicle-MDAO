# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:35:03 2021
============== Optimization loop =============================================
==============================================================================

This function identifies the sample that contributes the most to the uncertainty
reduction goal. It is build on the CMA-ES optimizer but it could use a different
optimizer.

@author: jorge
"""

from metamodels import  FullMetamodel
import numpy as np
import openturns as ot
import cma
# ================ attention =================================================
# disable parallel processing from openturns as the code is parallelized 
# at the optimizer level
ot.TBB.Disable()
# ot.TBB.Enable()
# ============================================================================
# CMA parallelization
# https://github.com/CMA-ES/pycma/blob/7b7a4a1cd09db69ee72e71ba5ed64a03b1b80d4e/cma/optimization_tools.py

def obj_fun(u, ksi_u, inputTrainingSample, GPTrained, karhunenLoeveLiftingFunction, 
            eigenFunctions, eigenValues, areaArgs, areaFunction):
    # objective function for the CMA optimizer
    # update the GP metamodel by using a virtual input and its metamodel output
    GPTrained.updateVirtualInput(ot.Point(u), ksi_u, inputTrainingSample)
    # rebuild full metamodel with updated GP
    fullMetamodel = FullMetamodel(karhunenLoeveLiftingFunction, GPTrained, \
                                  eigenFunctions, eigenValues)
    # compute the area of the confidence interval
    CIArea = areaFunction( fullMetamodel, areaArgs )
    return CIArea

def run_opt(ksi_u, activeLearningProblem, GPTrained, karhunenLoeveLiftingFunction, 
            eigenFunctions, eigenValues, areaArgs, areaFunction, optArgs):
    # unpack optimaztion args
    popSize, njobs, iterations =optArgs
    # get the dimension of the input training sample
    dim = activeLearningProblem.inputTrainingSample.getDimension()
    # define options for cma
    savePath = areaArgs[-1]+'/outcmaes/iteration_%i/'%areaArgs[8]
    # It is assumed that the input variable is normalized using pre-defined bounds
    # in such a way that its maximum value is 1 and its lowest value is 0.
    options = {'tolfun':1e-11,'tolx':1e-11,'seed':1, 'bounds':[[0]*dim,[1]*dim],
               'popsize':popSize, 'verb_filenameprefix':savePath} 
    # define arguments for the objective function
    args = (ksi_u, activeLearningProblem.inputTrainingSample, GPTrained, 
            karhunenLoeveLiftingFunction, eigenFunctions, eigenValues, areaArgs, 
            areaFunction)
    # intialize evloution strategy in the central point (0.5) with initial 
    # variance of 0.3 so that 
    es = cma.CMAEvolutionStrategy([0.5]*dim, 0.3, options)
    # run the optimizer 
    es.optimize(obj_fun, iterations = iterations, args = args, verb_disp=1, n_jobs=njobs)
    # return the input variable realization that minimizes the CI Area
    return es.result.xbest
