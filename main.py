# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:28:56 2021
===============================================================================
=========== Active learning strategy for quantile refinement ==================
===============================================================================
This code estimates a quantile of a unidemnsional field variable on an stochastic
process. It uses the Karhunen-Loève loeve (KL) decomposition to reduce the 
dimensonality of the stochastic process and predict new outputs based on a 
surrogate model  of the uncertain variables of the KL decomposition using
Gaussian process (GP). The active learning technique looks to reduce the 
vairance of the GPs by identifying in an optimization loop the new training samples
that contribute the most for this purpose.

Two example cases are presented in the script "ActiveLearningProblem".
A simple case using the Onera Damage Model for Composite Material with Ceramix 
Matrix (ODM-CMC) and more complex launch vehicle multidisciplinary optimization
case based on the Dymos and OpenMDAO environments.

To define a new  "ActiveLearningProblem" see the ""ActiveLearningProblem.py""
script.

@author: jorge
"""


from optimization import run_opt
from calculateCIArea import CIAreaA, CIAreaB
from metamodels import KLDecomposition, GPMetamodel, FullMetamodel
from activeLearningProblem import ODMActiveLeaningProblem, dymosActiveLeaningProblem

import pickle
from pathlib import Path
from copy import deepcopy


#%% ========================= settings ==========================================

# define the number of samples to be generated from the PDF of the input random
# vector U that are to used for the estimation of the quantile
sizeOfQuantileEsimationSample = 500
# define the number of Karhunen Loeve modes to be included in the truncated base
numberOfKLModes    = 6
# define how many samples are to be added to improve the quality of the surrogate model
enrichmentRuns     = 5
# define an scaling factor for the centered output trajectories. 
# For instance, if the CENTERED output trajectories for the output radius
# are in the order of 10^3 meters it is convenient to define outputScaler = 1/1e3
outputScaler       = 1/1e3
# ========================= confidence area calculation settings ==============
# define the confidence interval area function to be used for optimization 
# and reporting the results. Options are:
# CIAreaA: expensive method with double loop based on GP trajectories
# CIAreaB: cheap method based on propagation of GP error through the KL reduction
reportAreaFunction = CIAreaB
optimAreaFunction  = CIAreaB
# Settings for CIAreaA method =================================================
# define eta in [0,1] to estimate the eta-quantile of the stochastic process
etaA               = 0.99
# define the significance level of the confidence interval of the quantile estimation
# for "CI area - A" method CIEta in [0.0,1.0]
CIEta_CIAreaA      = 0.95
# Define the number of analyses or Gaussian process trajectories to be used 
# in the quantile estimation. Only significant for "CI area - A" method
NumberOfAnalyses     = 100
# Settings for CIAreaB method =================================================
# define the multiplier for the standard deviation for the "CI area - B" method.
# computes quantiles using the mean +- Nsigma * sigma realizations. [1,2,3,4]
NSigma_CIAreaB     = 2
# define eta in [0.0,1.0] to estimate the eta-quantile of the stochastic process
etaB               = 0.99
# =============== optimization settings =======================================
# the code is based on the CMA-ES optimizer.
# define the population size. 
# recommended value : 4 + np.floor(3*np.log(number of uncertain variables))
popSize              = 8
# define the number of jobs or processes to be used by CMA. 
# 0 for serial execution on Windows systems
# 1 or more for parallel execution on Linux based systems. Parallel efficiency
# is low and has to improoved. Not worth parallelizing cheap analyses.
njobs                = 0
# define the number of iterations of CMA-ES. use around 50 for popSize 9
iterations           = 40
# =============== Import active learning problem ==============================
pb = ODMActiveLeaningProblem(sizeOfQuantileEsimationSample, 0)
# pb = dymosActiveLeaningProblem(sizeOfQuantileEsimationSample, 0)
# define normalization. Comment the line if normalization is not desired
# normalize inputs
pb.normalizeInputSamples()
# center and scale training output sample
pb.normalizeOutputSample(outputScaler)
# ================ define results folder ======================================
resultsPath = 'results/' 
# =============================================================================

#%% 
# create results folder and parents if they don't exist
Path(resultsPath).mkdir(parents=True, exist_ok=True)
# pack output scaling arguments
outputScalingArgs = [pb.outputMean, pb.outputScaler]

# add new samples to the training set iteratively
for i in range(enrichmentRuns+1):
    # perform KL decomposition. 
    karhunenLoeveLiftingFunction, ksi_u, eigenFunctions, eigenValues = \
        KLDecomposition(pb.outputTrainingSample, numberOfKLModes, i, resultsPath, 
                        test = False)
    # train GPs 
    GPTrained = GPMetamodel( ksi_u, pb.inputTrainingSample )
    # build full metamodel consisting of the KL reduction + the GPs predictions
    fullMetamodel = FullMetamodel( karhunenLoeveLiftingFunction, GPTrained, 
                                    eigenFunctions, eigenValues )
    # only for testing and check algorithm progress - comment for fast execution
    # =========================================================================
    # fullMetamodel.testMeanRealization(  pb.quantileEstimationInputSample, 
    #                                     pb.outputTrainingSample, 
    #                                     pb.mesh, i, resultsPath ) 
    # fullMetamodel.testSigmaModel( pb.quantileEstimationInputSample, 
    #                               pb.mesh, resultsPath )
    # fullMetamodel.testSigmaModelMulti( pb.quantileEstimationInputSample, 
    #                                     pb.mesh, resultsPath, i)
    # fullMetamodel.testSigmaModelMultiExtended( pb.quantileEstimationInputSample, 
    #                                             pb. mesh, resultsPath )
    # =========================================================================
    
    # pack area arguments
    areaArgs = [pb.mesh, pb.quantileEstimationInputSample,CIEta_CIAreaA, 
                NSigma_CIAreaB, etaA, etaB, NumberOfAnalyses, i, True, True, 
                outputScalingArgs, resultsPath] 
    # calculate and plot the confidence interval area 
    CIArea = reportAreaFunction( fullMetamodel, areaArgs )
    # run optimization loop if more samples are to be added using enrichment 
    if i < enrichmentRuns :
        # change the Test and save variables of the CIArea method to False so that no
        # tests  are runned nor info saved during optimization
        areaArgs[-3] = False
        areaArgs[-4] = False
        optArgs = [popSize, njobs, iterations]
        # run optimization loop
        u_star = run_opt( ksi_u, pb, GPTrained, karhunenLoeveLiftingFunction, 
                          eigenFunctions, eigenValues, areaArgs, 
                          optimAreaFunction, optArgs)
        # add the optimal virtual training sample by evaluating the original function   
        pb.enrich(u_star, i)
    # Savoid optimization if all the enrichment runs were performed
    elif i == enrichmentRuns:
        print('Done with all enrichmentRuns. %i samples were added.'%(i))
    # save problem and metamodel using pickle. Overwrite each time that a new
    # point is added to the training samples.
    # the attribute "function" of an ActiveLearningProblem instance cannot be pickled
    aux_pb = deepcopy(pb)
    # delete the 'function' attribute
    delattr(aux_pb, 'function')
    pickle.dump(aux_pb, open(resultsPath + 'activeLearningProblem.p', "wb"))
    pickle.dump(fullMetamodel, open(resultsPath + 'fullMetamodel.p', "wb"))
    
