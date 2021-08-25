# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:37:40 2021

@author: jorge
"""

from __future__ import print_function
import openturns as ot
import numpy as np
from Y import calcul_comportement
import json
# import codecs

# define function to save a smaple set in json format in txt file
def saveSampleSet(results, fileName):
    "This function saves the sample in a txt file"

    # convert np.arrays into lists
    results = results.tolist()
        
    # write json file
    exDict = {'resultsDict': results}
    
    with open(fileName, 'w') as file:
         file.write(json.dumps(exDict))
         file.close()
         
def saveOutputSampleSet(results, mesh, fileName):
    "This function saves the sample in a txt file"

    # convert np.arrays into lists
    results = results.tolist()
    
    # save mesh
    mesh = {'simplices' : np.array(mesh.getSimplices()).tolist(),
            'vertices' : np.array(mesh.getVertices()).tolist()}
        
    # write json file
    exDict = {'resultsDict': results,
              'mesh':mesh}
    
    with open(fileName, 'w') as file:
         file.write(json.dumps(exDict))
         file.close()
         
# define fucntion to generate and save smaples
def generateSamples(N_samples, mesh, inputName, outputName, distribution_U, Func_OT):
    samples_U = distribution_U.getSample(N_samples)
    samples_U = np.array(samples_U) #+  np.array(std_vector) * 3

    ###### Call to the exact field function over the training samples
    inputSample = ot.Sample(samples_U)
    
    outputSample = Func_OT(inputSample)
    
    saveSampleSet(np.array(inputSample), inputName)
    
    saveOutputSampleSet(np.array(outputSample), mesh, outputName)
    

#%% 
# Problem setup
def generateDistributionAndFunction():         
    #### ODM model is dedicated to woven ceramic matrix composites. 
    #### It describes an elastic damage behaviour: the constraint in the material (sigma) as a function of the deformation (epsilon) and the model parameters (U)
    #### sigma = f(epsilon,U)  with sigma in MPa, epsilon (no unit) and U defined below
    #### The model parameters set are U = [E0, ycs, y0s, dc] with
    #### E0: the Young's modulus in MPa, ycs: the damage evolution celerity (MPa), y0s: the damage threshold (MPa) and dc: the damage saturation (no unit)
    #### As U is an uncertain vector, sigma is a stochastic process.
    ### Definition of the mesh in terms of deformation epsilon
    eps_min=0.0 # Min deformation
    eps_max=0.009 # Maximum time
    gridsize=100 # Number of discretization of the deformation epsilon
    mesh = ot.IntervalMesher([gridsize-1]).build(ot.Interval(eps_min, eps_max))
    
    vertices = mesh.getVertices()
    
    deformations_vertices = np.array(vertices)
    deformations_vertices = np.reshape(np.array(deformations_vertices),len(deformations_vertices,))
    
    #### Definition of the uncertainty caracterizing the input parameters U
    mean_vector =  [1.80e+05, 5.000e-03, 4.27e+00, 5.33e+00]
    std_vector  = np.array([2.0e4,8.0e-5,0.6,0.2]) * 1.8
    marginals = [ot.Normal(mean_vector[0],std_vector[0]), ot.Normal(mean_vector[1],std_vector[1]),
                                   ot.Normal(mean_vector[2],std_vector[2]), ot.Normal(mean_vector[3],std_vector[3])]
    distribution_U = ot.ComposedDistribution(marginals)
    
    ######## Creation of a field function compatible with OT
    def func(deformations_vertices,x):
        yobs = calcul_comportement(deformations_vertices,x)
        results = yobs[:,np.newaxis].tolist()
        return results

    inputDimension = distribution_U.getDimension()
    outputDimension = 1
    
    Func = lambda x:  func(deformations_vertices,x)
    Func_OT = ot.PythonPointToFieldFunction(inputDimension, mesh, outputDimension, Func)
    
    return distribution_U, Func_OT, mesh

def createSamples(seed, numberOfTrainingSamples, numberOfValidationSamples):
    # fix random seed 
    ot.RandomGenerator.SetSeed(seed)
    # generate distribution, function  and mesh
    distribution_U, Func_OT, mesh = generateDistributionAndFunction()
    # generate and save training and validation samples
    generateSamples(numberOfTrainingSamples, mesh, 'samples/inputTrainingSample.txt', 
                    'samples/outputTrainingSample.txt', distribution_U, Func_OT)
    generateSamples(numberOfValidationSamples, mesh, 'samples/inputValidationSample.txt', 
                    'samples/outputValidationSample.txt', distribution_U, Func_OT)
    # print message 
    print('Input and output samples were created')
    
    
# create samples
if __name__ == '__main__':
    createSamples(0, 50, 1000)