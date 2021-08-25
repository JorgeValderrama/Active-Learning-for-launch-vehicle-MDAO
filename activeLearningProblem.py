# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 14:05:39 2021

This script define the class ActiveLearningProblem.
Two examples are given where a pythin function is given to instanciate the class
in this script and pass only the instanciated object to the main file.
A simple case using the Onera Damage Model for Composite Material with Ceramix 
Matrix (ODM-CMC) and a more complex launch vehicle multidisciplinary optimization
case based on the Dymos and OpenMDAO environments.

@author: jorge
"""
import openturns as ot
import numpy as np
import random

from launcherMDAO_Dymos.createStateSample import createStateSample
from launcherMDAO_Dymos.importSamples import ALValidationSamples
from launcherMDAO_Dymos.main_opt_traj import opt_traj
from launcherMDAO_Dymos.interpolate_results_normalized import load_result_and_interpolate

from ONERADeformationModel.Y import calcul_comportement

class ActiveLearningProblem():
    def __init__(self, function, composedDistribution, uBounds, inputTrainingSample,
                 outputTrainingSample, mesh, sizeOfQuantileEstimationSample, seed):
        """
        When instantiated, this class contains the parameters to define 
        an active learning problem and generates the quantile estimation 
        input sample based on the 'composedDistribution' of the input vector 
        an the random 'seed'. It computes normalization parameters of the
        input samples based on their bounds ('uBounds').

        Parameters
        ----------
        function : function
            DESCRIPTION. python function wrapping an openturns.func.PointToFieldFunction
            This warrping allows to use extra arguments as the "index" necessary
            to create different folders for the output results of the function
        composedDistribution : openturns.model_copula.ComposedDistribution
            DESCRIPTION. Joint distribution of the input variable vector
        uBounds : Dict
            DESCRIPTION. dictionary containing numpy.arrays of the upper and lower
            bounds.
        inputTrainingSample : openturns.Sample
            DESCRIPTION. The realizations of the input random variable vector
            that are used for training
        outputTrainingSample : openturns.ProcessSample
        DESCRIPTION. OpenTurns object representing the stochastic process
        mesh : openturns.geom.Mesh
            DESCRIPTION. mesh of  the output stochastic process
        sizeOfQuantileEstimationSample : Int
            DESCRIPTION. size of the quantile estimation sample to be generated
        seed : Int
            DESCRIPTION. random seed to control the sampling of the 
            composedDistribution

        Returns
        -------
        None.

        """
        self.function = function
        self.composedDistribution = composedDistribution
        self.uBounds = uBounds
        self.inputTrainingSample  = inputTrainingSample
        self.outputTrainingSample = outputTrainingSample
        self.mesh = mesh
        # generate quantile estimation input samples
        ot.RandomGenerator.SetSeed(seed)
        self.quantileEstimationInputSample = \
            composedDistribution.getSample(sizeOfQuantileEstimationSample)
        # calculate normalization parameters
        self.adder  = -self.uBounds['lb']
        self.scaler = 1 / (self.uBounds['ub'] - self.uBounds['lb'])
        self.inputSamplesSpace = 'standard space'
        self.outputSamplesSpace = 'standard space'
        self.outputScaler = 1
        self.outputMean = 0
        # print information to the console
        space = 40
        print('============= Active Learning problem =============')
        print('Dimension of training input sample: '.ljust(space) + \
              str(inputTrainingSample.getDimension()))
        print('Size of training input sample: '.ljust(space) + \
              str(inputTrainingSample.getSize()))
        print('Generated quantile estimation samples: '.ljust(space) + \
              str(self.quantileEstimationInputSample.getSize()))
        print('===================================================')

        
    def normalizeInputSamples(self):
        """
        This method normalizes the input samples only if they area in 
        'standard space'.
        
        adder  = -lb
        scaler = 1 / (ub- lb)
        
        x_norm = (x + adder) * scaler

        Returns
        -------
        None.

        """
        if self.inputSamplesSpace == 'standard space':
            # normalize input training sample
            for i in range(self.inputTrainingSample.getSize()):
                self.inputTrainingSample[i] = ot.Point( np.array((self.inputTrainingSample[i] + self.adder)) * self.scaler )
            # normalize quantile estimation input sample
            for i in range(self.quantileEstimationInputSample.getSize()):
                self.quantileEstimationInputSample[i] = ot.Point( np.array((self.quantileEstimationInputSample[i] + self.adder)) * self.scaler )
            self.inputSamplesSpace = 'normalized space'
        else:
            print("""
                  
                  You are trying to normalize something that is normalized already
                  
                  """)
        
    def deNormalizeInputSamples(self):
        """
        This method de-normalizes the input samples only if they are in 
        'normalized space'.
        
        adder  = -lb
        scaler = 1 / (ub- lb)
        
        x = x_norm / scaler - adder 

        Returns
        -------
        None.

        """
        if self.inputSamplesSpace == 'normalized space':
            # normalize input training sample
            for i in range(self.inputTrainingSample.getSize()):
                self.inputTrainingSample[i] = ot.Point( np.array(self.inputTrainingSample[i]) / self.scaler - self.adder )
            # normalize quantile estimation input sample
            for i in range(self.quantileEstimationInputSample.getSize()):
                self.quantileEstimationInputSample[i] = ot.Point( np.array(self.quantileEstimationInputSample[i]) / self.scaler - self.adder )
            self.inputSamplesSpace = 'standard space'
        else:
            print("""
                  
                  You are trying to denormalize something that is in standard space
                  
                  """)
        
    def normalizeOutputSample(self, outputScaler):
        """
        This method normalizes the output samples only if they are in 
        'standard space'.
        
        x_norm = (x - mean) / outputScaler

        Parameters
        ----------
        outputScaler : Float
            DESCRIPTION. scaler for the output

        Returns
        -------
        None.

        """
        if self.outputSamplesSpace == 'standard space':
            self.outputScaler = outputScaler
            self.outputMean   = self.outputTrainingSample.computeMean()
            for i in range(self.outputTrainingSample.getSize() ):
                self.outputTrainingSample[i] = (self.outputTrainingSample[i] - \
                                               self.outputMean ) * outputScaler
            self.outputSamplesSpace = 'normalized space'
        else:
            print("""
                  
                  You are trying to normalize something that is  normalized already
                  
                  """)
        
    def deNormalizeOutputSample(self):
        """
        This method de-normalizes the output samples only if they are in 
        'normalized space'.
        
        x = x_norm * outputScaler + mean 

        Returns
        -------
        None.

        """
        if self.outputSamplesSpace == 'normalized space':
            for i in range(self.outputTrainingSample.getSize() ):
                self.outputTrainingSample[i] = (self.outputTrainingSample[i] / \
                                               self.outputScaler ) + self.outputMean
            self.outputSamplesSpace = 'standard space'
        else:
            print("""
                  
                  You are trying to denormalize something that is in standard space
                  
                  """)
        
    def enrich(self, u_normalized, index):
        """
        This method takes as input the value of u tha was found by the optimizer
        and evaluates the original function to obtain its response. 
        the input (u) and its response are added to the training sets.

        Parameters
        ----------
        u_normalized : numpy.ndarray
            DESCRIPTION. new input point found by the optimizer that will
            enrich the training sets
        index : Int
            DESCRIPTION. iteration number. Serves to define folders of results 
            for each iteration and store outputs of the function evaluation

        Returns
        -------
        None.

        """
        # take u_normlaized to standard space if necessary
        if self.inputSamplesSpace == 'normalized space':
            u = u_normalized / self.scaler -self.adder
        elif self.inputSamplesSpace == 'standard space':
            u = u_normalized 
        # evalaute the function to obtain f(u)
        field = self.function(ot.Sample([u]), index)[0]
        # normalize the output if necessary
        if self.outputSamplesSpace == 'normalized space':
            normalized_field = ( field- self.outputMean ) * self.outputScaler
        elif self.outputSamplesSpace == 'standard space':
            normalized_field = field
        # add new samples to the training sets
        self.inputTrainingSample.add(u_normalized)
        self.outputTrainingSample.add(normalized_field)
        
        
#%% =========================================================================== 
def dymosActiveLeaningProblem(sizeOfQuantileEstimationSample, seed):
    # =================== define Dymos Active Learning Problem ====================
    # =============================================================================   
    # composed distribution for input samples u
    marg_Isp1 = {'distribution': 'normal', 'params':(0,1)}         # normal
    marg_Isp2 = {'distribution': 'normal', 'params':(0,1)}         # normal
    marg_m1   = {'distribution': 'uniform', 'params':(-750,750)}   # uniform
    marg_m2   = {'distribution': 'uniform', 'params':(-250,250)}   # uniform
    marg_q1   = {'distribution': 'normal', 'params':(0,5)}         # normal
    marg_q2   = {'distribution': 'normal', 'params':(0,5)}         # normal
    marg_CD   = {'distribution': 'uniform', 'params':(-0.05,0.05)} # uniform
    
    uncertaiVars_list = [marg_Isp1, marg_Isp2, marg_m1, marg_m2, marg_q1, marg_q2, marg_CD]
    # =========================== define composed distribution ====================
    marginals = []
    for uncertainVar in uncertaiVars_list:
        if uncertainVar['distribution'] == 'normal':
            marginals.append(ot.Normal(uncertainVar['params'][0], 
                                       uncertainVar['params'][1] ))
        if uncertainVar['distribution'] == 'uniform':
            marginals.append(ot.Uniform(uncertainVar['params'][0], 
                                        uncertainVar['params'][1] ))
    composedDistribution = ot.ComposedDistribution(marginals)
    # =============================================================================
    # =========================== define optimization bounds ======================
    # create lists with upper and lower bounds
    lb = []
    ub = []
    # define sigma level for normal variable optimization bounds
    NSigma = 4 
    for u in uncertaiVars_list:
            if u['distribution'] == 'normal':
                mean, sigma = u['params']
                lb.append(mean - NSigma * sigma)
                ub.append(mean + NSigma * sigma)
            elif u['distribution'] == 'uniform':
                a, b = u['params']
                lb.append(a)
                ub.append(b)
    uBounds = {'lb':np.array(lb), 'ub':np.array(ub)}
    # =============================================================================
    # =========================== define training samples =========================
    totalSamples      = 1130             
    partitionTraining = 230
    # define whether to use the interpolated results using the pseudospectral nodes
    # (implies normalized time) or the simulate method based on Runge-Kutta (all 
    # trajectories are extended during the coast phase and truncated at 430 s ).
    # options:  'pseudospectral' or 'simulate'
    solutionType = 'pseudospectral'
    # define the trajectory state to be surrogated. info on the following script:
    # launcherMDAO_Dymos\plotState_sim
    # some state options: 'r', 'v', 'q_heat'
    state = 'r'
    resultsPath = 'results/' 
    sizeTrainingSample   = 50
    # shuffle the samples from Dymos
    indices = list(range(totalSamples))
    random.Random(0).shuffle(indices)
    createStateSample(np.array(indices[0:partitionTraining]),
                      np.array(indices[partitionTraining:totalSamples]), 
                      state, solutionType, resultsPath)
    # fix random seed for training
    ot.RandomGenerator.SetSeed(0)
    # import training samples
    trainingSamples = ALValidationSamples(resultsPath + '/inputTrainingSample.txt', 
                                          resultsPath + '/outputTrainingSample.txt', 
                                          sizeTrainingSample)
    
    # =============================================================================
    # =========================== define function =================================
    # =============================================================================

    def dymosFunction(u, index):
        
        if solutionType == 'pseudospectral':
            dymosSolution = 'interpolated_results'
        elif solutionType == 'simulate':
            dymosSolution = 'interpolated_resultsSim'
        # =================================================
            
        def opt_traj_dymos(u):
            opt_traj(u, str(index), resultsPath)
            results_dict = load_result_and_interpolate(resultsPath + '/'+ str(index)+'_trajectory_state_history.txt')
            results = results_dict[dymosSolution][state]  
            return ot.Sample(np.array([results])[0])
        
        # define field function from OT
        inputDimension  = trainingSamples.input.getDimension()
        outputDimension = 1
        # vertices        = np.array(self.mesh.getVertices())
        
        Func_OT = ot.PythonPointToFieldFunction(inputDimension, trainingSamples.mesh, 
                                                outputDimension, opt_traj_dymos )
        
        return Func_OT(u)
    
    # ========================== instanciate Problem ==============================
    return ActiveLearningProblem(dymosFunction, composedDistribution, uBounds, 
                                 trainingSamples.input, trainingSamples.output, 
                                 trainingSamples.mesh, sizeOfQuantileEstimationSample, seed)

#%% =========================================================================== 
def ODMActiveLeaningProblem(sizeOfQuantileEstimationSample, seed):
    # =================== define ODM Active Learning Problem ====================
    # =============================================================================   
    # composed distribution for input samples u
    mean_vector =  [1.80e+05, 5.000e-03, 4.27e+00, 5.33e+00]
    std_vector  = np.array([2.0e4,8.0e-5,0.6,0.2]) * 1.8
    marginals = [ot.Normal(mean_vector[0],std_vector[0]), 
                 ot.Normal(mean_vector[1],std_vector[1]),
                 ot.Normal(mean_vector[2],std_vector[2]), 
                 ot.Normal(mean_vector[3],std_vector[3])]
    composedDistribution = ot.ComposedDistribution(marginals)
    # =============================================================================
    # =========================== define optimization bounds ======================
    # create lists with upper and lower bounds
    lb = []
    ub = []
    # define sigma level for normal variable optimization bounds
    NSigma = 4 
    for i in range(len(mean_vector)):
        lb.append(mean_vector[i]- NSigma * std_vector[i])
        ub.append(mean_vector[i]+ NSigma * std_vector[i])
    uBounds = {'lb':np.array(lb), 'ub':np.array(ub)}
    # =============================================================================
    # =========================== define mesh =====================================
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
    # define deformation vertices
    vertices = mesh.getVertices()
    deformations_vertices = np.array(vertices)
    deformations_vertices = np.reshape(np.array(deformations_vertices),
                                       len(deformations_vertices,))
    # =============================================================================
    # =========================== define function =================================
    # =============================================================================
    
    def ODMFunction(u, index):
        
        def ODM(u):
            yobs = calcul_comportement(deformations_vertices, u)
            results = yobs[:,np.newaxis].tolist()
            return results 
        
        # define field function from OT
        inputDimension  = len(mean_vector)
        outputDimension = 1
        Func_OT = ot.PythonPointToFieldFunction(inputDimension, mesh, 
                                                outputDimension, ODM )
        return Func_OT(u)
    
    # =========================== define training samples =========================
    sizeOfTrainingSamples = 4
    inputSample = composedDistribution.getSample(sizeOfTrainingSamples)
    inputSampleArray = np.array(inputSample)
    outputSample = ODMFunction(inputSample, None)
    # =============================================================================
    # ========================== instanciate Problem ==============================
    return ActiveLearningProblem(ODMFunction, composedDistribution, uBounds, 
                                 inputSample, outputSample, mesh, 
                                 sizeOfQuantileEstimationSample, seed)