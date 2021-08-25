# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:35:17 2021

@author: jorge
"""

import openturns as ot
import numpy as np
import openturns.viewer as viewer
import matplotlib
# matplotlib.use('agg', force=True)
import matplotlib.pyplot as plt
# print ("Switched to:",matplotlib.get_backend())
import scipy.stats as st
from launcherMDAO_Dymos.importSamples import ALValidationSamples
from sklearn.metrics import mean_squared_error 

# attention! this code is to prevent multiple messages regarding the cov
# scale when running on Spiro. It can prevent the display of important warnings.
ot.Log.Show(ot.Log.NONE)


# define function to calculate average Q2 
def averageQ2(validationOutputSample, OutputSample_metamodel):
    """
    
    Parameters
    ----------
    validationOutputSample : openturns.ProcessSample
        DESCRIPTION. the original samples
    OutputSample_metamodel : openturns.ProcessSample
        DESCRIPTION. the samples computed with a metamodel

    Returns
    -------
    Q2: Float
        DESCRIPTION. Computed predictivity factor 

    """
    # store residuals in array
    residuals = np.asmatrix(np.array(validationOutputSample) - 
                            np.array(OutputSample_metamodel))
    # square and sum residuals
    aux = np.square(residuals)
    aux = aux.sum(1)
    # compute Q2 for each realization
    Q2_vec = 1 - aux / np.var(np.array(OutputSample_metamodel),1)
    return np.mean(Q2_vec)

def KLDecomposition(outputTrainingSample, numberOfModes, iterationNumber, \
                    resultsPath, test):
    """
    This function performs the Karhunen-Loeve decompostion of the stochastic 
    process represented by the "outputTrainingSample" and truncates the base
    at a given "numberOfModes"
    

    Parameters
    ----------
    outputTrainingSample : openturns.ProcessSample
        DESCRIPTION. OpenTurns object representing the stochastic process
    numberOfModes : Int
        DESCRIPTION. number of modes used for the KL decomposition
    iterationNumber : Int
        DESCRIPTION. number of the enrichment iteration. Use for the labeling 
        of results
    resultsPath : Str
        DESCRIPTION. string indicating the folder to save the results
    test : Bool
        DESCRIPTION. wether to test or not the KL expansion by projecting
        the training samples back and forth and calculate the predictivity 
        factor. A plot with results is saved in the "resultsPath"

    Returns
    -------
    karhunenLoeveLiftingFunction : openturns.algo.KarhunenLoeveLifting
        DESCRIPTION. Function to be used in tandem with a vector of random 
        coefficients to create a Karhune Loeve field
    ksi_u :  openturns.typ.Sample
        DESCRIPTION. set of random coefficients of the karhunen Loeve expansion
        associated to each KL mode and each item of the training set
    eigenFunctions : openturns.func.ProcessSample
        DESCRIPTION. eigen functions of the resulting KL decomposition
    eigenValues : openturns.typ.Point
        DESCRIPTION. eigen values of the resulting KL decomposition

    """
    # set algorithm to truncate the expansion with a given numberOfModes
    algo  = ot.KarhunenLoeveSVDAlgorithm(outputTrainingSample, 0)
    algo.setNbModes(numberOfModes)
    # run and extract KL result
    algo.run()
    KLResult  = algo.getResult()
    # project the training output sample and extract random coefs (\xi(u))
    karhunenLoeveLiftingFunction = ot.KarhunenLoeveLifting(KLResult)
    ksi_u = KLResult.project(outputTrainingSample)
    # extract eigen functions and eigenvalues
    eigenFunctions = KLResult.getModesAsProcessSample()
    eigenValues    = KLResult.getEigenValues()
    # perform test using the training sample.
    # project the trainin samples back and forth and compare with the 
    # original samples to assess the error of the KL projection truncated
    # at the given number of modes.
    if test == True:
        # karhunen loeve validation
        validation = ot.KarhunenLoeveValidation(outputTrainingSample, KLResult)
        # define a projection function
        reductionFunction  = ot.FieldToFieldConnection(
            ot.KarhunenLoeveLifting(KLResult), 
            ot.KarhunenLoeveProjection(KLResult))
        # project validation data
        projectedTrainingOutputSample = reductionFunction(outputTrainingSample)
        # calculate the average predicitivity factor, Q2
        avg_Q2 = averageQ2(outputTrainingSample, projectedTrainingOutputSample)
        # plot validation graph
        plt.rcParams.update({'font.size': 12})
        graph = validation.drawValidation()
        size  = outputTrainingSample.getSize()
        size2 = outputTrainingSample.getSize()
        graph.setTitle("Average Q2=%.3f%%" % (avg_Q2*100) + 
                       ' with training sample of size: %i'%size+
                       '\n Number of modes = %i.' %numberOfModes +
                       ' Training sample of size: %i'%size2 )
        view = viewer.View(graph)
        savePath = resultsPath+'/%i_average_Q2'%iterationNumber
        plt.ioff()
        plt.savefig(savePath, dpi=400, bbox_inches='tight')
        # karhune loeve scaled modes
        KLScaledMdes = KLResult.getScaledModesAsProcessSample()
        graph2 = KLScaledMdes.draw()
        view = viewer.View(graph2)
        savePath2 = resultsPath+'/%i_scaledModes'%iterationNumber
        plt.ioff()
        plt.savefig(savePath2, dpi=400, bbox_inches='tight')
        # ================== plot residuals ===================================
        residual = validation.computeResidual()
        vertices = np.array(residual.getMesh().getVertices())[:,0]
        plt.figure()
        plt.rcParams.update({'font.size': 16})
        plt.plot(vertices, projectedTrainingOutputSample[-1], 'b',
                  label = r'projected training samples. $N_k: %i$'%numberOfModes)
        for i in range(residual.getSize()):
            plt.plot(vertices, projectedTrainingOutputSample[i], 'b')
        plt.plot(vertices, residual[-1], 'r', 
                  label = r'residuals. Average $Q_2: %1.4f$'%avg_Q2)
        for i in range(residual.getSize()):
            plt.plot(vertices, residual[i], 'r')
        # plt.ylabel('centered height (km)')
        # plt.ylim([-12,15])
        # plt.xlabel(r'% of flight')
        plt.legend(loc = 'lower right', fontsize = 12)
        plt.savefig(resultsPath + str(numberOfModes) +'_KLResiduals.pdf', 
                    format = 'pdf', bbox_inches='tight')
        # # ================== plot residuals validation=========================
        # # ======= modify function KLDecomposition to take "pb" as input =======
        # validationSamples = ALValidationSamples(resultsPath + '/inputValidationSample.txt', 
        #                                         resultsPath + '/outputValidationSample.txt', 
        #                                         900)
        # for i in range(validationSamples.output.getSize()):
        #     validationSamples.output[i] = (validationSamples.output[i] - pb.outputMean)*pb.outputScaler
        
        # # karhunen loeve validation
        # validation = ot.KarhunenLoeveValidation(validationSamples.output, KLResult)
        # # project validation data
        # projectedValidationOutputSample = reductionFunction(validationSamples.output)
        # # calculate the average predicitivity factor, Q2
        # avg_Q2 = averageQ2(validationSamples.output, projectedValidationOutputSample)
        # residual = validation.computeResidual()
        # plt.figure()
        # plt.rcParams.update({'font.size': 16})
        # plt.plot(vertices, projectedValidationOutputSample[-1], 'b',
        #           label = r'projected training samples. $N_k: %i$'%numberOfModes)
        # for i in range(residual.getSize()):
        #     plt.plot(vertices, projectedValidationOutputSample[i], 'b')
        # plt.plot(vertices, residual[-1], 'r', 
        #           label = r'residuals. Average $Q_2: %1.4f$'%avg_Q2)
        # for i in range(residual.getSize()):
        #     plt.plot(vertices, residual[i], 'r')
        # plt.ylabel('centered height (km)')
        # plt.ylim([-12,15])
        # plt.xlabel(r'% of flight')
        # plt.legend(loc = 'lower right', fontsize = 12)
        # plt.savefig(resultsPath + str(numberOfModes) +'_KLResidualsValidation.pdf', 
        #             format = 'pdf', bbox_inches='tight')
        # # quantile error estimation after projection
        # plt.figure()
        # Qval  = projectedValidationOutputSample.computeQuantilePerComponent(0.99)
        # Qreal = validationSamples.output.computeQuantilePerComponent(0.99)
        # plt.plot(np.array(Qval), 'k')
        # plt.plot(np.array(Qreal), '-.r')
        # plt.title(str(mean_squared_error(Qval, Qreal)**0.5))
        
            
    return karhunenLoeveLiftingFunction, ksi_u, eigenFunctions, eigenValues


class GPMetamodel():
    """
    This class is a container for the Gaussian Processes (GP) used to predict
    the random variables of the Karhunen Loeve expansion. A GP is trained
    for each KL mode.
    Two training modes are defined:
    GPTraining : standard training
    GPVirtualtraining: is the training used for the active learning technique
    when the optimizer proposes a new input (u) and the training set is expanded
    """
    def __init__(self, ksi_u, inputSample):
        """
        When the class is initalized, it calls the GPTraining method
        to train all the GPs and store the results in a list

        Parameters
        ----------
        ksi_u : openturns.typ.Sample
            DESCRIPTION. set of random coefficients of the karhunen Loeve expansion
        associated to each KL mode and each item of the training set
        inputSample : openturns.Sample
            DESCRIPTION. the realizations of the input random variable vector
            used for training
        
        Returns
        -------
        None.

        """
        # train the GPs and store openturns objects with results in a list.
        # this list is changed during virtual training
        self.GPResult_list = self.GPTraining(ksi_u, inputSample)
        # this list is not changed during virtual training, hence it can be
        # be used for the prediction necessary to update the input during 
        # virtual training
        self.GPResult_list_freezed = self.GPResult_list[:]
        # get dimension of input sample (ksi_u). equivalent to number of KL modes
        self.KLModes = ksi_u.getDimension()
        # get number of uncertain variables
        self.numberUncertainVars = inputSample.getDimension()
    
    def GPTraining(self, ksi_u, inputSample ):
        """
        Traing the GPs associated for the prediction of the KL randomu variables
        Stores the basis and the covariance model, so that the same are used
        for the active learning virtual training.
        Stores the resulting amplitudes and scales so that they can be used 
        during the active learning virtual training.

        Parameters
        ----------
        ksi_u : openturns.typ.Sample
            DESCRIPTION.set of random coefficients of the karhunen Loeve expansion
        associated to each KL mode and each item of the training set
        inputSample : openturns.Sample
            DESCRIPTION. the realizations of the input random variable vector
            used for training

        Returns
        -------
        GPResult_list : List of openturns.KrigingResult objects
            DESCRIPTION. Contains the results of trained GPs

        """
        # exclude the "amplitude" from the optimization variables by estimating it 
        # analytically
        ot.ResourceMap.SetAsBool(
            'GeneralLinearModelAlgorithm-UseAnalyticalAmplitudeEstimate', True)
        # get dimension of input sample (ksi_u). equivalent to number of KL modes
        KLModes = ksi_u.getDimension()
        # get the  size of the input sample (ksi_u). equivalent to the number
        # of uncertain variables
        numberUncertainVars = inputSample.getDimension()
        # define solver
        solver = ot.NLopt('LN_COBYLA')
        # in order to define multiple starting point, lower and upper bounds are
        # obtained and used as the bounds of a normal distribution that is later
        # sampled to obtain multiple points.
        # define bounds for initialization of parameters
        # lowerBounds = inputSample.getMin()
        # upperBounds = inputSample.getMax()
        # scaleOptimizationBounds = ot.Interval(lowerBounds, upperBounds)
        lb, ub = [0.001, 10]
        scaleOptimizationBounds = ot.Interval([lb/100]*(numberUncertainVars), \
                                              [ub*100]*(numberUncertainVars))
        # define distributions of each paramter
        distributionList = []
        for i in range(numberUncertainVars):
            # distributionList.append(ot.Uniform(lowerBounds[i], upperBounds[i]))
            distributionList.append(ot.Uniform(lb, ub))
        # sample the distribution to obtain multiple starting points
        distribution = ot.ComposedDistribution(distributionList)
        lhsExperiment = ot.LHSExperiment(distribution, 10) 
        # define list to store the results object of the GP metamodels
        GPResult_list = []
        self.amplitudes = []
        self.scales = []
        # define kernel models
        self.basis = ot.ConstantBasisFactory(numberUncertainVars).build()
        self.covarianceModel = ot.SquaredExponential(numberUncertainVars)
        for mode in range(KLModes):
            algo = ot.KrigingAlgorithm(inputSample, ksi_u[:,mode], \
                                       self.covarianceModel, self.basis)
            # set multistarts
            algo.setOptimizationAlgorithm\
                (ot.MultiStart(solver, lhsExperiment.generate()))
            # set optimization bounds
            algo.setOptimizationBounds(scaleOptimizationBounds)
            algo.setOptimizeParameters(True)
            algo.run()
            result = algo.getResult()
            GPResult_list.append( result )
            # save scale and amplitude parameters to be used for active learning
            # virtual training
            self.amplitudes.append(result.getCovarianceModel().getAmplitude())
            self.scales.append(result.getCovarianceModel().getScale())
        return GPResult_list
    
    def GPVirtualTraining(self, ksi_u_augmented, inputSample_augmented ):
        """
        For the active learning technique, the last trained metamodel is 
        used in an optimization loop to identify the new input variable that 
        reduces the most the uncertainity on the estimation of a quantile.
        
        The last trained metamodel (represented by GPResult_list_freezed) is 
        used to predict the random variables (ksi) for an input (u) given by 
        the optimizer. The new ksi and u are used to augment or expand the 
        training sets and create new GPS whose initialization is based on the
        parameteters of the last trained metamodel.
        
        The option:
        setOptimizeParameters(Bool)
        Controls wether to train the GPs or just use the same parameters of 
        the last trained metamodel

        Parameters
        ----------
        ksi_u_augmented : openturns.typ.Sample
            DESCRIPTION. Augmented ksi_u with the prediction of the value
            for ksi for a new input (u) given by the optimizer. The prediction
            is used using the last trained metamodel (not the last virtually trained)
        inputSample_augmented : openturns.Sample
            DESCRIPTION. augmented training input sample with the new input (u)
            given by the optimizer.

        Returns
        -------
         GPResult_list : List of openturns.KrigingResult objects
            DESCRIPTION. Contains the results of trained GPs

        """
        # exclude the "amplitude" from the optimization variables by estimating it 
        # analytically
        # get dimension of input sample (ksi_u). equivalent to number of KL modes
        KLModes = ksi_u_augmented.getDimension()
        # get the  size of the input sample (ksi_u). equivalent to the number
        # of uncertain variables
        numberUncertainVars = inputSample_augmented.getDimension()
        # define solver
        solver = ot.NLopt('LN_COBYLA')
        
        # initialize list to store the results of the GPs
        GPResult_list = []
        for mode in range(KLModes):
            # use the stored paramaters for the covariance model
            covarianceModel = self.covarianceModel
            covarianceModel.setScale(self.scales[mode])
            covarianceModel.setAmplitude( self.amplitudes[mode])
            algo = ot.KrigingAlgorithm(inputSample_augmented, 
                                       ksi_u_augmented[:,mode], 
                                       covarianceModel, self.basis)
            # set the optimizaton algorithm
            algo.setOptimizationAlgorithm(solver)
            # define wether to train or not the GPs
            # if setOptimizeParameters(False) the parameters of the previously
            # trained metamodel are used and no training is done
            algo.setOptimizeParameters(False)
            # run algorith mand store results
            algo.run()
            result = algo.getResult()
            GPResult_list.append( result )
            
        return GPResult_list
    
    def meanPrediction(self, u):
        """
        use the current GPResult_list to perform a mean GP prediction.
        If virtual training has been done, i.e. if the active learning 
        optimization loop has been exexuted, the mean prediction corresponds 
        to that of the GPS trained with the last augmented training set

        Parameters
        ----------
        u : openturns.Point
            realization of input variable whose mean output is to be predicted

        Returns
        -------
        metamodelEvaluation : openturns.Point
            DESCRIPTION. Predicted random variable of the KL decomposition

        """
        # create empty point to store results
        metamodelEvaluation = ot.Point()
        # loop through the GP meta models and evaluate them individually
        for mode in range(self.KLModes):
            metamodelEvaluation.add(self.GPResult_list[mode].getMetaModel()(u))
        return metamodelEvaluation
    
    def meanPredictionFreezed(self, u):
        """
        use the GPResult_list_freezed to perform a mean GP prediction.
        This prediction is not affected by any iterations of the virtual 
        training necessary for the active learning technique.

        Parameters
        ----------
        u : openturns.Point
            realization of input variable whose mean output is to be predicted

        Returns
        -------
        metamodelEvaluation : openturns.Point
            DESCRIPTION. Predicted random variable of the KL decomposition

        """
        # create empty point to store results
        metamodelEvaluation = ot.Point()
        # loop through meta models and evaluate them individually
        for mode in range(self.KLModes):
            metamodelEvaluation.add\
                (self.GPResult_list_freezed[mode].getMetaModel()(u))
        return metamodelEvaluation
        
    def randomRealizationPrediction(self, u):
        """
        Evaluate a GP random trajectory to predict the random variables of
        the KL expansion. a random seed can be set before calling this function
        to always obtain the same random prediction.

        Parameters
        ----------
        u : openturns.Point
            realization of input variable whose mean output is to be predicted

        Returns
        -------
        metamodelEvaluation : openturns.Point
            DESCRIPTION. Predicted random variable of the KL decomposition

        """
        # create empty point to store results
        metamodelEvaluation = ot.Point()
        # loop through meta models and evaluate them individually   
        for mode in range(self.KLModes):
            # build random vector
            rvector = ot.KrigingRandomVector(self.GPResult_list[mode], u) 
            # evaluate and store result 
            metamodelEvaluation.add( rvector.getRealization() )
        return metamodelEvaluation
    
    def sigmaPrediction(self,u, NSigma):
        """
        evaluate GP models at their (mean + NSigma * sigma) value

        Parameters
        ----------
        u : openturns.Point
            realization of input variable whose mean output is to be predicted
        NSigma : int
            DESCRIPTION. number of standard deviations. 

        Returns
        -------
        metamodelEvaluation : openturns.Point
            DESCRIPTION. Predicted random variable of the KL decomposition

        """
        # create empty point to store results
        metamodelEvaluation = ot.Point()
        # loop through meta models and evaluate them individually
        for mode in range(self.KLModes):
            metamodelEvaluation.add(
                self.GPResult_list[mode].getMetaModel()(u) + ot.Point
                ([NSigma * self.GPResult_list[mode].
                  getConditionalMarginalVariance(u)**0.5]))
        return metamodelEvaluation
    
    def variancePrediction(self,u):
        """
        calculate the variance of the GPs for the input u

        Parameters
        ----------
        u : openturns.Point
            realization of input variable whose mean output is to be predicted

        Returns
        -------
        varianceEvaluation : openturns.Point
            DESCRIPTION. Predicted variance of random variable of the KL decomposition

        """
        # create empty point to store results
        varianceEvaluation = ot.Point()
        # loop through meta models and evaluate them individually
        for mode in range(self.KLModes):
            varianceEvaluation.add(ot.Point([self.GPResult_list[mode].
                                             getConditionalMarginalVariance(u)]))
        return varianceEvaluation
      
    def updateVirtualInput(self, x, ksi_u, inputSample):
        """
        estimate ksi for a new input x proposed by the optimizer. augment
        the training sets with x and its metamodel response and then train
        the GPs on the augmented set.

        Parameters
        ----------
        x : openturns.Point
            DESCRIPTION. realization of input variable whose mean output is to 
            be predicted
        ksi_u : openturns.typ.Sample
            DESCRIPTION.set of random coefficients of the karhunen Loeve expansion
        associated to each KL mode and each item of the training set
        inputSample : openturns.Sample
            DESCRIPTION. the realizations of the input random variable vector
            used for training

        Returns
        -------
        None.

        """
        # estimate ksi(u) using the mean of the current metamodel and add it 
        # to the training samples
        ksi_u_augmented = np.concatenate( (np.array(ksi_u),
                                           np.array([self.meanPredictionFreezed(x)]) ) )
        
        inputSample_augmented = np.concatenate( (np.array(inputSample), 
                                                 np.array([x]) ) )

        # update the KG results list with the new trained GPs
        self.GPResult_list = self.GPVirtualTraining(ot.Sample(ksi_u_augmented), 
                                                    ot.Sample(inputSample_augmented) )
        
class FullMetamodel():
    
    """
    Metamodel composed by the KL expansion and the gaussian processes
    """
    
    def __init__(self, karhunenLoeveLiftingFunction, GPTrained, eigenFunctions, \
                 eigenValues ):
        """
        

        Parameters
        ----------
        karhunenLoeveLiftingFunction : openturns.algo.KarhunenLoeveLifting
            DESCRIPTION. Function to be used in tandem with a vector of random 
            coefficients to create a Karhune Loeve field
        GPTrained : instance of class GPMetamodel
            DESCRIPTION. contains methods to do predictions on the trained 
            GP metamodels
        eigenFunctions : openturns.func.ProcessSample
            DESCRIPTION. eigen functions of the resulting KL decomposition
        eigenValues : openturns.typ.Point
            DESCRIPTION. eigen values of the resulting KL decomposition

        Returns
        -------
        None.

        """
        # create python function for a random realization of  the GP metamodels
        rdmRealizationGPMetamodels= ot.PythonFunction(
            GPTrained.numberUncertainVars, 
            GPTrained.KLModes,  
            GPTrained.randomRealizationPrediction)
        
        # create python function for the mean realization of  the GP metamodels
        meanRealizationGPMetamodels= ot.PythonFunction( 
            GPTrained.numberUncertainVars, 
            GPTrained.KLModes,  
            GPTrained.meanPrediction)
        
        # store GP results list and KL lifting function and eig func and vals
        self.GPTrained = GPTrained
        self.karhunenLoeveLiftingFunction = karhunenLoeveLiftingFunction
        self.eigenFunctions = eigenFunctions
        self.eigenValues = eigenValues
        
        # create full metamodel combining the KL lifting function and a random 
        # realization ot the GP metamodels 
        self.evaluateRandomRealization = ot.PointToFieldConnection(
            karhunenLoeveLiftingFunction, 
            rdmRealizationGPMetamodels)
        
        # create full metamodel combining the KL lifting function and the mean 
        # realization ot the GP metamodels 
        self.evaluateMeanRealization = ot.PointToFieldConnection(
            karhunenLoeveLiftingFunction, 
            meanRealizationGPMetamodels)
        
    def evaluateNSigmaRealization(self,u, NSigma):
        """
        calculate the confidence interval at NSigma level by propgating
        the GP error model through the KL expansion by using the properties
        of Gaussian distributions

        Parameters
        ----------
        u : openturns.Point
            DESCRIPTION. realization of input variable whose output confidence interval is 
            to be predicted
        NSigma : int
            DESCRIPTION. number of standard deviations.

        Returns
        -------
        bounds: List
            List containing the Fields corresponding to the lower and upper
            bounds of the confidence intercal at NSigma level. [lb,ub]

        """
        
        numberOfModes = self.eigenFunctions.getSize()
        # get variance from GPs
        GPVariance = self.GPTrained.variancePrediction(u)
        # evaluate variance model for u
        variance = np.zeros([self.eigenFunctions.getTimeGrid().
                             getN(),self.eigenFunctions.getDimension()])
        # compute total output variance by iterating though the KL modes
        for k in range(numberOfModes):
            variance += self.eigenValues[k] * GPVariance[k] *\
                np.array(self.eigenFunctions[k][:,0] )**2
        # return evaluation mu + NSigma * sigma
        return [self.evaluateMeanRealization(u) + NSigma * np.sqrt(variance),
                self.evaluateMeanRealization(u) - NSigma * np.sqrt(variance)]
        
    
    def testMeanRealization(self, quantileEstimationInputSample, 
                            outputTrainingSample, mesh,iterationNumber, 
                            resultsPath):
        """
        This function plots the original training samples and and the
        the quantile estimation output samples using the mean realization
        of the full surrogate model

        Parameters
        ----------
        quantileEstimationInputSample : openturns.Sample
            DESCRIPTION. Sample containing the input sample to estimate the
            quantile
        outputTrainingSample : openturns.ProcessSample
            DESCRIPTION. OpenTurns object representing the stochastic process
        mesh : openturns.geom.Mesh
            DESCRIPTION. mesh of  the output stochastic process
        iterationNumber : Int
            DESCRIPTION. iteration number
        resultsPath : Str
            DESCRIPTION. string indicating the folder to save the results

        Returns
        -------
        None.

        """
        # use metamodel to estimate output
        metamodelOutput = \
            self.evaluateMeanRealization(quantileEstimationInputSample)
        # extract vertices and convert smaples to arrays
        vertices = np.array(mesh.getVertices())[:,0]
        trainingSamplesOutput = np.array(outputTrainingSample)
        metamodelOutput = np.array(metamodelOutput)
        # plot 
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=[8,5])
        # plot metamodel samples
        for i in range(quantileEstimationInputSample.getSize()):
            plt.plot(vertices, metamodelOutput[i,:,0],'b-', linewidth=0.2)
        # plot training samples
        for i in range(outputTrainingSample.getSize()):
            plt.plot(vertices, trainingSamplesOutput[i,:,0],'r-', linewidth=2)
        # plt.ylim([-50,550])
        savePath = resultsPath+'/%i_training'%iterationNumber
        plt.ioff()
        plt.savefig(savePath, dpi=200, bbox_inches='tight')
        
    def testSigmaModel(self, quantileEstimationInputSample, mesh, resultsPath):
        """
        
        This function produces various plots showing the mean prediction for
        one sample, the condidence interval(sigma = 1,2,3) based on the 
        propagation of the GP error through the KL decomposition and compares 
        it with brute force sampling of the GPs.
        
        Parameters
        ----------
        quantileEstimationInputSample : openturns.Sample
            DESCRIPTION. Sample containing the input sample to estimate the
            quantile
        mesh : openturns.geom.Mesh
            DESCRIPTION. mesh of  the output stochastic process
        resultsPath : Str
            DESCRIPTION. string indicating the folder to save the results

        Returns
        -------
        None.

        """
        # select one realization of the input sample
        u = quantileEstimationInputSample[0]
        # use metamodel to estimate output
        metamodelMeanOutput = self.evaluateMeanRealization(u)
        # extract vertices and convert smaples to arrays
        vertices = np.array(mesh.getVertices())[:,0] 
        # plot mean
        plt.rcParams.update({'font.size': 16})
        # ====== Plot predicted sample ========================================
        plt.figure(figsize=[8,5])
        plt.plot(vertices, np.array(metamodelMeanOutput)[:,0], 'k-', linewidth= 1,
                 label = 'metamodel mean prediction')
        # plt.title('metamodel prediction for 1 input sample')
        plt.legend(loc = 'lower right', fontsize = 8)
        plt.savefig(resultsPath+'/0_sigmaModel.pdf', dpi=400, bbox_inches='tight')
        # =====================================================================
        # ======== Plot confidence interval of the predicted sample ===========
        # plot CI
        sigma_list = [1,2,3]
        for sigma in sigma_list:
            ub, lb = self.evaluateNSigmaRealization(u, sigma)
            plt.fill_between(vertices,
                          np.array(ub)[:,0],  
                          np.array(lb)[:,0],
                          facecolor= 'gray', alpha= 1/sigma, 
                          label = r'%i$\sigma$ bounds using KL + GP error model'%sigma)
        plt.legend(loc = 'lower right', fontsize = 8)
        # plt.ylim([-10,2])
        # plt.xlabel(r'% of flight')
        # plt.ylabel(r'centered height (km)')
        plt.savefig(resultsPath+'/1_sigmaModel.pdf', dpi=400, bbox_inches='tight')
        # =====================================================================
        # generete multiple predictions of the same sample based on GP trajectories
        # plot the confidence interval of the resulting Gaussian process
        # get mesh and number of vertices
        verticesNumber = mesh.getVerticesNumber()
        # initialize quantile process sample
        rdmGP = ot.ProcessSample(mesh,0,verticesNumber)
        realizations = 1000
        for i in range(realizations):
            # generate new random seed
            ot.RandomGenerator.SetSeed(i)
            # store evaluation
            aux = self.evaluateRandomRealization(u)
            # add evaluation to process sample
            rdmGP.add( aux )
            
        # plot quantiles of thousands of samples
        for sigma in sigma_list:
            plt.plot(vertices, np.array(rdmGP.computeQuantilePerComponent\
                                        (st.norm.cdf(sigma)))[:,0],'r-.', 
                     linewidth=1)
            plt.plot(vertices, np.array(rdmGP.computeQuantilePerComponent\
                                        (st.norm.cdf(-sigma)))[:,0],'r-.', 
                     linewidth=1)
        # add legend for multiple realizations
        handles, labels = plt.gca().get_legend_handles_labels()
        from matplotlib.lines import Line2D
        line  = Line2D([0], [0], label = r'%i GP realizations prediction' 
                       % realizations ,color='r', linestyle='-', linewidth = 0.5)
        line2 = Line2D([0], [0], label = 
                       r'$\sigma = [1,2,3]$ bounds based on %i GP realizations ' 
                       % realizations ,color='r', linestyle='-.', linewidth = 1)
        handles.extend([line, line2])
        plt.legend(handles = handles, loc = 'lower right', fontsize = 8)
        plt.savefig(resultsPath+'/3_sigmaModel.pdf', dpi=400, bbox_inches='tight')
        # ======================================================================
        # plot the thousands of trajectories
        rdmGPArray = np.array(rdmGP)
        for i in range(realizations):
            # plot the thing
            plt.plot(vertices,  rdmGPArray[i,:,0],
                      'r-', linewidth=0.06)
        plt.savefig(resultsPath+'/2_sigmaModel.pdf', dpi=400, bbox_inches='tight')
        
    def testSigmaModelMulti(self, quantileEstimationInputSample, mesh, \
                            resultsPath, index):
        """
        Similarly to the testSigmaModel method. This function plots the 2sigma
        confidence interval of a set of sample predictions using 
        the propagation of the GP variance through the KL expansion.

        Parameters
        ----------
        quantileEstimationInputSample : openturns.Sample
            DESCRIPTION. Sample containing the input sample to estimate the
            quantile
        mesh : openturns.geom.Mesh
            DESCRIPTION. mesh of  the output stochastic process
        resultsPath : Str
            DESCRIPTION. string indicating the folder to save the results
        index : Int
            DESCRIPTION. index indicating the iteration number

        Returns
        -------
        None.

        """
        # plot mean
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=[8,5])
        # loop through a set of samples
        for i in range(0,3):
            u = quantileEstimationInputSample[i]
            # use metamodel to estimate output
            metamodelMeanOutput = self.evaluateMeanRealization(u)
            # extract vertices and convert smaples to arrays
            vertices = np.array(mesh.getVertices())[:,0] 
            plt.plot(vertices, np.array(metamodelMeanOutput)[:,0], 'k', \
                     linewidth= 1)
            # plt.title('''
            #           metamodel prediction for multiple input samples.  
            #           CI bounds with $\sigma =2$
            #           ''')
            # plot CI
            for sigma in range(2,3):
                ub, lb = self.evaluateNSigmaRealization(u, sigma)
                plt.fill_between(vertices,
                              np.array(ub)[:,0],  
                              np.array(lb)[:,0], alpha= 1/sigma, 
                              label = 
                              r'%i$\sigma$ bounds using KL + GP error model'%sigma)
        # plt.ylim([-50,550])
        # plt.legend(loc = 'lower right', fontsize = 8) 
        # plt.legend()
        # plt.xlabel(r'% of flight')
        # plt.ylabel(r'centered height (km)')
        # plt.ylim([-5,12])
        plt.savefig(resultsPath+'/%i_sigmaModelMulti.pdf'%index, dpi=400, \
                    bbox_inches='tight')
        
    def testSigmaModelMultiExtended(self, quantileEstimationInputSample, mesh, 
                                    resultsPath):
        """
        
        This function combines the methods described in the methods
        "testSigmaModel" and "testSigmaMulti"
        
        Parameters
        ----------
        quantileEstimationInputSample : openturns.Sample
            DESCRIPTION. Sample containing the input sample to estimate the
            quantile
        mesh : openturns.geom.Mesh
            DESCRIPTION. mesh of  the output stochastic process
        resultsPath : Str
            DESCRIPTION. string indicating the folder to save the results

        Returns
        -------
        None.

        """
        # plot mean
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=[8,5])
        
        for i in range(5,9):
            u = quantileEstimationInputSample[i]
            # use metamodel to estimate output
            metamodelMeanOutput = self.evaluateMeanRealization(u)
            # extract vertices and convert smaples to arrays
            vertices = np.array(mesh.getVertices())[:,0]
            plt.plot(vertices, np.array(metamodelMeanOutput)[:,0], 'k', linewidth= 1)
            plt.title(''''
                      metamodel prediction for multiple input samples. 
                      CI bounds with $\sigma =2$
                      ''')
            # plot CI
            for sigma in range(2,4):
                ub, lb = self.evaluateNSigmaRealization(u, sigma)
                plt.fill_between(vertices,
                              np.array(ub)[:,0],  
                              np.array(lb)[:,0], alpha= 1/sigma,
                              facecolor = 'gray',
                              label = 
                              r'efficient prediction of %i$\sigma$ bounds'%sigma)
 
        realizations = 1000
        for i in range(realizations):
            # generate new random seed
            ot.RandomGenerator.SetSeed(i)
            for i in range(5,9):
                u = quantileEstimationInputSample[i]
                # store evaluation
                aux = self.evaluateRandomRealization(u)
                plt.plot(vertices,  np.array(aux), 'r-', linewidth = 0.06)
        # plt.ylim([-50,550])
        # plt.legend(loc = 'lower right', fontsize = 8) 
        plt.savefig(resultsPath+'/0_sigmaModelMultiExtended.png', dpi=400, \
                    bbox_inches='tight')
            
            