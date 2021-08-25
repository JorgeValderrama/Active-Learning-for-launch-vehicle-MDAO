# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 08:29:36 2021

Two functions to calculate the confidence interval uncertainty are defined in this
script. Uncertainty is measured by the area of the confidence interval but
a different metric could be used.

@author: jorge
"""
import openturns as ot
import matplotlib
# matplotlib.use('agg', force=True)
import matplotlib.pyplot as plt
# print ("Switched to:",matplotlib.get_backend())
import numpy as np
import scipy.integrate as inte
import scipy.stats as st
import pickle


def CIAreaA(fullMetamodel, areaArgs):
    """
    This function calculates the confidence interval at a level "CIEta_CIAreaA"
    of the estimated quantile of level "etaA" based on the generation
    of random GP trajectories. It has a double loop of full surrogate model
    evaluations that makes it computationally expensive.
    
    When test ==True, the function plots the confidence interval of the estimated
    qunatile and saves the curves using pickle.

    Parameters
    ----------
    fullMetamodel : metamodels.FullMetamodel
        DESCRIPTION. instance of FullMetamodel class containing the full
        surrogate model based on the KL expansion and GPs for the prediction
        of the uncertain part of the KL decomposition.
    areaArgs : List
        DESCRIPTION. List containing arguments to be used for the CI Area methods

    Returns
    -------
    CIArea : Float
        DESCRIPTION. Value of the confidence interval area

    """
    
    # unpack arguments
    mesh, quantileEstimationInputSample, CIEta_CIAreaA, __, etaA,__, \
        numberOfAnalyses, iterationNumber, test , scalingArgs, resultsPath = areaArgs
    
    # get mesh and number of vertices
    verticesNumber = mesh.getVerticesNumber()
    # initialize quantile process sample
    Q_sample = ot.ProcessSample(mesh,0,verticesNumber)
    # loop through the number of analyses generating a new trajectory of the
    # GP metamodels (controlled with a random seed) and evaluate the whole
    # input sample. Calculate the quantile at a level etaA of the output and 
    # store it.
    for i in range(numberOfAnalyses):
        # define a random seed that changes for every analysis
        ot.RandomGenerator.SetSeed(i)
        # evaluate random realization of full metamodel
        outputMetamodel = fullMetamodel.evaluateRandomRealization(
            quantileEstimationInputSample)
        # compute and store quantile at level etaA
        Q_sample.add(outputMetamodel.computeQuantilePerComponent(etaA))
    # estimate the confidence interval at a level CIEta_CIAreaA and the mean 
    # of the estimated quantile
    lowerLimit = (1-CIEta_CIAreaA)/2
    QQ_lb   = Q_sample.computeQuantilePerComponent(lowerLimit)
    QQ_ub   = Q_sample.computeQuantilePerComponent(lowerLimit + CIEta_CIAreaA)
    QQ_mean = Q_sample.computeMean()
    # extract the vertices of the mesh as numpy array
    vertices = np.array(mesh.getVertices())
    # calculate area of the confidence interval using a numerical integrator
    ub_area = inte.simps(np.array(QQ_ub)[:,0], vertices[:,0]) 
    lb_area = inte.simps(np.array(QQ_lb)[:,0], vertices[:,0])
    CIArea  = ub_area - lb_area 
    
    if test ==True:
        # obtain the estimated quantile with the mean realization of the 
        # metamodel
        outputMetamodelMean = fullMetamodel.evaluateMeanRealization(
            quantileEstimationInputSample)
        QMean = outputMetamodelMean.computeQuantilePerComponent(etaA)
        # plot the confidence interval
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=[8,5])
        plt.fill_between(vertices[:,0],
                      np.array(QQ_lb)[:,0],  
                      np.array(QQ_ub)[:,0], 
                      facecolor='red', alpha=0.5)
        # plot the quantila of the validation data
        # plot the mean of the estimated quantile
        plt.plot(vertices, np.array(QQ_mean)[:,0], 'k')
        # # plot the quantile estimated with the mean GP evaluation
        # plt.plot(vertices, np.array(QMean)[:,0],'b')
        plt.title('Scaled CI area A: %.5f'%CIArea)
        # plt.ylim([-50,550])
        savePath = resultsPath+'/%i_CIArea'%iterationNumber
        # plt.savefig(savePath, dpi=400, bbox_inches='tight')
        # save curves 
        saveCIAreaCurves(QQ_ub, QQ_lb, QQ_mean, CIArea, scalingArgs, resultsPath,
                         iterationNumber)
        # plot trajectories
        # ======================================================================
        # plt.rcParams.update({'font.size': 16})
        # plt.figure()
        # aux = quantileEstimationInputSample.getSize()
        # for i in range(aux):
        #     plt.plot(vertices, outputMetamodel[i], 'r')
        # plt.plot(vertices, outputMetamodel[-1], 'r', label = r'%i output predictions'%aux)
        # plt.plot(vertices, Q_sample[0], '--k', label = r'$q_{%1.2f}$'%eta)
        # plt.ylabel('centered height (km)')
        # plt.ylim([-12,12])
        # plt.xlabel(r'% of flight')
        # plt.legend(loc = 'lower right')
        # plt.savefig(resultsPath + 'oneQuantile.pdf', format = 'pdf', bbox_inches='tight')
        # ======================================================================
        # plot quantiles
        # ======================================================================
        # plt.rcParams.update({'font.size': 16})
        # plt.figure()
        # aux = numberOfAnalyses
        # for i in range(aux):
        #     plt.plot(vertices, Q_sample[-i], '--k')
        # plt.plot(vertices, Q_sample[-1], '--k', label = r'%i estimations of $q_{%1.2f}$'%(numberOfAnalyses, eta))
        # plt.plot(vertices, QQ_lb, '--r')
        # plt.plot(vertices, QQ_ub, '--r', 
        #          label = r'$q_{%1.3f}$ and $q_{%1.3f}$'%(lowerLimit, lowerLimit + CIEta_CIAreaA))
        # plt.ylabel('centered height (km)')
        # plt.ylim([-12,12])
        # plt.xlabel(r'% of flight')
        # plt.legend(loc = 'lower right')
        # plt.savefig(resultsPath + 'variousQuantile.pdf', format = 'pdf', bbox_inches='tight')
        # ======================================================================
        # plot confidence interval
        # ======================================================================
        # plt.rcParams.update({'font.size': 16})
        # plt.figure()
        # plt.fill_between(vertices[:,0],
        #               np.array(QQ_lb)[:,0],  
        #               np.array(QQ_ub)[:,0], 
        #               facecolor='red', alpha=0.5,
        #               label = '%1.2f confidence interval'%CIEta_CIAreaA)
        # # plot the quantila of the validation data
        # # plot the mean of the estimated quantile
        # plt.plot(vertices, np.array(QQ_mean)[:,0], 'k', 
        #          label = r'mean estimation of $q_{%1.2f}$'%eta)
        # plt.ylabel('centered height (km)')
        # plt.ylim([-12,12])
        # plt.xlabel(r'% of flight')
        # plt.legend(loc = 'lower right')
        # plt.savefig(resultsPath + 'confidenceInterval.pdf', format = 'pdf', bbox_inches='tight')
        
    return CIArea

def CIAreaB(fullMetamodel, areaArgs):
    """
    This function calculates the confidence interval area at a unknown level of 
    confidence of the quantile at level "etaB". It uses the error model based
    on the propagation of the variance of the GPs through the KL expansion to
    genereate two sets of predictions. The sets are obtained from the 
    mean prediction +- NSigma_CIAreaB * Sigma and the quantile at level "etaB"
    of each set is obtained. The area between both quantiles is returned.
    This method is computationally cheaper than CIArea A.
    
    When test ==True, the function plots the confidence interval of the estimated
    qunatile and saves the curves using pickle.

    Parameters
    ----------
    fullMetamodel : metamodels.FullMetamodel
        DESCRIPTION. instance of FullMetamodel class containing the full
        surrogate model based on the KL expansion and GPs for the prediction
        of the uncertain part of the KL decomposition.
    areaArgs : List
        DESCRIPTION. List containing arguments to be used for the CI Area methods

    Returns
    -------
    CIArea : Float
        DESCRIPTION. Value of the confidence interval area

    """
    
    # unpack arguments
    mesh, quantileEstimationInputSample, __, NSigma_CIAreaB, __, etaB, __, \
        iterationNumber, test, scalingArgs, resultsPath = areaArgs
    
    # get mesh and number of vertices
    verticesNumber = mesh.getVerticesNumber()
    # initialize quantile process sample
    aux_sample_ub = ot.ProcessSample(mesh,0,verticesNumber)
    aux_sample_lb = ot.ProcessSample(mesh,0,verticesNumber)
    # loop through validation input smaples
    for u in quantileEstimationInputSample:
        ub, lb = fullMetamodel.evaluateNSigmaRealization(u, NSigma_CIAreaB)
        aux_sample_ub.add(ub)
        aux_sample_lb.add(lb)
    # find quantiles
    Q_ub = aux_sample_ub.computeQuantilePerComponent(etaB)
    Q_lb = aux_sample_lb.computeQuantilePerComponent(etaB)
    
    # extract the vertices of the mesh as numpy array
    vertices = np.array(mesh.getVertices())
    # calculate area of the confidence interval
    ub_area = inte.simps(np.array(Q_ub)[:,0], vertices[:,0]) 
    lb_area = inte.simps(np.array(Q_lb)[:,0], vertices[:,0])
    boundsArea = ub_area-lb_area
    
    # test - do plot
    if test ==  True:
        
        # metamodel mean realization quantile
        outputMetamodelMean = fullMetamodel.evaluateMeanRealization(
            quantileEstimationInputSample)
        QMean = outputMetamodelMean.computeQuantilePerComponent(etaB)
        # plot the confidence interval
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=[8,5])
        plt.fill_between(vertices[:,0],
                      np.array(Q_lb)[:,0],  
                      np.array(Q_ub)[:,0], 
                      facecolor='red', alpha=0.5)
        # plot the quantile estimated with the mean GP evaluation
        plt.plot(vertices, np.array(QMean)[:,0],'k-')
        plt.title('Scaled CI area B: %.5f'%boundsArea)
        # plt.ylim([-50,550])
        savePath = resultsPath+'/%i_boundsArea'%iterationNumber
        # plt.savefig(savePath, dpi=400, bbox_inches='tight')
        # save curves
        saveCIAreaCurves(Q_ub, Q_lb, QMean, boundsArea, scalingArgs, resultsPath,
                         iterationNumber)
        # # ======================================================================
        # # plot confidecne interval area
        # plt.rcParams.update({'font.size': 16})
        # plt.figure()
        # plt.fill_between(vertices[:,0],
        #               np.array(Q_lb)[:,0],  
        #               np.array(Q_ub)[:,0], 
        #               facecolor='red', alpha=0.5,
        #               label = 'confidence interval using $\gamma =%i$'%NSigma_CIAreaB)
        # plt.plot(vertices, np.array(QMean)[:,0],'k-', 
        #          label = r'estimation of $q_{%1.2f}$ using $\hat{\bar{\mathbf{X}}}^*$'%eta)
        # plt.ylabel('centered height (km)')
        # plt.ylim([-12,12])
        # plt.xlabel(r'% of flight')
        # plt.legend(loc = 'lower right', fontsize = 12)
        # plt.savefig(resultsPath + 'sigmaConfidenceInterval.pdf', format = 'pdf', bbox_inches='tight')
        # # ======================================================================
        # # plot trajectories
        # # ======================================================================
        # plt.rcParams.update({'font.size': 16})
        # plt.figure()
        # aux = quantileEstimationInputSample.getSize()
        # for i in range(aux):
        #     plt.plot(vertices, aux_sample_ub[i], 'k' )
        # for i in range(aux):
        #     plt.plot(vertices, aux_sample_lb[i], '--b' )
        # plt.plot(vertices, aux_sample_ub[-1], 'k',  label = r'$\hat{\bar{\mathbf{X}}}^* + %i \Sigma$'%NSigma_CIAreaB)
        # plt.plot(vertices, aux_sample_lb[-1], '--b', label = r'$\hat{\bar{\mathbf{X}}}^* - %i \Sigma$'%NSigma_CIAreaB)
        # plt.plot(vertices, Q_ub, '--r')
        # plt.plot(vertices, Q_lb, '--r', label = r'$q_{%1.2f}$'%eta)
        # # plt.plot(vertices, Q_sample[0], '--k', label = r'$q_{%1.2f}$'%eta)
        # plt.ylabel('centered height (km)')
        # plt.ylim([-12,12])
        # plt.xlabel(r'% of flight')
        # plt.legend(loc = 'lower right', fontsize = 12)
        # plt.savefig(resultsPath + 'sigmaRealizations.pdf', format = 'pdf', bbox_inches='tight')
       
    return boundsArea    
             
def saveCIAreaCurves(Q_ub, Q_lb, QMean, boundsArea, scalingArgs, resultsPath,
                     iterationNumber):
    """
    

    Parameters
    ----------
    Q_ub : openturns.func.Field
        DESCRIPTION. upper bound of the confidence interval
    Q_lb : openturns.func.Field
        DESCRIPTION. lower bound of the confidence interval
    QMean : openturns.func.Field
        DESCRIPTION. mean estimation of the quantile
    boundsArea : float
        DESCRIPTION. confidence interval area
    scalingArgs : List
        DESCRIPTION. List of arguments used to escale curves and area
    resultsPath : Str
        DESCRIPTION. path to the folder where results should be stored
    iterationNumber : Int
        DESCRIPTION. Iteration number

    Returns
    -------
    None.

    """
    
    # unpack scale args
    outputMean, outputScale = scalingArgs
    # scale back results and pack them in a dictionary
    results = {'Q_ub':( np.array(Q_ub) /  outputScale + outputMean).tolist(),
               'Q_lb':( np.array(Q_lb) /  outputScale + outputMean).tolist(),
               'QMean':( np.array(QMean) /  outputScale + outputMean).tolist(),
               'area':( boundsArea / outputScale) }
    
    # define file name
    fileName = resultsPath + "CIAreaCurves.p"
    # In the firstiteration create an empty list
    # append results dictionary
    if iterationNumber == 0:
        CIAreaCurves_list = []
    else:
        CIAreaCurves_list = pickle.load( open( fileName, "rb" ) )
    CIAreaCurves_list.append(results)
    # save pickle file
    pickle.dump(CIAreaCurves_list, open(fileName, "wb"))