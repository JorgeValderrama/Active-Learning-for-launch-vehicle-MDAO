# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 13:44:30 2021
This script plot the Confidence interval area evolution as a function
of the number of iterations and the confidence interval graphs.
@author: jorge
"""
from pathlib import Path
p = Path(__file__).parents[1]
import sys
sys.path.append( str(p) )
import pickle
import matplotlib.pyplot as plt
import numpy as np

# import list of dictionaries containing quantile information
resultsPath= str(p) + '/results/'
CIAreaCurves_list = pickle.load( open( resultsPath + 'CIAreaCurves.p', "rb" ) )
pb = pickle.load( open( resultsPath + 'activeLearningProblem.p', "rb" ) )

#%% 
# plot CI Area evolution
# numberOfIterations = len(CIAreaCurves_list)
# extract area
area_list =[]
for my_dict in CIAreaCurves_list:
    area_list.append(my_dict['area'])
plt.figure()
plt.plot(area_list)
plt.xlabel('iteration number')
plt.ylabel('CI Area')
savePath = str(p)+'/results/CIAreaEvolution'
plt.savefig(savePath + '.pdf', format='pdf', bbox_inches='tight')

#%%
# plot the confidence interval for a given iteration
plt.rcParams.update({'font.size': 16})
def plotCIAreaBounds(results_dict, pb, index):
    vertices = np.array(pb.mesh.getVertices())
    plt.figure(figsize=[8,5])
    plt.fill_between(vertices[:,0],
                  np.array(results_dict['Q_lb'])[:,0],  
                  np.array(results_dict['Q_ub'])[:,0],
                  facecolor='red', alpha=0.5)
        # plot the quantile estimated with the mean GP evaluation
    plt.plot(vertices, np.array(results_dict['QMean'])[:,0],'k-')
    savePath = str(p)+'/results/CIAreaBounds' + str(index)
    plt.savefig(savePath + '.pdf', format='pdf', bbox_inches='tight')
    
# iterate through the available iterations, plotting the Confidence intervals 
for index, results_dict in enumerate(CIAreaCurves_list):
    plotCIAreaBounds(results_dict, pb, index)

