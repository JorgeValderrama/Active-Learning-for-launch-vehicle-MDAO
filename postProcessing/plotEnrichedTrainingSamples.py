# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 13:40:12 2021
This script plots the training samples, highlighting those that were added
with the active learning technique.

@author: jorge
"""
from pathlib import Path
p = Path(__file__).parents[1]
import sys
sys.path.append( str(p) )
import openturns as ot
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pickle

# define the original number of samples that were used to train the model
# and the number of samples that were added thorugh the enrichment process
originalSamples = 4
addedSamples    = 5
# add samples
totalSamples    = originalSamples + addedSamples

# select the list of labels 
# labels for dymos
u_list = [r'$Isp_1$ (s)', r'$Isp_2$ (s)', r'$m_1$ (kg)', r'$m_2$ (kg)',
              r'$q_1$ (kg/s)', r'$q_2$ (kg/s)', r'$C_D$ ()']
# labels for ODM
# u_list = [r'$E_0 $ (MPa)', r'$ycs$ (MPa)', r'$y0s$ (MPa)', r'$dc$ ()']

# import active learning problem and denormalize if needed
resultsPath= str(p) + '/results/'
pb = pickle.load( open( resultsPath + 'activeLearningProblem.p', "rb" ) )
pb.deNormalizeInputSamples()
enrichedSamples = pb.inputTrainingSample
numberOfDimensions = enrichedSamples.getDimension()
# generate iterator to get all possible combinations of two variables
# iterator = combinations(range(numberOfDimensions),2)
iterator = combinations(range(numberOfDimensions-1, -1, -1),2)

#%% plotting for inputs
# create new matplotlib figure
fig = plt.figure(figsize=(26, 26),  tight_layout= False, constrained_layout= False)
plt.rcParams.update({'font.size': 16})
# iterate through all possible combinations of two variables
for i, j in iterator:
    graph = ot.Graph('','','',True)
    graph.setLegendPosition('')
    # plot the original and the added samples
    cloud1 = ot.Cloud(enrichedSamples[:originalSamples,[i,j]])
    cloud2 = ot.Cloud(enrichedSamples[originalSamples:totalSamples, [i,j]], 'red', 'fsquare')
    graph.add(cloud1)
    graph.add(cloud2)
    # add the graph to a grid plot using matplotlib
    # index = numberOfDimensions * (numberOfDimensions - j - 1) + i + 1
    index = numberOfDimensions * (numberOfDimensions - j - 1) + (numberOfDimensions-i)
    ax = fig.add_subplot(numberOfDimensions, numberOfDimensions, index)
    _ = ot.viewer.View(graph, figure=fig, axes=[ax])
    # print axis labels
    if j == 0:
        plt.xlabel(u_list[i],  fontsize=24)
    if i == numberOfDimensions-1:
        plt.ylabel(u_list[j],  fontsize=24)
    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 20)
_ = fig.suptitle("")

#add legend
ax = fig.add_subplot(numberOfDimensions, numberOfDimensions, numberOfDimensions*2-1)
ax.axis("off")
# where some data has already been plotted to ax
# handles, labels = ax.get_legend_handles_labels()
# manually define a new patch 
red_square = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                          markersize=10, label='added input samples')
blue_cross = mlines.Line2D([], [], color='blue', marker='+', linestyle='None',
                          markersize=10, label='original input samples')
ax.legend(handles = [red_square, blue_cross], loc='best')
fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
savePath = str(p)+'/results/enrichedInput'
plt.savefig(savePath + '.pdf', format='pdf', bbox_inches='tight')

#%%
# plotting for outputs
# de-normalize if needed
pb.deNormalizeOutputSample()
enrichedOutput = pb.outputTrainingSample
vertices = np.array(pb.mesh.getVertices())
plt.figure()
for i in range(0,originalSamples):
    plt.plot(vertices, enrichedOutput[i], 'k')
plt.plot(vertices, enrichedOutput[i], 'k', label = 'original samples')
for i in range(originalSamples, totalSamples):
    plt.plot(vertices, enrichedOutput[i], 'r')
plt.plot(vertices, enrichedOutput[i], 'r', label = 'added samples')
plt.legend(loc = 'best')
savePath = str(p)+'/results/enrichedOutput'
plt.savefig(savePath + '.pdf', format='pdf', bbox_inches='tight')