# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 09:13:03 2021

@author: jorge
"""

import openturns as ot
import numpy as np
import json
import codecs

import openturns.viewer as viewer
from matplotlib import pylab as plt

number_of_realizations = 2

t_min    = 0.0  
t_max    = 430 - 2
gridsize = 215

mesh = ot.IntervalMesher([gridsize-1]).build(ot.Interval(t_min, t_max))

vertices = mesh.getVertices()

deformations_vertices = np.array(vertices)
deformations_vertices = np.reshape(np.array(deformations_vertices),
                                   len(deformations_vertices,))

# # read data sets
obj_text = codecs.open('results/intepolated_results/intepolated_results.txt', 'r', encoding='utf-8').read()

# define process sample
trainingOutputSample = ot.ProcessSample(mesh,0,gridsize)

# dfine function to convert elemtns of list into lists
def extractElements(lst):
    return [[el] for el in lst]

for Id in range(number_of_realizations):
    timeseries = json.loads(obj_text)[str(Id) + '_interp_results']['interpolated_resultsSim']['m'] # 
    trainingOutputSample.add( ot.Sample( extractElements(timeseries) ) )

graph = trainingOutputSample.drawMarginal()
graph.setTitle('%i trajectories'%number_of_realizations)
graph.setXTitle('time (s)')
graph.setYTitle('m (kg)')

view = viewer.View(graph)
#plt.show()
plt.savefig('100_traj_d.png', dpi = 600)