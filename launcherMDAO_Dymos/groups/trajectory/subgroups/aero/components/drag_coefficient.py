# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:08:43 2020

This class calculates the drag coefficient as a function of Mach number.
First, an interpolator is fitted to tabulated data. This is done only 1 time per optimization procedure.

The drag coefficient table corresponds to Ariane 5 data according to 
Figure 3.11 of Pagano - Global Launcher Trajectory Optimization for Lunar Base Settlement.
This has to be replaced by better quality data in the future.
@author: jorge
"""

from scipy.interpolate import PchipInterpolator
import numpy as np
import openmdao.api as om


int_data = np.array([
                [0   , 0.42],
                [0.3 , 0.51],
                [0.7 , 0.63],
                [1.1 , 1.30],
                [1.3 , 1.65],
                [1.5 , 1.38],
                [1.8 , 1.22],
                [2.1 , 1.15],
                [2.7 , 0.95],
                [3.0 , 0.90],
                [4   , 0.70],
                [5   , 0.60],
                [7   , 0.58],
                [8.5 , 0.58],
                [10  , 0.59], 
                [11  , 0.59],
                [12  , 0.59],
                [13  , 0.59],
                [14  , 0.59],
                [15  , 0.59],
                ])

# pchip
CdCurve = PchipInterpolator((int_data[:,0]),(int_data[:,1]))
CdCurve_p = CdCurve.derivative(1)

class Drag_coefficient(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')
        
    def setup(self):
        
        nn = self.options['num_nodes']
        
        self.add_input('Mach',
                       val=np.zeros(nn),
                       desc='Mach number',
                       units=None)
        
        self.add_output('Cd',
                        val=np.zeros(nn),
                        desc='Drag coefficient',
                        units=None)
        
        # Setup partials 
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of = 'Cd', wrt = 'Mach', rows=ar, cols=ar)
        
    def compute(self, inputs, outputs):
        
        outputs['Cd'] = CdCurve(inputs['Mach'], extrapolate=True)
        
    def compute_partials(self, inputs, jacobian):
        
        jacobian['Cd','Mach'] = CdCurve_p(inputs['Mach'], extrapolate=True)
        