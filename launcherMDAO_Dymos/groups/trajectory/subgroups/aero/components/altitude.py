# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:44:01 2020

This class converts "radius" in to altitude. This is necessary for the atmospheric models.

@author: jorge
"""

import openmdao.api as om
import numpy as np

class Altitude(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('r0', types=float,
                             desc = 'Earths radius (m)')
        
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('r', 
                       val=np.zeros(nn),
                       desc='radius',
                       units='m')
        
        self.add_output('h', 
                        val=np.zeros(nn), 
                        desc='altitude',
                        units='m')
        
        ar = np.arange(nn)
        
        # Define the partial with its constant value of 1. This avoids the need to create the method "compute_partials"
        self.declare_partials(of = 'h', wrt = 'r', rows=ar, cols=ar, val = 1.0)
        
    def compute(self, inputs, outputs):
        
        r       = inputs['r']
        r0 = self.options['r0']
        
        outputs['h'] = r - r0
        