# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:14:40 2020

This class finds the inertial speed from the speed in the Earth-Rotating Earth centered frame.

@author: jorge
"""

import numpy as np

import openmdao.api as om

class Speed_inertial(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('omega', types=float)
        self.options.declare('r_ref', types=float)
    
    def setup(self):
        
        nn     = self.options['num_nodes']
        
        self.add_input('v',
                       val = np.zeros(nn),
                       desc = 'speed',
                       units = 'm/s')
        
        self.add_output('v_i',
                       val = np.zeros(nn),
                       desc = 'speed in inertial frame',
                       units = 'm/s')
        
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of = 'v_i', wrt = 'v', rows=ar, cols=ar , val = 1.0)
        
    def compute(self, inputs, outputs):
        
        omega = self.options['omega']
        r_ref = self.options['r_ref']
        v     = inputs['v']
        
        outputs['v_i'] = v + omega * r_ref