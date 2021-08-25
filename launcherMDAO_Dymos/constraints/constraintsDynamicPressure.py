# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:35:18 2020

@author: jorge
"""

import openmdao.api as om
import numpy as np

class ConstraintsDynamicPressure(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('num_nodes', types=int, default = 21, desc='Number of nodes to be evaluated in the RHS')
    
    def setup(self):
        
        nn     = self.options['num_nodes']
        
        self.add_input('max_q_dyn_1_m',
                        val = 0.0,
                        units = 'Pa',
                        desc = 'maximum dynamic pressure. taken from massSizing')
        
        
        self.add_input('max_q_dyn_1_t',
                        val = 0.0,
                        units = 'Pa',
                        desc = 'maximum dynamic pressure. taken from trajectory')
      
        
        self.add_output('residual_max_q_dyn',
                        val = 0.0,
                        units = 'Pa',
                        desc = 'residual of operation of max_q_dyn')
        
        self.declare_partials(of = 'residual_max_q_dyn', wrt = 'max_q_dyn_1_m', val = 1)
        self.declare_partials(of = 'residual_max_q_dyn', wrt = 'max_q_dyn_1_t', val = -1)
        
        
    def compute(self, inputs, outputs):
        
        max_q_dyn_1_m = inputs['max_q_dyn_1_m']
        max_q_dyn_1_t = inputs['max_q_dyn_1_t']
        
        outputs['residual_max_q_dyn'] = max_q_dyn_1_m - max_q_dyn_1_t
        