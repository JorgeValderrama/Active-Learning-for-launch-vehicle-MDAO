# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:32:25 2020

This function calculates the local dynamic pressure and heat flux 

@author: jorge
"""

import openmdao.api as om
import numpy as np

class HeatFluxAndDynamicPressure(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')
        
    def setup(self):
    
        nn = self.options['num_nodes']
        
        self.add_input('v',
                        val = np.zeros(nn),
                        desc = 'speed',
                        units = 'm/s')
        
        self.add_input('rho',
                       val=np.zeros(nn),
                       desc='density',
                       units='kg/m**3')
        
        self.add_output('q_dyn',
                        val=np.zeros(nn),
                        desc='dynamic pressure',
                        units='Pa')
        
        self.add_output('q_heat',
                        val=np.zeros(nn),
                        desc='heat flux',
                        units='W/m**2')
        
        # Setup partials by JLVZ
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of = 'q_dyn', wrt = 'v' , rows=ar, cols=ar)
        self.declare_partials(of = 'q_dyn', wrt = 'rho' , rows=ar, cols=ar)
        
        self.declare_partials(of = 'q_heat', wrt = 'v' , rows=ar, cols=ar)
        self.declare_partials(of = 'q_heat', wrt = 'rho' , rows=ar, cols=ar)
        
    def compute(self, inputs, outputs):
        
        v   = inputs['v']
        rho = inputs['rho']
        
        outputs['q_dyn'] = 0.5 * rho * v**2
        
        outputs['q_heat'] = 0.5 * rho * v**3
        
    def compute_partials(self, inputs, jacobian):
        
        v   = inputs['v']
        rho = inputs['rho']
        
        jacobian['q_dyn','v']    = rho * v
        jacobian['q_dyn','rho']  = 0.5 * v**2
        
        jacobian['q_heat','v']   = 1.5 * rho * v**2
        jacobian['q_heat','rho'] = 0.5 * v**3
    