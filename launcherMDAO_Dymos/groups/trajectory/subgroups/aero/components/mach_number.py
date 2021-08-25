# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:53:54 2020

This function calculates the Mach number at a given speed and local speed of sound.

@author: jorge
"""

import openmdao.api as om
import numpy as np

class Mach_number(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')
        
    def setup(self):
    
        nn = self.options['num_nodes']
        
        self.add_input('v',
                        val = np.zeros(nn),
                        desc = 'speed',
                        units = 'm/s')
        
        self.add_input('sos', 
                       val=np.zeros(nn), 
                       desc = 'speed of sound',
                       units='m/s')
        
        self.add_output('Mach',
                        val = np.zeros(nn),
                        desc = 'Mach number',
                        units = None)
        
        # Setup partials 
        ar = np.arange(self.options['num_nodes'])
            
        self.declare_partials(of = 'Mach', wrt = 'v', rows=ar, cols=ar)
        self.declare_partials(of = 'Mach', wrt = 'sos', rows=ar, cols=ar)
    
    def compute(self, inputs, outputs):
        
        v   = inputs['v']
        sos = inputs['sos']
        
        outputs['Mach'] = v / sos
            
    def compute_partials(self, inputs, jacobian):
        
        v   = inputs['v']
        sos = inputs['sos']
        
        jacobian['Mach','v']   = 1 /sos
        jacobian['Mach','sos'] = -v/sos**2
        
        
        