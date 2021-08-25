# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:01:35 2020

Newton's gravity model.
This function calculates the gravitational acceleration "g" of a body located
at a radius "r" from a central body with standard gravitation parameter "mu".

@author: jorge
"""

import numpy as np

import openmdao.api as om

i = 0
class Gravity (om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('num_nodes', types=int)
        
        self.options.declare('mu', types=float,
                             desc = 'Standard gravitational parameter Earth (m**3/s**2)')
        
    def setup(self):
        nn     = self.options['num_nodes']
        
        self.add_input('r',
                       val = np.zeros(nn),
                       desc = 'radius',
                       units = 'm')
        
        self.add_output('g',
                        val = np.zeros(nn),
                        desc = 'gravity acceleration',
                        units = 'm/s**2')
        
        # Setup partials 
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of = 'g', wrt = 'r', rows=ar, cols=ar)
        
    def compute(self, inputs, outputs):
        
        r       = inputs['r']
        
        mu = self.options['mu']
        
        outputs['g'] = mu/r**2
        
        
    def compute_partials(self, inputs, jacobian):
        
        r       = inputs['r']
        
        mu = self.options['mu']
        
        jacobian['g', 'r']   = -2 * mu / r**3