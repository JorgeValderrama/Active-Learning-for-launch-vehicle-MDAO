# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:02:36 2020
This class finds the eccentricity and tsemi major axis of the orbit based on the
Specific energy and momentum
@author: jorge
"""

import numpy as np

import openmdao.api as om

class EccentricityAndMajorAxis(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('mu', desc = 'mu=GMe (m**3/s**2)')
        self.options.declare('num_nodes', types=int)
        
    def setup(self):
        
        nn     = self.options['num_nodes']
        
        self.add_input('E',
                        val = np.zeros(nn),
                        desc = 'total energy per unit mass',
                        units = 'm**2/s**2')
        
        self.add_input('H',
                        val = np.zeros(nn),
                        desc = 'angular momentum per unit mass',
                        units = 'm**2/s')
        
        # -----------------------------------------------------------
        
        self.add_output('e',
                        val = np.zeros(nn),
                        desc = 'eccentricity',
                        units = None)
        
        self.add_output('a',
                        val = np.zeros(nn),
                        desc = 'semi major axis',
                        units = 'm')
        
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of = 'e', wrt = 'H', rows=ar, cols=ar)
        self.declare_partials(of = 'e', wrt = 'E', rows=ar, cols=ar)
        
        self.declare_partials(of = 'a', wrt = 'E', rows=ar, cols=ar)
        
    def compute(self, inputs, outputs):
        
        E = inputs['E']
        H = inputs['H']
        
        mu = self.options['mu']
        
        outputs['a'] = -mu / (2*E)
        
        outputs['e'] = np.sqrt( 1 + 2 * (H**2*E) / (mu**2) ) 
        
    def compute_partials(self, inputs, jacobian):
        
        E = inputs['E']
        H = inputs['H']
        
        mu = self.options['mu']
        
        k = 2 * H**2 / mu**2
        c = 2 * E / mu**2
        
        jacobian['a', 'E'] = mu / (2*E**2)
        
        jacobian['e', 'E'] = k / (2 * np.sqrt(1 + k * E))
        jacobian['e', 'H'] = c * H / ( np.sqrt(1 + c * H**2) )