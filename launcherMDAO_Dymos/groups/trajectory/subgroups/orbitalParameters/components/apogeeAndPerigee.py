# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:51:02 2020

This class calculates the radius at apogee and perigee

@author: jorge
"""

import numpy as np

import openmdao.api as om

class ApogeeAndPerigee(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        
        nn     = self.options['num_nodes']
        
        self.add_input('e',
                        val = np.zeros(nn),
                        desc = 'eccentricity',
                        units = None)
        
        self.add_input('a',
                        val = np.zeros(nn),
                        desc = 'semi major axis',
                        units = 'm')
        
        # ------------------------------------------
        
        self.add_output('ra',
                        val = np.zeros(nn),
                        desc = 'radius at apogee',
                        units = 'm')
        
        self.add_output('rp',
                        val = np.zeros(nn),
                        desc = 'radius at perigee',
                        units = 'm')
        
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of = 'ra', wrt = 'e', rows=ar, cols=ar)
        self.declare_partials(of = 'ra', wrt = 'a', rows=ar, cols=ar)
        
        self.declare_partials(of = 'rp', wrt = 'e', rows=ar, cols=ar)
        self.declare_partials(of = 'rp', wrt = 'a', rows=ar, cols=ar)
        
    def compute(self, inputs, outputs):
        
        e = inputs['e']
        a = inputs['a']
        
        outputs['rp'] = a * (1 - e)
        outputs['ra'] = a * (1 + e)
        
    def compute_partials(self, inputs, jacobian):
        
        e = inputs['e']
        a = inputs['a']
        
        jacobian['rp', 'e'] = -a
        jacobian['rp', 'a'] = 1 - e
        
        jacobian['ra', 'a'] = 1 + e
        jacobian['ra', 'e'] = a
         