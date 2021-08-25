# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:33:50 2020

This class finds the specifics energy and momentum of the orbit based on the inertial speed, radius and flight path angle

@author: jorge
"""

import numpy as np

import openmdao.api as om

class EnergyAndMomentum(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('mu', desc = 'mu=GMe (m**3/s**2)')
        self.options.declare('num_nodes', types=int)
        
    def setup(self):
        
        nn     = self.options['num_nodes']
        
        self.add_input('r',
                       val = np.zeros(nn),
                       desc = 'radius',
                       units = 'm')
        
        self.add_input('v_i',
                       val = np.zeros(nn),
                       desc = 'speed in inertial frame',
                       units = 'm/s')
        
        self.add_input('phi',
                       val = np.zeros(nn),
                       desc = 'flight path angle',
                       units = 'rad')
        # -------------------------------------------------------
        self.add_output('E',
                        val = np.zeros(nn),
                        desc = 'total energy per unit mass',
                        units = 'm**2/s**2')
        
        self.add_output('H',
                        val = np.zeros(nn),
                        desc = 'angular momentum per unit mass',
                        units = 'm**2/s')
        
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of = 'E', wrt = 'r', rows=ar, cols=ar)
        self.declare_partials(of = 'E', wrt = 'v_i', rows=ar, cols=ar)
        
        self.declare_partials(of = 'H', wrt = 'r', rows=ar, cols=ar)
        self.declare_partials(of = 'H', wrt = 'v_i', rows=ar, cols=ar)
        self.declare_partials(of = 'H', wrt = 'phi', rows=ar, cols=ar)
        
    def compute(self, inputs, outputs):
        
        v   = inputs['v_i']
        r   = inputs['r']
        phi = inputs['phi']
        
        mu = self.options['mu']
        
        outputs['E'] = v**2/2 - mu/r
        
        outputs['H'] = r * v * np.cos(phi)
        
    def compute_partials(self, inputs, jacobian):
        
        v   = inputs['v_i']
        r   = inputs['r']
        phi = inputs['phi']
        
        mu = self.options['mu']
        
        jacobian['E', 'r']   = mu / r**2
        jacobian['E', 'v_i']   = v
        
        jacobian['H', 'r']   = v * np.cos(phi)
        jacobian['H', 'v_i'] = r * np.cos(phi)
        jacobian['H', 'phi'] = r * v * (-1) * np.sin(phi)
        
        
        
        
        