# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:30:58 2020

This class calculates the specific impulse at vacuum.

@author: jorge
"""

import openmdao.api as om

class SpecificImpulse(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('g0', types=float,
                              desc='gravity at r0 (m/s**2)')
        
        self.options.declare('uncertainty_Isp', types = float, desc= 'uncertainty about Isp (s)',
                             default = 0.0)        
    def setup(self):
        
        self.add_input('cStar',
                       val = 0.0,
                       desc='characteristic velocity of the propellants',
                       units = 'm/s')
        
        self.add_input('C_f',
                       val = 0.0,
                       desc = 'thrust coefficient',
                       units = None)
        
        self.add_output('Isp',
                        val = 0.0,
                        desc='specific impulse at vacuum',
                        units = 's')
        
        self.declare_partials(of='Isp', wrt = 'cStar')
        self.declare_partials(of='Isp', wrt = 'C_f')
        
    def compute(self, inputs, outputs):
        
        g0         = self.options['g0']
        
        cStar = inputs['cStar']
        C_f   = inputs['C_f']
        
        outputs['Isp'] =  cStar / g0 * C_f + self.options['uncertainty_Isp']
        
        # print('Isp (s)'.ljust(20) + str(outputs['Isp']))
        
    def compute_partials(self, inputs, jacobian):
        
        g0         = self.options['g0']
        
        cStar = inputs['cStar']
        C_f   = inputs['C_f']
        
        jacobian['Isp', 'cStar'] = C_f  / g0 
        jacobian['Isp', 'C_f']   = cStar / g0 