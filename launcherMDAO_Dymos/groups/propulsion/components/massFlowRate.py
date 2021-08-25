# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:12:12 2020

This class calculates the maximum mass flow rate.
It's named like that because this code is intended to be used with a dynamic control in the throttle.
The throttle is asumed to take values between [ 0 , 1 ] to vary the mass flow rate.

@author: jorge
"""

import openmdao.api as om

class MassFlowRate(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('g0', types=float,
                              desc='gravity at r0 (m/s**2)')
       
        
    def setup(self):
        
        self.add_input('thrust',
                       val = 0.0,
                       desc='thrust at vacuum of all engines',
                       units = 'N')
        
        self.add_input('Isp',
                       val = 0.0,
                       desc='specific impulse at vacuum',
                       units = 's')
        
        self.add_output('mfr_max',
                        val = 0.0,
                        desc='maximum mass flow rate of all engines',
                        units = 'kg/s')
        
        # Setup partials by JLVZ
        
        self.declare_partials(of = 'mfr_max', wrt = 'thrust')
        self.declare_partials(of = 'mfr_max', wrt = 'Isp')
        
    def  compute(self,inputs,outputs):
        
        g0 = self.options['g0']
        
        thrust  = inputs['thrust']
        Isp     = inputs['Isp']
        
        outputs['mfr_max'] =  thrust / (Isp * g0)
        
        # print('mfr_max (kg/s)'.ljust(20) + str( outputs['mfr_max']))
        # print('Isp'.ljust(20) + str( Isp ))
    
    def compute_partials(self, inputs, jacobian):
        
        g0 = self.options['g0']
        
        thrust  = inputs['thrust']
        Isp     = inputs['Isp']
        
        jacobian['mfr_max', 'thrust']  = 1 / (Isp * g0)
        jacobian['mfr_max', 'Isp']    = - thrust / (Isp**2 * g0)