# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:14:00 2020

This class calculates the nozzle expansion ratio.

@author: jorge
"""

import openmdao.api as om
from numpy import sqrt
from numpy import log as ln

class Noisy_thrust(om.ExplicitComponent):
    
    def initialize(self):


        self.options.declare('uncertainty_mfr', types = float, desc= 'uncertainty about mass flow rate (kg/s)',
                             default = 0.0)    
        
        self.options.declare('g0', types=float,
                              desc='gravity at r0 (m/s**2)')

        
    def setup(self):
        

        
        self.add_input('Isp',
                       val = 0.0,
                       desc='specific impulse at vacuum',
                       units = 's')

        
        self.add_input('thrust_nominal',
                       val = 0,
                       desc='Nominal thrust',
                       units = None)
        
        # ---------------------------------------------
        self.add_output('thrust',
                       val = 0.0,
                       desc='thrust at vacuum of all engines',
                       units = 'N')
        # declare partials
        
        self.declare_partials(of = 'thrust', wrt = 'thrust_nominal')
        self.declare_partials(of = 'thrust', wrt = 'Isp')

        
    def compute(self, inputs, outputs):
        

        
        outputs['thrust'] = inputs['thrust_nominal'] + self.options['g0']* inputs['Isp']*self.options['uncertainty_mfr']
        
    def compute_partials(self, inputs, jacobian):

        
        jacobian['thrust', 'thrust_nominal'] = 1
        jacobian['thrust', 'Isp']     = self.options['g0'] * self.options['uncertainty_mfr']
        
        