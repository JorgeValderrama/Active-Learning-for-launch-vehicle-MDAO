# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 22:29:15 2020


This class calculates the "thrust" and mass flow rate "mfr" to be used by the trajectory.
It models the thrust losses as function of atmospheric pressure due to nozzle expansion. 
It allows to apply throttle by modifying the "mfr".
Throttle can be modeled as a dynamic control but for the moment it is just a constant 
with a value of 1.

@author: jorge
"""

import openmdao.api as om
import numpy as np

class Thrust_losses(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('num_nodes', types=int)
        
        
    def setup(self):
        
        nn = self.options['num_nodes']
        
        self.add_input('thrust_vac',
                       val = 0.0,
                       desc='thrust at vacuum',
                       units = 'N')
        
        self.add_input('throttle',
                        val=np.zeros(nn),
                        desc= 'throttle. Takes values between 0 and 1',
                        units=None)
        
        self.add_input('P_a',
                       val = np.zeros(nn),
                       desc = 'atmospheric pressure',
                       units = 'Pa')
        
        self.add_input('Ae_t',
                       val = 0,
                       desc='total nozzle exit area',
                       units = 'm**2')
        
        self.add_input('mfr_max',
                       val = 0.0,
                       desc='maximum mass flow rate.',
                       units = 'kg/s')
        
        self.add_output('thrust',
                        val=np.zeros(nn),
                        desc='thrust accounting for losses and throttle',
                        units='N')
        
        self.add_output('mfr',
                        val = np.zeros(nn),
                        desc='mass flow rate after throttle is applied',
                        units = 'kg/s')
        
        # declare partials
        
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of = 'thrust', wrt = 'thrust_vac')
        self.declare_partials(of = 'thrust', wrt = 'throttle', rows=ar, cols=ar)
        self.declare_partials(of = 'thrust', wrt = 'P_a', rows=ar, cols=ar)
        self.declare_partials(of = 'thrust', wrt = 'Ae_t')
        
        self.declare_partials(of = 'mfr', wrt = 'mfr_max')
        self.declare_partials(of = 'mfr', wrt = 'throttle', rows=ar, cols=ar)
        
    def compute(self, inputs, outputs):
        
        thrust_vac  = inputs['thrust_vac']
        throttle    = inputs['throttle']
        P_a         = inputs['P_a']
        # P_e         = inputs['P_e']
        Ae_t        = inputs['Ae_t']  
        mfr_max     = inputs['mfr_max']
        
        outputs['thrust'] = thrust_vac * throttle  -  Ae_t * P_a
        outputs['mfr']    = mfr_max * throttle
        
        
    def compute_partials(self, inputs, jacobian):
        
        thrust_vac  = inputs['thrust_vac']
        throttle    = inputs['throttle']
        P_a         = inputs['P_a']
        Ae_t        = inputs['Ae_t']   
        mfr_max     = inputs['mfr_max']
        
        jacobian['thrust','thrust_vac']  = throttle
        jacobian['thrust','throttle']    = thrust_vac
        jacobian['thrust','P_a']         = - Ae_t
        jacobian['thrust','Ae_t']        = - P_a
        
        jacobian['mfr', 'throttle']      = mfr_max
        jacobian['mfr', 'mfr_max']       = throttle