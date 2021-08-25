# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:41:51 2020

This clasee calculates the characteristic velocity of the propellants.

@author: jorge
"""

import openmdao.api as om
import numpy as np

class CharacteristicVelocity(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('Rmc', types=float,
                             desc = 'gass constant times molecular weight')
        
        self.options.declare('eta_cStar', types=float,
                             desc = 'cStar efficiency')
    
    def setup(self):
        
        self.add_input('gamma_t',
                        val=0.0,
                        desc='Isentropic coefficient at the throat',
                        units=None)
        
        self.add_input('tc',
                        val=0.0,
                        desc='flame temperature',
                        units='K')
        
        self.add_input('mc',
                        val=0.0,
                        desc='molecular mass at combustion',
                        units='g/mol')
        
        self.add_output('cStar',
                        val=0.0,
                        desc='characteristic velocity of the propellants',
                        units = 'm/s')
        
        # declare partials
        
        self.declare_partials(of = 'cStar', wrt = 'gamma_t')
        self.declare_partials(of = 'cStar', wrt = 'tc')
        self.declare_partials(of = 'cStar', wrt = 'mc')
        
    def compute(self, inputs, outputs):
        
        Rmc        = self.options['Rmc']
        eta_cStar  = self.options['eta_cStar']
        
        gamma = inputs['gamma_t']
        tc    = inputs['tc']
        mc    = inputs['mc']
        
        outputs['cStar'] = eta_cStar * (( (np.sqrt(gamma * Rmc/mc * tc)) / (gamma * (2/(gamma + 1)) **((gamma + 1)/(2*(gamma -1))) ) ) )
            
        # print('cStar (m/s)'.ljust(20) + str(outputs['cStar']))
        
    def compute_partials(self, inputs, jacobian):
        
        Rmc        = self.options['Rmc']
        eta_cStar  = self.options['eta_cStar']
        
        gamma = inputs['gamma_t']
        tc    = inputs['tc']
        mc    = inputs['mc']
        
        jacobian['cStar','gamma_t'] = eta_cStar * ( (2/(gamma + 1))**(-(gamma + 1)/(2*(gamma - 1)))*np.sqrt(Rmc*gamma*tc/mc)*(gamma*(gamma + 2*np.log(2/(gamma + 1)) - 1) - (gamma - 1)**2)/(2*gamma**2*(gamma - 1)**2))
        jacobian['cStar','tc']      = eta_cStar * ( (2/(gamma + 1))**(-(gamma + 1)/(2*(gamma - 1)))*np.sqrt(Rmc*gamma*tc/mc)/(2*gamma*tc))
        jacobian['cStar','mc']      = eta_cStar * (-(2/(gamma + 1))**(-(gamma + 1)/(2*(gamma - 1)))*np.sqrt(Rmc*gamma*tc/mc)/(2*gamma*mc))
        