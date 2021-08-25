# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:14:00 2020

This class calculates the nozzle expansion ratio.

@author: jorge
"""

import openmdao.api as om
from numpy import sqrt
from numpy import log as ln

class ThrustCoefficient(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('P_a', types = float, desc= 'atmospheric pressure')
        
        self.options.declare('eta_C_f', types=float,
                             desc = 'C_f efficiency')
    
    def setup(self):
        
        self.add_input('gamma_t', 
                       val = 0,
                       desc ='heat capacity ratio at the throat',
                       units = None)
        
        self.add_input('P_c',
                       val = 0, 
                       desc='chamber pressure',
                       units = 'Pa')
        
        self.add_input('P_e', 
                       val = 0, 
                       desc = 'exit of the nozzle pressure',
                       units = 'Pa')
        
        self.add_input('epsilon',
                       val = 0,
                       desc='expansion ratio',
                       units = None)
        
        # ---------------------------------------------
        self.add_output('C_f',
                        val = 0.0,
                        desc = 'thrust coefficient',
                        units = None)
        
        # declare partials
        
        self.declare_partials(of = 'C_f', wrt = 'gamma_t')
        self.declare_partials(of = 'C_f', wrt = 'P_c')
        self.declare_partials(of = 'C_f', wrt = 'P_e')
        self.declare_partials(of = 'C_f', wrt = 'epsilon')
        
    def compute(self, inputs, outputs):
        
        gamma_t = inputs['gamma_t']
        P_c     = inputs['P_c']
        P_e     = inputs['P_e']
        epsilon = inputs['epsilon']
        
        P_a      = self.options['P_a']
        eta_C_f  = self.options['eta_C_f']
        
        aux = (2/(gamma_t - 1)) * (2/(gamma_t + 1))**((gamma_t+1)/(gamma_t-1)) * (1-(P_e/P_c)**((gamma_t-1)/gamma_t))
        
        outputs['C_f'] = eta_C_f * (gamma_t * sqrt(aux) + epsilon/P_c * (P_e - P_a) )
        
    def compute_partials(self, inputs, jacobian):
        gamma_t = inputs['gamma_t']
        P_c     = inputs['P_c']
        P_e     = inputs['P_e']
        epsilon = inputs['epsilon']
        
        P_a      = self.options['P_a']
        eta_C_f  = self.options['eta_C_f']
        
        jacobian['C_f', 'gamma_t'] = eta_C_f * ( sqrt(2)*sqrt(-(2/(gamma_t + 1))**((gamma_t + 1)/(gamma_t - 1))*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1)/(gamma_t - 1))*(gamma_t**2*(1 - (P_e/P_c)**((gamma_t - 1)/gamma_t))*(gamma_t - 1)**2/2 - gamma_t**2*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1)*(-2*(1 - gamma_t)*ln(2/(gamma_t + 1)) + (gamma_t - 1)**2)/2 + gamma_t*(gamma_t - 1)**3*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1) + (P_e/P_c)**((gamma_t - 1)/gamma_t)*(gamma_t - 1)**3*ln(P_e/P_c)/2)/(gamma_t*(gamma_t - 1)**3*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1)))
        jacobian['C_f', 'P_c']     = eta_C_f * ( (sqrt(2)*P_c*(P_e/P_c)**((gamma_t - 1)/gamma_t)*sqrt(-(2/(gamma_t + 1))**((gamma_t + 1)/(gamma_t - 1))*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1)/(gamma_t - 1))*(1 - gamma_t)/2 + epsilon*(P_a - P_e)*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1))/(P_c**2*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1)))
        jacobian['C_f', 'P_e']     = eta_C_f * ( (sqrt(2)*P_c*(P_e/P_c)**((gamma_t - 1)/gamma_t)*sqrt(-(2/(gamma_t + 1))**((gamma_t + 1)/(gamma_t - 1))*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1)/(gamma_t - 1))*(gamma_t - 1)/2 + P_e*epsilon*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1))/(P_c*P_e*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1)))
        jacobian['C_f', 'epsilon'] = eta_C_f * ( (-P_a + P_e)/P_c)
        
        