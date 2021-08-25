# -*- coding: utf-8 -*-
"""
Created on Sat May 09 2020

This file contains the definition of the epsilon component, which computes the expansion ratio of the engine. See Balesdents's thesis for details on the
formula used.

Modified on Thu October 22 2020 by j.valderrama

@author : g.fiorello
"""

#### Imports

import numpy as np
from numpy import log as ln
from numpy import sqrt as sqrt

from openmdao.api import ExplicitComponent

#### Component definition
"""
This class defines the epsilon component.

Inputs:
-P_c: chamber pressure #Pa
-P_e: exit of the nozzle pressure #Pa
-gamma_t: heat capacity ratio at the throat #unitless

Ouputs:
-epsilon: expansion ratio #unitless
"""
class AreaRatio(ExplicitComponent):

    def setup(self):
        #Inputs outside the module
        # I modified the entries for all "val" to 0 - j.valderrama
        self.add_input('P_c',
                       val = 0, 
                       desc='chamber pressure',
                       units = 'Pa')
        
        self.add_input('P_e', 
                       val = 0, 
                       desc = 'exit of the nozzle pressure',
                       units = 'Pa')

        #Inputs inside the module
        self.add_input('gamma_t', 
                       val = 0,
                       desc ='heat capacity ratio at the throat',
                       units = None)

        #Outputs
        self.add_output('epsilon',
                        val = 0,
                        desc='expansion ratio',
                        units = None)

        #Partial derivatives declaration
        self.declare_partials(['epsilon'], ['P_c','P_e', 'gamma_t'])


    def compute(self,inputs, outputs):
        #Inputs of the component
        P_c     = inputs ['P_c']
        P_e     = inputs ['P_e']
        gamma_t = inputs ['gamma_t']

        #Computation of the expansion ratio
        A = (2/(gamma_t+1)) ** (1/(gamma_t-1)) #unitless
        B = (P_c/P_e) ** (1/gamma_t) #unitless
        C = (gamma_t+1)/(gamma_t-1) #unitless
        D = 1 - (P_e/P_c) ** ((gamma_t-1)/gamma_t) #unitless
        num = A * B #unitless
        den = np.sqrt(C * D) #unitless
        epsilon = num / den #unitless

        #Ouputs of the component
        outputs['epsilon'] = epsilon
        

    def compute_partials(self, inputs, partials):
        #Inputs of the component
        P_c     = inputs ['P_c']
        P_e     = inputs ['P_e']
        gamma_t = inputs ['gamma_t']

        #Partial derivatives of the component
        partials['epsilon', 'P_c'] = -(2/(gamma_t + 1))**(1/(gamma_t - 1))*(P_c/P_e)**(1/gamma_t)*sqrt(-(gamma_t + 1)*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1)/(gamma_t - 1))*(gamma_t - 1)*((P_e/P_c)**((gamma_t - 1)/gamma_t)*(gamma_t - 1) + 2*(P_e/P_c)**((gamma_t - 1)/gamma_t) - 2)/(2*P_c*gamma_t*(gamma_t + 1)*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1)**2)
        partials['epsilon', 'P_e'] = (2/(gamma_t + 1))**(1/(gamma_t - 1))*(P_c/P_e)**(1/gamma_t)*sqrt(-(gamma_t + 1)*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1)/(gamma_t - 1))*(gamma_t - 1)*((P_e/P_c)**((gamma_t - 1)/gamma_t)*(gamma_t - 1) + 2*(P_e/P_c)**((gamma_t - 1)/gamma_t) - 2)/(2*P_e*gamma_t*(gamma_t + 1)*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1)**2)
        partials['epsilon', 'gamma_t'] = (2/(gamma_t + 1))**(1/(gamma_t - 1))*(P_c/P_e)**(1/gamma_t)*(gamma_t**2*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1)*(gamma_t + (gamma_t + 1)*ln(2/(gamma_t + 1)) - 1) + (gamma_t - 1)**2*(gamma_t + 1)*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1)*ln(P_c/P_e) + (gamma_t - 1)*(gamma_t**2*(gamma_t - 1)*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1) - gamma_t**2*(gamma_t + 1)*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1) + (P_e/P_c)**((gamma_t - 1)/gamma_t)*(gamma_t - 1)*(gamma_t + 1)*ln(P_e/P_c))/2)/(gamma_t**2*(-(gamma_t + 1)*((P_e/P_c)**((gamma_t - 1)/gamma_t) - 1)/(gamma_t - 1))**(3/2)*(gamma_t - 1)**3)