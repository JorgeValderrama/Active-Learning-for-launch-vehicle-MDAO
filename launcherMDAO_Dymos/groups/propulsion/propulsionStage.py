# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 07:55:08 2020

This group calculates the thrust parameters at vacuum.
Nomenclature is based on Balesdent but equations are taken from Sutton.
The definition of efficiencies follows Sutton's.
@author: jorge
"""

import openmdao.api as om

from .components.chemistry import Chemistry
from .components.massFlowRate import MassFlowRate
from .components.nozzleExitArea import NozzleExitArea
from .components.throatArea import ThroatArea
from .components.noisy_thrust import Noisy_thrust
class PropulsionStage(om.Group):
    
    def initialize(self):
        
        self.options.declare('g0', types=float,
                              desc='gravity at r0 (m/s**2)')
        
        self.options.declare('nb_e', types=int,
                              desc='number of engines')
        
        self.options.declare('eta_C_f', types=float,
                             desc = 'C_f efficiency')
        
        self.options.declare('eta_cStar', types=float,
                             desc = 'cStar efficiency')
        
        self.options.declare('Rmc', types=float,
                             desc = 'gass constant times molecular weight (mJ/K/mol )')
        
        self.options.declare('P_a', types = float, desc= 'atmospheric pressure. Set to zero to calculate values at vacuum.')
        
        self.options.declare('P_a', types = float, desc= 'atmospheric pressure. Set to zero to calculate values at vacuum.')
        
        self.options.declare('uncertainty_Isp', types = float, desc= 'uncertainty about Isp (s)',
                             default = 0.0)

         
        self.options.declare('uncertainty_mfr', types = float, desc= 'uncertainty about mass flow rate (s)',
                             default = 0.0)         
                
        
    def setup(self):
        
        g0         = self.options['g0']
        eta_C_f    = self.options['eta_C_f']
        eta_cStar  = self.options['eta_cStar']
        Rmc        = self.options['Rmc']
        nb_e       = self.options['nb_e']
        P_a        = self.options['P_a']
        uncertainty_Isp = self.options['uncertainty_Isp']
        
        
        self.add_subsystem('chemistry', Chemistry(g0 = g0, eta_C_f = eta_C_f, eta_cStar = eta_cStar, Rmc = Rmc, P_a = P_a,uncertainty_Isp = uncertainty_Isp ),
                           promotes_inputs = ['P_c', 'P_e', 'o_f'],
                           promotes_outputs = ['Isp', 'cStar', 'epsilon'])
        
        self.add_subsystem('noisy_thrust', Noisy_thrust(g0 = g0,uncertainty_mfr = self.options['uncertainty_mfr'] ),
                           promotes_inputs = ['Isp', 'thrust_nominal'],
                           promotes_outputs = ['thrust'])
                
        
        self.add_subsystem('massFlowRate', MassFlowRate( g0 = g0),
                           promotes_inputs = ['thrust', 'Isp'],
                           promotes_outputs = ['mfr_max'])
        
        self.add_subsystem('throatArea', ThroatArea(nb_e = nb_e),
                           promotes_inputs = ['P_c', 'cStar', 'mfr_max'],
                           promotes_outputs = ['At'])
        
        self.add_subsystem('nozzleExitArea', NozzleExitArea(nb_e = nb_e),
                           promotes_inputs = ['At', 'epsilon'],
                           promotes_outputs = ['Ae', 'Ae_t'])
        