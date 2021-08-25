# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:42:45 2020

This group puts together the subsystems in charge of the propulsion models of the first and second stage.
The models for both stages are the same and call the same PropulsionStage group.

@author: jorge
"""

import openmdao.api as om

from .propulsionStage import PropulsionStage

class Propulsion(om.Group):
    
    def initialize(self):
         self.options.declare('g0', types=float,
                              desc='gravity at r0 (m/s**2)')
         
         self.options.declare('nb_e_first_stage', 
                             types = int, 
                             desc = 'number of engines for the first stage')
          
         self.options.declare('nb_e_second_stage', 
                             types = int, 
                             desc = 'number of engines for the first stage')
         
         self.options.declare('Rmc', types=float,
                             desc = 'gass constant times molecular weight (mJ/K/mol )',
                             default = 8314.0)
        
         self.options.declare('P_a', types = float, desc= 'atmospheric pressure. Set to zero to calculate values at vacuum.',
                             default = 0.0)
         

         self.options.declare('uncertainty_Isp_1', types = float, desc= 'uncertainty about Isp for stage 1 (s)',
                             default = 0.0)


         self.options.declare('uncertainty_Isp_2', types = float, desc= 'uncertainty about Isp for stage2 (s)',
                             default = 0.0)         
         
         self.options.declare('uncertainty_mfr_1', types = float, desc= 'uncertainty about mass flow rate for stage 1 (kg/s)',
                             default = 0.0)

         self.options.declare('uncertainty_mfr_2', types = float, desc= 'uncertainty about mass flow rate for stage 1 (kg/s)',
                             default = 0.0)
         
    
    def setup(self):
        
        g0                = self.options['g0']
        nb_e_first_stage  = self.options['nb_e_first_stage']
        nb_e_second_stage = self.options['nb_e_second_stage']
        Rmc               = self.options['Rmc']
        P_a               = self.options['P_a']
        uncertainty_Isp_1 = self.options['uncertainty_Isp_1']
        uncertainty_Isp_2 = self.options['uncertainty_Isp_2']
        uncertainty_mfr_1 = self.options['uncertainty_mfr_1']
        uncertainty_mfr_2 = self.options['uncertainty_mfr_2']
        

        
        self.add_subsystem('propulsion_stage_2', PropulsionStage(g0 = g0, nb_e = nb_e_second_stage, eta_C_f = 0.98, eta_cStar = 0.98, Rmc = Rmc, P_a = P_a,\
                                                                 uncertainty_Isp = uncertainty_Isp_2,uncertainty_mfr = uncertainty_mfr_2 ))

        self.add_subsystem('propulsion_stage_1', PropulsionStage(g0 = g0, nb_e = nb_e_first_stage, eta_C_f = 0.98, eta_cStar = 0.98, Rmc = Rmc, P_a = P_a,\
                                                                 uncertainty_Isp = uncertainty_Isp_1,uncertainty_mfr = uncertainty_mfr_1 ))

        
