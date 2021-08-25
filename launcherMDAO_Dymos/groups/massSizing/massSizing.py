# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 08:02:31 2020

This group calls inegrates the mass sizing models of first and second stages as in FELIN
The first stage mass models are based on Castellini.
The second stage model is much simpler.

@author: jorge
"""

import openmdao.api as om

from .components.Dry_Mass_stage_1 import Dry_Mass_stage_1_Comp
from .components.Dry_Mass_stage_2 import Dry_Mass_stage_2_Comp 

class MassSizing(om.Group):
    
    
    def initialize(self):
          
          self.options.declare('nb_e_first_stage', 
                             types = int, 
                             desc = 'number of engines for the first stage')
        
          self.options.declare('mass_aux_1', 
                             types = float, 
                             desc = 'auxiliary mass for the first stage in kg')
          
          self.options.declare('uncertainty_dry_mass_1', 
                             types = float, 
                             desc = 'uncertainty of dry mass for the first stage in  kg')

          self.options.declare('uncertainty_dry_mass_2', 
                             types = float, 
                             desc = 'uncertainty of dry mass for the second stage in  kg')
          
          
          
    def setup(self):
        mass_aux_1       = self.options['mass_aux_1']
        nb_e_first_stage = self.options['nb_e_first_stage']
        
        u_ms_1 = self.options['uncertainty_dry_mass_1']
        u_ms_2 = self.options['uncertainty_dry_mass_2']
        

        
        self.add_subsystem('dry_mass_stage_1', Dry_Mass_stage_1_Comp( mass_aux_1 = mass_aux_1, nb_e = nb_e_first_stage,\
                                                                     uncertainty_dry_mass_1 = u_ms_1 ),
                           promotes_inputs = ['*'],
                           promotes_outputs = [('Dry_mass_stage_1', 'ms_1')])
            
            

        self.add_subsystem('dry_mass_stage_2', Dry_Mass_stage_2_Comp(uncertainty_dry_mass_2= u_ms_2),
                           promotes_inputs = ['*'],
                           promotes_outputs = [('Dry_mass_stage_2', 'ms_2')])