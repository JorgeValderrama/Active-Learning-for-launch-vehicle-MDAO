# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:35:18 2020

@author: jorge
"""

import openmdao.api as om
import numpy as np

class ConstraintsExitArea(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('areaFactor_1',
                             types = float,
                             desc = 'percentage of stage 1 area that can be used as nozzle exit area')
        
        self.options.declare('areaFactor_2',
                             types = float,
                             desc = 'percentage of stage 1 area that can be used as nozzle exit area')
    
    def setup(self):
        
        self.add_input('D_stage_1_m',
                        val = 0.0,
                        units = 'm',
                        desc = 'diameter of stage 1. Taken from massSizing')
        
        self.add_input('D_stage_2_m',
                        val = 0.0,
                        units = 'm',
                        desc = 'diameter of stage 2. Taken from mass')
        
        self.add_input('Ae_t_1',
                        val = 0.0,
                        units = 'm**2',
                        desc = 'Area of stage 1 taken from propulsion')
        
        self.add_input('Ae_t_2',
                        val = 0.0,
                        units = 'm**2',
                        desc = 'Area of stage 2 taken from propulsion')
        
        # ---------------------------------------------------------------------------
        
        self.add_output('residual_area_1',
                        val =0.0,
                        units = 'm**2',
                        desc = '')
        
        self.add_output('residual_area_2',
                        val =0.0,
                        units = 'm**2',
                        desc = '')
        
        
        self.declare_partials(of = 'residual_area_1', wrt = 'D_stage_1_m')
        self.declare_partials(of = 'residual_area_1', wrt = 'Ae_t_1')
        
        self.declare_partials(of = 'residual_area_2', wrt = 'D_stage_2_m')
        self.declare_partials(of = 'residual_area_2', wrt = 'Ae_t_2')
        
        
    def compute(self, inputs, outputs):
        
        areaFactor_1 = self.options['areaFactor_1']
        areaFactor_2 = self.options['areaFactor_2']
        
        D_stage_1_m = inputs['D_stage_1_m']
        D_stage_2_m = inputs['D_stage_2_m'] 
       
        Ae_t_1 = inputs['Ae_t_1']
        Ae_t_2 = inputs['Ae_t_2']
        
        outputs['residual_area_1'] = np.pi/4 * D_stage_1_m**2 * areaFactor_1 - Ae_t_1
        outputs['residual_area_2'] = np.pi/4 * D_stage_2_m**2 * areaFactor_2 - Ae_t_2

        
    def compute_partials(self, inputs, jacobian):
        
        areaFactor_1 = self.options['areaFactor_1']
        areaFactor_2 = self.options['areaFactor_2']
        
        D_stage_1_m = inputs['D_stage_1_m']
        D_stage_2_m = inputs['D_stage_2_m'] 
        
        
        jacobian['residual_area_1', 'D_stage_1_m'] = np.pi/2 * D_stage_1_m * areaFactor_1 
        jacobian['residual_area_1', 'Ae_t_1']      = -1
        
        jacobian['residual_area_2', 'D_stage_2_m'] = np.pi/2 * D_stage_2_m * areaFactor_2
        jacobian['residual_area_2', 'Ae_t_2']      = -1
        
       
      