# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:25:50 2021

This file defines the external_params object. 
Contains and defines the units, bounds and scaling parameters of 
the variables to be optimized related to the massSizing and propulsion modules.
@author: jorge
"""

import openmdao.api as om

external_params = om.IndepVarComp()

# define outputs for variables of propulsion second stage
external_params.add_output('P_c_stage_2',    units='Pa')
external_params.add_output('P_e_stage_2',    units='Pa')
external_params.add_output('o_f_stage_2',    units=None)
external_params.add_output('thrust_vac_stage_2_nominal', units='N')
# define variables for massSizing second stage
external_params.add_output('mp_2',  units='kg')

# define outputs for variables of propulsion first stage
external_params.add_output('P_c_stage_1', units='Pa')
external_params.add_output('P_e_stage_1', units='Pa')
external_params.add_output('o_f_stage_1', units=None)
external_params.add_output('thrust_vac_stage_1_nominal', units='N')
# define variables for massSizing first stage and constraint coupling variables
external_params.add_output('D_stage_1',    units='m')
external_params.add_output('mp_1',         units='kg')
external_params.add_output('max_n_f_1',    units= None)
external_params.add_output('max_q_dyn_1',  units= 'Pa')


# add design variables for propulsion second stage
# external_params.add_design_var('P_c_stage_2', units = 'Pa',
#                                 lower = 6e6 , upper = 10e6 ,
#                                 ref=10e6, ref0=6e6)

# external_params.add_design_var('P_e_stage_2', units = 'Pa',
#                                 lower = 1 , upper = 1e4,
#                                 ref=1e4, ref0=0)

# external_params.add_design_var('o_f_stage_2', units = None ,
#                                 lower = 2.0 , upper = 4.0 ,
#                                 ref=4.0, ref0=2.0)

# external_params.add_design_var('thrust_vac_stage_2_nominal', units = 'N' ,
#                                 lower = 1e5, upper = 1.4e6,
#                                 ref=1.4e6, ref0=1e5)

# add design variables for massSizing second stage
external_params.add_design_var('mp_2', units = 'kg' ,
                                lower = 50e3 , upper = 90e3 ,
                                ref=90e3, ref0=50e3)

# add design variables for propulsion first stage
# external_params.add_design_var('P_c_stage_1', units = 'Pa',
#                                 lower = 6e6 , upper = 10e6 ,
#                                 ref=10e6, ref0=6e6)

# # Lower limit for P_e is 0.4 * P0. Refer to Summerfield criterion.
# external_params.add_design_var('P_e_stage_1', units = 'Pa',
#                                 lower = 0.4 * 101325.0 , upper = 2e5,
#                                 ref=2e5, ref0=0.4 * 101325.0) 
 
# external_params.add_design_var('o_f_stage_1', units = None ,
#                                 lower = 2.0 , upper = 4.0 ,
#                                 ref=4.0, ref0=2.0)

# external_params.add_design_var('thrust_vac_stage_1_nominal', units = 'N' ,
#                                 lower = 5e6 , upper = 15e6 ,
#                                 ref=15e6, ref0=5e6)

# add design variables for massSizing first stage and constraint coupling variables
external_params.add_design_var('mp_1', units = 'kg' ,
                                lower = 150e3 , upper = 280e3 ,
                                ref=280e3, ref0=150e3)

# external_params.add_design_var('max_n_f_1', units = None,
#                                 lower = 1 , upper = 10,
#                                 ref=10, ref0=1)

# external_params.add_design_var('max_q_dyn_1', units = 'Pa',
#                                 lower = 1e1 , upper = 100e3,
#                                 ref=100e3, ref0=1e1)

# external_params.add_design_var('D_stage_1', units = 'm' ,
#                                 lower = 1 , upper = 5 ,
#                                 ref=5, ref0=1)