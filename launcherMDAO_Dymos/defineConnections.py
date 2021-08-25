# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 00:45:11 2021

@author: jorge
"""

def defineConnections(p):

    
    # connections for propulsion module
    # =======================================================================================================
    p.model.connect('external_params.P_c_stage_2',        'propulsion.propulsion_stage_2.P_c')
    p.model.connect('external_params.P_e_stage_2',        'propulsion.propulsion_stage_2.P_e')
    p.model.connect('external_params.o_f_stage_2',        'propulsion.propulsion_stage_2.o_f')
    p.model.connect('external_params.thrust_vac_stage_2_nominal', 'propulsion.propulsion_stage_2.thrust_nominal')
    
    p.model.connect('propulsion.propulsion_stage_2.Ae_t',    'traj.input_parameters:Ae_t_stage_2')
    p.model.connect('propulsion.propulsion_stage_2.mfr_max', 'traj.input_parameters:mfr_max_stage_2')
    p.model.connect('propulsion.propulsion_stage_2.Isp',     'traj.input_parameters:Isp_stage_2')
    p.model.connect('propulsion.propulsion_stage_2.thrust',    'traj.input_parameters:thrust_vac_stage_2')
    
    p.model.connect('external_params.P_c_stage_1',        'propulsion.propulsion_stage_1.P_c')
    p.model.connect('external_params.P_e_stage_1',        'propulsion.propulsion_stage_1.P_e')
    p.model.connect('external_params.o_f_stage_1',        'propulsion.propulsion_stage_1.o_f')
    p.model.connect('external_params.thrust_vac_stage_1_nominal', 'propulsion.propulsion_stage_1.thrust_nominal')
    
    p.model.connect('propulsion.propulsion_stage_1.Ae_t',    'traj.input_parameters:Ae_t_stage_1')
    p.model.connect('propulsion.propulsion_stage_1.mfr_max', 'traj.input_parameters:mfr_max_stage_1')
    p.model.connect('propulsion.propulsion_stage_1.thrust',    'traj.input_parameters:thrust_vac_stage_1')
    

    # connections mass sizing
    # =======================================================================================================
    p.model.connect('external_params.mp_2',         'massSizing.Prop_mass_stage_2')
    p.model.connect('external_params.mp_1',         'massSizing.mp')
    
    p.model.connect('external_params.max_n_f_1' ,   'massSizing.n_ax_max')
    p.model.connect('external_params.max_q_dyn_1' , 'massSizing.P_dyn_max')
    
    p.model.connect('external_params.o_f_stage_1', 'massSizing.o_f')
    p.model.connect('propulsion.propulsion_stage_1.thrust', 'massSizing.thrust_vac_stage_1')
    
    p.model.connect('external_params.D_stage_1', 'massSizing.D_stage_1')
    p.model.connect('external_params.D_stage_1', 'massSizing.D_stage_2')
    
    # Connect diameters to aero
    
    
    p.model.connect('external_params.D_stage_1', 'traj.input_parameters:D_stage_1')
    # p.model.connect('external_params.D_stage_1', 'traj.input_parameters:D_stage_2')
    
    
    #%% add jettison component 

    p.model.connect('massJettison.massjettison_first_stage', 'constraintsJettison.massjettison_first_stage')
    p.model.connect('massJettison.massjettison_plf', 'constraintsJettison.massjettison_plf')
    
    
    p.model.connect('massSizing.ms_1', 'constraintsJettison.ms_1')
    p.model.connect('massSizing.ms_2', 'constraintsJettison.ms_2')
    
    p.model.connect('external_params.mp_1', 'constraintsPropellants.mp_1_propulsion')
    p.model.connect('external_params.mp_2', 'constraintsPropellants.mp_2_propulsion')
    p.model.connect('traj.lift_off.timeseries.states:m', 'constraintsPropellants.mf_a', src_indices=[0])
    p.model.connect('traj.gravity_turn_b.timeseries.states:m', 'constraintsPropellants.me_a', src_indices=[-1])
    p.model.connect('traj.gravity_turn_c.timeseries.states:m', 'constraintsPropellants.mf_b', src_indices=[0])
    p.model.connect('traj.exoatmos_b.timeseries.m_final', 'constraintsPropellants.m_final', src_indices=[-1])
    
    p.model.connect('traj.gravity_turn_b.timeseries.states:m', 'massJettison.me_a', src_indices=[-1])
    p.model.connect('traj.gravity_turn_c.timeseries.states:m', 'massJettison.mf_b', src_indices=[0])
    p.model.connect('traj.exoatmos_a.timeseries.states:m', 'massJettison.mi_b', src_indices=[-1])
    p.model.connect('traj.exoatmos_b.timeseries.states:m', 'massJettison.mi_c', src_indices=[0])
    p.model.connect('traj.exoatmos_b.timeseries.m_final' , 'constraintsJettison.m_final', src_indices=[-1])
    
    # careful with this. Im not actually choosing the max
    p.model.connect('traj.gravity_turn_b.timeseries.n_f' , 'constraintsLoadFactor.max_n_f_1_t', src_indices=[-1])
    # p.model.connect('traj.exoatmos_a.timeseries.n_f' , 'constraintsLoadFactor.max_n_f_2_t', src_indices=[-1])
    
    p.model.connect('external_params.max_n_f_1', 'constraintsLoadFactor.max_n_f_1_p')
    # p.model.connect('external_params.max_n_f_2', 'constraintsLoadFactor.max_n_f_2_p')
    
    
    
    
    p.model.connect('external_params.max_q_dyn_1', 'constraintsDynamicPressure.max_q_dyn_1_m')
    
    
    
    
    
    
    # p.model.connect('traj.gravity_turn.timeseries.q_dyn', 'constraintsDynamicPressure.max_q_dyn_1_t')
    p.model.connect('traj.gravity_turn.timeseries.q_dyn', 'constraintsDynamicPressure.max_q_dyn_1_t', src_indices=[-1])
    
    
    
    
    
    
    
    
    p.model.connect('external_params.D_stage_1', 'constraintsExitArea.D_stage_1_m')
    p.model.connect('external_params.D_stage_1', 'constraintsExitArea.D_stage_2_m')
    
    p.model.connect('propulsion.propulsion_stage_1.Ae_t', 'constraintsExitArea.Ae_t_1')
    p.model.connect('propulsion.propulsion_stage_2.Ae_t', 'constraintsExitArea.Ae_t_2')
    
    return p