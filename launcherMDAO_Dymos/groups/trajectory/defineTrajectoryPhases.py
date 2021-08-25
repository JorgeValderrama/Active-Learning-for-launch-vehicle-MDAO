# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:10:25 2021

@author: jorge
"""
import dymos as dm
import numpy as np

from .launch_vehicle_ode import LaunchVehicleODE_lift_off, LaunchVehicleODE_pitch_over_linear, LaunchVehicleODE_pitch_over_exponential
from .launch_vehicle_ode import LaunchVehicleODE_gravity_turn, LaunchVehicleODE_exoatmos_a, LaunchVehicleODE_exoatmos_b

def defineTrajectoryPhases(central_body, vehicle, ha, hp_min):
    
    # Initialize Trajectory 
    traj = dm.Trajectory()  
    
    # Set to True if throttle is to be considered as dynamic parameter
    opt_throttle = False
    
    # Define phases:
    # ========================================================================================================
    
    lift_off               = dm.Phase(ode_class = LaunchVehicleODE_lift_off, 
                                      transcription = dm.GaussLobatto(num_segments=7, order=3, compressed=False),
                                      ode_init_kwargs={'central_body':central_body,'uncertainty_cd':vehicle.u_cd }) 
    
    pitch_over_linear      = dm.Phase(ode_class = LaunchVehicleODE_pitch_over_linear,
                                      transcription = dm.GaussLobatto(num_segments=7, order=3, compressed=False),
                                      ode_init_kwargs={'central_body':central_body,'uncertainty_cd':vehicle.u_cd })
    
    pitch_over_exponential = dm.Phase(ode_class = LaunchVehicleODE_pitch_over_exponential,
                                      transcription = dm.GaussLobatto(num_segments=7, order=3, compressed=False),
                                      ode_init_kwargs={'central_body':central_body,'uncertainty_cd':vehicle.u_cd })
    
    gravity_turn           = dm.Phase(ode_class=LaunchVehicleODE_gravity_turn,
                                      transcription = dm.GaussLobatto(num_segments=4, order=3, compressed=False),
                                      ode_init_kwargs={'central_body':central_body,'uncertainty_cd':vehicle.u_cd })
    
    gravity_turn_b         = dm.Phase(ode_class=LaunchVehicleODE_gravity_turn,
                                      transcription = dm.GaussLobatto(num_segments=3, order=3, compressed=False),
                                      ode_init_kwargs={'central_body':central_body,'uncertainty_cd':vehicle.u_cd })
    
    gravity_turn_c         = dm.Phase(ode_class=LaunchVehicleODE_gravity_turn,
                                      transcription = dm.GaussLobatto(num_segments=7, order=3, compressed=False),
                                      ode_init_kwargs={'central_body':central_body,'uncertainty_cd':vehicle.u_cd })
    
    exoatmos_a             = dm.Phase(ode_class = LaunchVehicleODE_exoatmos_a,
                                      transcription = dm.GaussLobatto(num_segments=7, order=3, compressed=False),
                                      ode_init_kwargs={'central_body':central_body,'uncertainty_cd':vehicle.u_cd })
    
    exoatmos_b             = dm.Phase(ode_class=LaunchVehicleODE_exoatmos_b,
                                      transcription = dm.GaussLobatto(num_segments=14, order=3, compressed=False),
                                      ode_init_kwargs={'central_body':central_body,'uncertainty_cd':vehicle.u_cd })
    
    
    # define design paramaters
    # ==========================================================================================================
    
    traj.add_design_parameter('delta_theta_pitch_over', units = 'rad', opt=True,
                              lower = np.radians(1), upper = np.radians(10),
                              ref= np.radians(10), ref0 = np.radians(1),
                              targets = {'lift_off':None,
                                         'pitch_over_linear':['guidance.delta_theta'],
                                         'pitch_over_exponential':['guidance.delta_theta'],
                                         'gravity_turn':None,
                                         'gravity_turn_b':None,
                                         'gravity_turn_c':None,
                                         'exoatmos_a':None,
                                         'exoatmos_b':None}) 
    
    traj.add_design_parameter('xi', units = None , opt=True,
                              lower = -1 , upper = 1 ,
                              ref=1.0, ref0=-1.0,
                              targets = {'lift_off':None,
                                         'pitch_over_linear':None ,
                                         'pitch_over_exponential':None,
                                         'gravity_turn':None ,
                                         'gravity_turn_b':None ,
                                         'gravity_turn_c':None,
                                         'exoatmos_a':['guidance.xi'],
                                         'exoatmos_b':['guidance.xi']})
    
    # upper limit should be 60 deg according to Castellini, keep in mind the domain of the tangent function
    # in the bilinear tangent law definition it can lead to weird jumps in pitch angle.
    traj.add_design_parameter('delta_theta_exoatmos', units='rad', opt=True,
                              lower = np.radians(-60) , upper = np.radians(60),
                              ref=1.05E0, ref0=-1.05E0,
                              targets = {'lift_off':None,
                                         'pitch_over_linear':None ,
                                         'pitch_over_exponential':None,
                                         'gravity_turn':None ,
                                         'gravity_turn_b':None ,
                                         'gravity_turn_c':None,
                                         'exoatmos_a':['guidance.delta_theta'],
                                         'exoatmos_b':['guidance.delta_theta']})
    
    traj.add_design_parameter('theta_f', units = 'rad',  opt=True,
                              lower = np.radians(-20), upper = np.radians(20),
                              ref= np.radians(20), ref0= np.radians(-20),
                              targets = {'lift_off':None,
                                         'pitch_over_linear':None ,
                                         'pitch_over_exponential':None,
                                         'gravity_turn':None ,
                                         'gravity_turn_b':None ,
                                         'gravity_turn_c':None,
                                         'exoatmos_a':['guidance.theta_f'],
                                         'exoatmos_b':['guidance.theta_f']})
    
    # The duration of the exoatmos phases has a different treatment as their values must be known a priori
    # in order to normalize time for the bilinear tangent law.
    traj.add_design_parameter('exoatmos_a_t_duration', units = 's', 
                              opt = True, dynamic=False,
                              lower = 1 , upper = 250,
                              ref=250.0, ref0=1.0,
                              targets={'lift_off':None,
                                      'pitch_over_linear':None,
                                      'pitch_over_exponential':None,
                                      'gravity_turn':None,
                                      'gravity_turn_b':None,
                                      'gravity_turn_c':None,
                                      'exoatmos_a':['time_exoatmos_a.phase_duration_a'],
                                      'exoatmos_b':['time_exoatmos_b.phase_duration_a']})
    
    traj.add_design_parameter('exoatmos_b_t_duration', units = 's',
                              opt = True, dynamic = False,
                              lower = 1 , upper = 250,
                              ref=250.0, ref0=1.0,
                              targets={'lift_off':None,
                                      'pitch_over_linear':None,
                                      'pitch_over_exponential':None,
                                      'gravity_turn':None,
                                      'gravity_turn_b':None,
                                      'gravity_turn_c':None,
                                      'exoatmos_a':['time_exoatmos_a.phase_duration_b'],
                                      'exoatmos_b':['time_exoatmos_b.phase_duration_b']})
    
    # traj.add_design_parameter('exoatmos_c_t_duration', units = 's',
    #                           opt = True, dynamic = False,
    #                           lower = 1 , upper = 500,
    #                           ref=130.0, ref0=1.0,
    #                           targets={'lift_off':None,
    #                                   'pitch_over_linear':None,
    #                                   'pitch_over_exponential':None,
    #                                   'gravity_turn':None,
    #                                   'gravity_turn_b':None,
    #                                   'gravity_turn_c':None,
    #                                   'exoatmos_a':['time_exoatmos_a.phase_duration_c'],
    #                                   'exoatmos_b':['time_exoatmos_b.phase_duration_c']})
    
    # define input paramaters
    # ==========================================================================================================
    
    # during lift_off the pitch angle is 90 deg
    lift_off.add_input_parameter('theta', units = 'rad',
                                 val = np.radians(90), 
                                 targets = ['eom.theta'])
    
    # The initial pitch angle (theta) during the bilinear tangent law is a function of delta_theta_exoatmos and
    # the final theta of th egravity turn phase (theta_gt)
    exoatmos_a.add_input_parameter('theta_gt', units='rad', dynamic=False,
                                   targets=['guidance.theta_gt'] ) 
 
    exoatmos_b.add_input_parameter('theta_gt', units='rad', dynamic=False,
                                   targets=['guidance.theta_gt'])  
    
    # exoatmos_c.add_input_parameter('theta_gt', units='rad', dynamic=False,
    #                                targets=['guidance.theta_gt'])
    
    # input parameters asociated to propulsion second stage
    traj.add_input_parameter('thrust_vac_stage_2', units = 'N',
                         dynamic = False,
                         targets={'lift_off':None,
                                  'pitch_over_linear':None,
                                  'pitch_over_exponential':None,
                                  'gravity_turn':None,
                                  'gravity_turn_b':None,
                                  'gravity_turn_c':['thrust_losses.thrust_vac'],
                                  'exoatmos_a':['thrust_losses.thrust_vac'],
                                  'exoatmos_b':['thrust_losses.thrust_vac']})

    traj.add_input_parameter('mfr_max_stage_2', units = 'kg/s',
                             dynamic = False,
                             targets={'lift_off':None,
                                      'pitch_over_linear':None,
                                      'pitch_over_exponential':None,
                                      'gravity_turn':None,
                                      'gravity_turn_b':None,
                                      'gravity_turn_c':['thrust_losses.mfr_max'],
                                      'exoatmos_a':['thrust_losses.mfr_max'],
                                      'exoatmos_b':['thrust_losses.mfr_max']})
    
    traj.add_input_parameter('Ae_t_stage_2', units = 'm**2',
                             dynamic = False,
                             targets={'lift_off':None,
                                      'pitch_over_linear':None,
                                      'pitch_over_exponential':None,
                                      'gravity_turn':None,
                                      'gravity_turn_b':None,
                                      'gravity_turn_c':['thrust_losses.Ae_t'],
                                      'exoatmos_a':['thrust_losses.Ae_t'],
                                      'exoatmos_b':['thrust_losses.Ae_t']})
    
    traj.add_input_parameter('Isp_stage_2', units = 's',
                             dynamic = False,
                             targets={'lift_off':None,
                                      'pitch_over_linear':None,
                                      'pitch_over_exponential':None,
                                      'gravity_turn':None,
                                      'gravity_turn_b':None,
                                      'gravity_turn_c':None,
                                      'exoatmos_a':None,
                                      'exoatmos_b':['orbitalParameters.Isp']})
    
    # input parameters asociated to propulsion first stage
    traj.add_input_parameter('thrust_vac_stage_1', units = 'N',
                             dynamic = False, shape =(1,),
                             targets={'lift_off':['thrust_losses.thrust_vac'],
                                      'pitch_over_linear':['thrust_losses.thrust_vac'],
                                      'pitch_over_exponential':['thrust_losses.thrust_vac'],
                                      'gravity_turn':['thrust_losses.thrust_vac'],
                                      'gravity_turn_b':['thrust_losses.thrust_vac'],
                                      'gravity_turn_c':None,
                                      'exoatmos_a':None,
                                      'exoatmos_b':None})
    
    traj.add_input_parameter('mfr_max_stage_1', units = 'kg/s',
                             dynamic = False,
                             targets={'lift_off':['thrust_losses.mfr_max'],
                                      'pitch_over_linear':['thrust_losses.mfr_max'],
                                      'pitch_over_exponential':['thrust_losses.mfr_max'],
                                      'gravity_turn':['thrust_losses.mfr_max'],
                                      'gravity_turn_b':['thrust_losses.mfr_max'],
                                      'gravity_turn_c':None,
                                      'exoatmos_a':None,
                                      'exoatmos_b':None})
    
    traj.add_input_parameter('Ae_t_stage_1', units = 'm**2',
                             dynamic = False,
                             targets={'lift_off':['thrust_losses.Ae_t'],
                                      'pitch_over_linear':['thrust_losses.Ae_t'],
                                      'pitch_over_exponential':['thrust_losses.Ae_t'],
                                      'gravity_turn':['thrust_losses.Ae_t'],
                                      'gravity_turn_b':['thrust_losses.Ae_t'],
                                      'gravity_turn_c':None,
                                      'exoatmos_a':None,
                                      'exoatmos_b':None})
    
    traj.add_input_parameter('D_stage_1', units = 'm',
                             dynamic = False,
                             targets={'lift_off':['aero.diameter'],
                                      'pitch_over_linear':['aero.diameter'],
                                      'pitch_over_exponential':['aero.diameter'],
                                      'gravity_turn':['aero.diameter'],
                                      'gravity_turn_b':['aero.diameter'],
                                      'gravity_turn_c':['aero.diameter'],
                                      'exoatmos_a':['aero.diameter'],
                                      'exoatmos_b':['aero.diameter']})
    
    # add phases to trajectory
    # ==========================================================================================================
    traj.add_phase('lift_off', lift_off)
    traj.add_phase('pitch_over_linear', pitch_over_linear)
    traj.add_phase('pitch_over_exponential', pitch_over_exponential)
    traj.add_phase('gravity_turn', gravity_turn)
    traj.add_phase('gravity_turn_b', gravity_turn_b)
    traj.add_phase('gravity_turn_c', gravity_turn_c)
    traj.add_phase('exoatmos_a', exoatmos_a)
    traj.add_phase('exoatmos_b', exoatmos_b)
    # traj.add_phase('exoatmos_c', exoatmos_c)
    
    
    # set time options and add states and controls:
    # lift_off
    # ===========================================================================================
    # in the lift_off phase the initial values of r, lambda, v, and phi are fixed as 
    # they are the known initial boundary conditions.
    
    lift_off.set_time_options(units = 's', 
                              fix_initial = True, 
                              duration_bounds=(1, 100), 
                              duration_ref=100)
    
    lift_off.add_state('r', units='m', rate_source='eom.rdot',
                       fix_initial=True, fix_final = False,   
                       targets=['eom.r','aero.r','gravity.r'], 
                       ref=6.38E6, ref0=6.37E6, defect_ref=1)
    
    lift_off.add_state('lambda', units='rad', rate_source='eom.lambdadot', 
                       fix_initial=True, fix_final = False,
                       targets=['eom.lambda'], 
                       ref=2.0E-8, ref0=0.0, defect_ref=1)
    
    lift_off.add_state('v', units='m/s', rate_source='eom.vdot',
                       fix_initial=True, fix_final = False,
                       targets=['eom.v','aero.v'], 
                       ref=1E1, ref0=0.0, defect_ref=1)
    
    lift_off.add_state('phi', units='rad', rate_source='eom.phidot',
                       fix_initial=True, fix_final = False,
                       targets=['eom.phi'],  
                       ref=1.57E0, ref0=0.0, defect_ref=1)
    
    lift_off.add_state('m', units='kg', rate_source='eom.mdot',
                       fix_initial=False, fix_final = False, 
                       targets=['eom.m'],  
                       ref=4.8E5, ref0 = 4E5, defect_ref=1)
    
    lift_off.add_control('throttle', units=None, opt=opt_throttle, 
                         targets= 'thrust_losses.throttle',
                         lower = 0.7, upper = 1.0, continuity = True , val=1.0) 
    
    # set time options and add states and controls:
    # pitch over linear
    # ===========================================================================================
    
    pitch_over_linear.set_time_options(units = 's',  
                                       time_phase_targets=['guidance.phase_time'],
                                       t_duration_targets=['guidance.phase_duration'],
                                       duration_bounds=(5, 10),
                                       duration_ref= 10,
                                       initial_ref=2e1)
    
    pitch_over_linear.add_state('r', units='m', rate_source='eom.rdot',    
                                fix_initial = False, fix_final = False , 
                                targets=['eom.r','aero.r','gravity.r'], 
                                ref=6.38E6, ref0=6.37E6, defect_ref=1)
    
    pitch_over_linear.add_state('lambda', units='rad', rate_source='eom.lambdadot',
                                fix_initial = False, fix_final = False , 
                                targets=['eom.lambda'], 
                                ref=2.0E-7, ref0=0.0, defect_ref=1)
    
    pitch_over_linear.add_state('v', units='m/s', rate_source='eom.vdot',    
                                fix_initial = False, fix_final = False , 
                                targets=['eom.v','aero.v'], 
                                ref=1E2, ref0=0.0, defect_ref=1)
    
    pitch_over_linear.add_state('phi', units='rad', rate_source='eom.phidot',  
                                fix_initial = False, fix_final = False, 
                                targets=['eom.phi','guidance.phi'], 
                                ref=1.57E0, ref0=0.0, defect_ref=1)
    
    pitch_over_linear.add_state('m', units='kg', rate_source='eom.mdot',    
                                fix_initial = False, fix_final = False , 
                                targets=['eom.m'], 
                                ref=4.2E5, ref0 = 3E5, defect_ref=1)
    
    pitch_over_linear.add_control('throttle', units=None, opt=opt_throttle, 
                                  targets= 'thrust_losses.throttle',
                                  lower = 0.7, upper = 1.0, continuity = True , val=1.0)
    
    # set time options and add states and controls:
    # pitch over exponential
    # ===========================================================================================
    
    pitch_over_exponential.set_time_options(units = 's', 
                                            time_phase_targets=['guidance.phase_time'],
                                            t_duration_targets=['guidance.phase_duration'],
                                            duration_bounds=(1, 100),
                                            duration_ref=100,
                                            initial_ref=2e2)
    
    pitch_over_exponential.add_state('r', units='m', rate_source='eom.rdot',   
                                     fix_initial = False, fix_final = False, 
                                     targets=['eom.r','aero.r','gravity.r'], 
                                     ref=6.41E6, ref0=6.37E6, defect_ref=1)
    
    pitch_over_exponential.add_state('lambda', units='rad', rate_source='eom.lambdadot',
                                     fix_initial = False, fix_final = False, 
                                     targets=['eom.lambda'], 
                                     ref=2.0E-5, ref0=0.0, defect_ref=1)
    
    pitch_over_exponential.add_state('v', units='m/s', rate_source='eom.vdot',    
                                     fix_initial = False, fix_final = False , 
                                     targets=['eom.v','aero.v'], 
                                     ref=1E3, ref0=0.0, defect_ref=1)
    
    pitch_over_exponential.add_state('phi', units='rad', rate_source='eom.phidot',  
                                     fix_initial = False, fix_final = False, 
                                     targets=['eom.phi','guidance.phi'], 
                                     ref=1.57E0, ref0=0.0, defect_ref=1)
    
    pitch_over_exponential.add_state('m', units='kg', rate_source='eom.mdot',    
                                     fix_initial = False, fix_final = False , 
                                     targets=['eom.m'], 
                                     ref=3.0E5, ref0 = 2E5, defect_ref=1)
    
    pitch_over_exponential.add_control('throttle', units=None, opt=opt_throttle, 
                                       targets= 'thrust_losses.throttle',
                                       lower = 0.7, upper = 1.0, continuity = True , val=1.0)
    
    # set time options and add states and controls:
    # gravity turn 
    # ===========================================================================================
    
    gravity_turn.set_time_options(units = 's', 
                                  duration_bounds=(1, 150), 
                                  duration_ref=150,
                                  initial_ref=2e2)
    
    gravity_turn.add_state('r', units='m', rate_source='eom.rdot',    
                           fix_initial = False, fix_final = False, 
                           targets=['eom.r','aero.r','gravity.r'], 
                           ref=6.43E6, ref0=6.37E6, defect_ref=1)
    
    gravity_turn.add_state('lambda',  units='rad', rate_source='eom.lambdadot',
                           fix_initial = False, fix_final = False, 
                           targets=['eom.lambda'],
                           ref=2.0E-2, ref0=0.0, defect_ref=1)
    
    gravity_turn.add_state('v', units='m/s', rate_source='eom.vdot',    
                           fix_initial = False, fix_final = False, 
                           targets=['eom.v','aero.v', 'qDot.v'], 
                           ref=2E3, ref0=0.0, defect_ref=1)
    
    gravity_turn.add_state('phi', units='rad', rate_source='eom.phidot',   
                           fix_initial = False, fix_final = False,
                           targets=['eom.phi','guidance.phi'], 
                           ref=1.57E0, ref0=0.0, defect_ref=1)
    
    gravity_turn.add_state('m', units='kg', rate_source='eom.mdot',    
                           fix_initial = False, fix_final = False, 
                           targets=['eom.m'], 
                           ref=2.3E5, ref0 = 1E5, defect_ref=1)
    
    gravity_turn.add_control('throttle', units=None, opt=opt_throttle, 
                             targets= 'thrust_losses.throttle',
                             lower = 0.7, upper = 1.0, continuity = True , val=1.0)
    
    # set time options and add states and controls:
    # gravity turn b
    # ===========================================================================================  
    
    gravity_turn_b.set_time_options(units = 's', 
                                    duration_bounds=(1, 150), 
                                    duration_ref=150,
                                    initial_ref=2e2)
    
    gravity_turn_b.add_state('r', units='m', rate_source='eom.rdot',    
                           fix_initial = False, fix_final = False, 
                           targets=['eom.r','aero.r','gravity.r'], 
                           ref=6.43E6, ref0=6.37E6, defect_ref=1)
    
    gravity_turn_b.add_state('lambda',  units='rad', rate_source='eom.lambdadot',
                           fix_initial = False, fix_final = False, 
                           targets=['eom.lambda'],
                           ref=2.0E-2, ref0=0.0, defect_ref=1)
    
    gravity_turn_b.add_state('v', units='m/s', rate_source='eom.vdot',    
                           fix_initial = False, fix_final = False, 
                           targets=['eom.v','aero.v', 'qDot.v'], 
                           ref=2E3, ref0=0.0, defect_ref=1)
    
    gravity_turn_b.add_state('phi', units='rad', rate_source='eom.phidot',   
                           fix_initial = False, fix_final = False,
                           targets=['eom.phi','guidance.phi'], 
                           ref=1.57E0, ref0=0.0, defect_ref=1)
    
    gravity_turn_b.add_state('m', units='kg', rate_source='eom.mdot',    
                           fix_initial = False, fix_final = False, 
                           targets=['eom.m'], 
                           ref=2.3E5, ref0 = 1E5, defect_ref=1)
    
    gravity_turn_b.add_control('throttle', units=None, opt=opt_throttle, 
                              targets= 'thrust_losses.throttle',
                              lower = 0.7, upper = 1.0, continuity = True , val=1.0) 
    
    # set time options and add states and controls:
    # gravity turn b
    # ===========================================================================================  
    
    gravity_turn_c.set_time_options(units = 's', 
                                    duration_bounds=(1, 150), 
                                    duration_ref=150,
                                    initial_ref=2e2)
    
    gravity_turn_c.add_state('r', units='m', rate_source='eom.rdot',    
                           fix_initial = False, fix_final = False, 
                           targets=['eom.r','aero.r','gravity.r'], 
                           ref=6.43E6, ref0=6.37E6, defect_ref=1)
    
    gravity_turn_c.add_state('lambda',  units='rad', rate_source='eom.lambdadot',
                           fix_initial = False, fix_final = False, 
                           targets=['eom.lambda'],
                           ref=2.0E-2, ref0=0.0, defect_ref=1)
    
    gravity_turn_c.add_state('v', units='m/s', rate_source='eom.vdot',    
                           fix_initial = False, fix_final = False, 
                           targets=['eom.v','aero.v', 'qDot.v'], 
                           ref=2E3, ref0=0.0, defect_ref=1)
    
    gravity_turn_c.add_state('phi', units='rad', rate_source='eom.phidot',   
                           fix_initial = False, fix_final = False,
                           targets=['eom.phi','guidance.phi'], 
                           ref=1.57E0, ref0=0.0, defect_ref=1)
    
    gravity_turn_c.add_state('m', units='kg', rate_source='eom.mdot',    
                           fix_initial = False, fix_final = False, 
                           targets=['eom.m'], 
                           ref=2.3E5, ref0 = 1E5, defect_ref=1)
    
    gravity_turn_c.add_control('throttle', units=None, opt=opt_throttle, 
                              targets= 'thrust_losses.throttle',
                              lower = 0.7, upper = 1.0, continuity = True , val=1.0) 
    
    # set time options and add states and controls:
    # exoatmos_a
    # ===========================================================================================
    
    exoatmos_a.set_time_options(units = 's', 
                                input_duration=True,
                                time_phase_targets=['guidance.phase_time'],
                                initial_ref=2e2)
    
    exoatmos_a.add_state('r', units='m', rate_source='eom.rdot',    
                         fix_initial = False, fix_final = False , 
                         targets=['eom.r', 'aero.r', 'gravity.r'], 
                         ref=6.45E6, ref0=6.37E6, defect_ref=1)
    
    exoatmos_a.add_state('lambda', units='rad', rate_source='eom.lambdadot',
                         fix_initial = False, fix_final = False , 
                         targets=['eom.lambda'], 
                         ref=2.0E-2, ref0=0.0, defect_ref=1)
    
    exoatmos_a.add_state('v', units='m/s', rate_source='eom.vdot',    
                         fix_initial = False, fix_final = False , 
                         targets=['eom.v', 'aero.v'], 
                         ref=3E3, ref0=1E3, defect_ref=1)
    
    exoatmos_a.add_state('phi', units='rad',  rate_source='eom.phidot',  
                         fix_initial = False, fix_final = False ,
                         targets=['eom.phi'],
                         ref=1.57E0, ref0=0.0, defect_ref=1) 
    
    exoatmos_a.add_state('m', units='kg', rate_source='eom.mdot',
                         fix_initial = False, fix_final = False, 
                         targets=['eom.m'], 
                         ref=1.1E5, ref0 = 9E4, defect_ref=1)
    
    exoatmos_a.add_control('throttle', units=None, opt=opt_throttle, 
                           targets= 'thrust_losses.throttle',
                           lower = 0.7, upper = 1.0, continuity = True , val=1.0)
    
    # set time options and add states and controls:
    # exoatmos_b
    # ===========================================================================================
    
    exoatmos_b.set_time_options(units = 's', 
                                input_duration=True,
                                time_phase_targets=['time_exoatmos_b.phase_time_b'],
                                initial_ref=2e2)
    
    exoatmos_b.add_state('r', units='m', rate_source='eom.rdot',   
                         fix_initial = False, fix_final = False , 
                         targets=['eom.r', 'aero.r', 'gravity.r', 'orbitalParameters.r'], 
                         ref=6.45E6, ref0=6.37E6, defect_ref=1)
    
    exoatmos_b.add_state('lambda', units='rad', rate_source='eom.lambdadot',
                         fix_initial = False, fix_final = False , 
                         targets=['eom.lambda'], 
                         ref=2.0E-2, ref0=0.0, defect_ref=1)
    
    exoatmos_b.add_state('v', units='m/s', rate_source='eom.vdot',    
                         fix_initial = False, fix_final = False , 
                         targets=['eom.v', 'aero.v', 'orbitalParameters.v'], 
                         ref=4E3, ref0=1E3, defect_ref=1)
    
    exoatmos_b.add_state('phi', units='rad', rate_source='eom.phidot',  
                         fix_initial = False, fix_final = False, 
                         targets=['eom.phi', 'orbitalParameters.phi'],
                         ref=1.57E0, ref0=0.0, defect_ref=1) 
    
    exoatmos_b.add_state('m', units='kg', rate_source='eom.mdot',   
                         fix_initial = False, fix_final = False ,
                         targets=['eom.m', 'orbitalParameters.m_0'], 
                         ref=9E4, ref0 = 2.2E4, defect_ref=1)
    
    exoatmos_b.add_control('throttle', units=None, opt=opt_throttle, 
                           targets= 'thrust_losses.throttle',
                           lower = 0.7, upper = 1.0, continuity = True , val=1.0)
    
    # # set time options and add states and controls:
    # # exoatmos_c
    # # ===========================================================================================
    
    # exoatmos_c.set_time_options(units = 's', 
    #                             input_duration=True,
    #                             time_phase_targets=['time_exoatmos_c.phase_time_c'],
    #                             initial_ref=2e2)
    
    # exoatmos_c.add_state('r', units='m',  rate_source='eom.rdot',   
    #                      fix_initial = False, fix_final = False , 
    #                      targets=['eom.r', 'aero.r', 'gravity.r', 'orbitalParameters.r'], 
    #                      ref=6.55E6, ref0=6.37E6, defect_ref=1)
    
    # exoatmos_c.add_state('lambda', units='rad', rate_source='eom.lambdadot',
    #                      fix_initial = False, fix_final = False , 
    #                      targets=['eom.lambda'], 
    #                      ref=3.0E-2, ref0=0.0, defect_ref=1)
    
    # exoatmos_c.add_state('v', units='m/s', rate_source='eom.vdot',    
    #                      fix_initial = False, fix_final = False , 
    #                      targets=['eom.v', 'aero.v', 'orbitalParameters.v'],
    #                      ref=7E3, ref0=1E3, defect_ref=1)
    
    # exoatmos_c.add_state('phi', units='rad', rate_source='eom.phidot',  
    #                      fix_initial = False, fix_final = False , 
    #                      targets=['eom.phi', 'orbitalParameters.phi'],
    #                      ref=1.57E0, ref0=0.0, defect_ref=1) 
    
    # exoatmos_c.add_state('m', units='kg', rate_source='eom.mdot',    
    #                      fix_initial = False, fix_final = False , 
    #                      targets=['eom.m','orbitalParameters.m_0'], 
    #                      ref=9E4, ref0 = 2.2E4, defect_ref=1)
    
    # exoatmos_c.add_control('throttle', units=None, opt=opt_throttle, 
    #                        targets= 'thrust_losses.throttle',
    #                        val=1.0)
    
    # ==========================================================================================
    
    # store phases in dictionary 
    phases = {'lift_off':lift_off,
              'pitch_over_linear':pitch_over_linear,
              'pitch_over_exponential':pitch_over_exponential,
              'gravity_turn':gravity_turn,
              'gravity_turn_b':gravity_turn_b,
              'gravity_turn_c':gravity_turn_c,
              'exoatmos_a':exoatmos_a, 
              'exoatmos_b':exoatmos_b}
    
    # define the variables to be stored in Dymos timeseries. 
    # =========================================================================================
    for phase in phases.values():
        phase.add_timeseries_output('aero.rho',units='kg/m**3')
        phase.add_timeseries_output('aero.Mach')
        phase.add_timeseries_output('aero.Cd')
        phase.add_timeseries_output('aero.q_dyn',units= 'Pa')
        phase.add_timeseries_output('aero.q_heat',units = 'W/m**2')
        phase.add_timeseries_output('gravity.g', units='m/s**2')
        phase.add_timeseries_output('thrust_losses.thrust', units = 'N')
        phase.add_timeseries_output('eom.n_f', units = None)
        
        if phase != lift_off:
            phase.add_timeseries_output('guidance.theta',units='rad')
            
    exoatmos_b.add_timeseries_output('orbitalParameters.ra', units = 'm')
    exoatmos_b.add_timeseries_output('orbitalParameters.rp', units = 'm')
    exoatmos_b.add_timeseries_output('orbitalParameters.m_final', units = 'kg')
    
    gravity_turn.add_timeseries_output('qDot.qDot', units = 'kg/m/s**3') 
    gravity_turn_b.add_timeseries_output('qDot.qDot', units = 'kg/m/s**3') 
    # ===========================================================================================
    
    # link phases. 
    # As jetisson events are discontinuities in mass, the exoatmos phases have not continuity for m
    # Such discontinuity is handle via the constraints components
    # =============================================================================================
    traj.link_phases(['lift_off', 'pitch_over_linear'],                 vars=['*'])
    traj.link_phases(['pitch_over_linear','pitch_over_exponential'],    vars=['*'])
    traj.link_phases(['pitch_over_exponential', 'gravity_turn'],        vars=['*'])
    traj.link_phases(['gravity_turn', 'gravity_turn_b'],                vars=['*'])
    traj.link_phases(['gravity_turn_b', 'gravity_turn_c'],              vars=['r','lambda','v','phi','time'])
    traj.link_phases(['gravity_turn_c', 'exoatmos_a'],                  vars=['*'])
    traj.link_phases(['exoatmos_a', 'exoatmos_b'],                      vars=['r','lambda','v','phi','time'])
    
    # connections within the trajectory group
    # =============================================================================================
    
    # connection of duration of exoatmos phases
    traj.connect('design_parameters:exoatmos_a_t_duration',['exoatmos_a.t_duration'])
    traj.connect('design_parameters:exoatmos_b_t_duration',['exoatmos_b.t_duration'])
    # traj.connect('design_parameters:exoatmos_c_t_duration',['exoatmos_c.t_duration'])
    
    # Connect the last value of theta from gravity turn phase and use it for the input 'theta_gt' of 
    # the exoatmos phase
    traj.connect('gravity_turn_c.timeseries.theta', ['exoatmos_a.input_parameters:theta_gt',
                                                     'exoatmos_b.input_parameters:theta_gt'],src_indices=[-1])
    
    return traj, phases