# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:12:36 2020

@author: jorge
"""

import openmdao.api as om
import numpy as np

import random


def importInitialGuess(p, phases, design_params, initial_guess, states, traj_phase_duration):
    
    " this function set as initial guess the values read from a dymos .db file"
    # set initial guess for design parameters
    
    ## ================= freezed variables ==============================
    freezed_var = ['P_c_stage_2', 'P_e_stage_2', 'o_f_stage_2',
                   'P_c_stage_1', 'P_e_stage_1', 'o_f_stage_1', 'D_stage_1']
    
    freezed_var = ['external_params.' + s for s in freezed_var]
    
    for param in freezed_var:
         p[param] = initial_guess.get_val(param)
    ## ===================================================================
    
    for param in design_params.keys():
        if param!='external_params.thrust_vac_stage_1_nominal' and param!='external_params.thrust_vac_stage_2_nominal':
            p[param] = initial_guess.get_val(param)
                
    p['external_params.thrust_vac_stage_1_nominal'] = initial_guess.get_val('external_params.thrust_vac_stage_1')
    p['external_params.thrust_vac_stage_2_nominal'] = initial_guess.get_val('external_params.thrust_vac_stage_2')
    
    # set initial guess for each phase and for each state
    for phase in phases:
        # the exoatmos phases have duration as a design parameter
        if 'exoatmos' not in phase:    
            p['traj.' + phase + '.t_duration'] = initial_guess.get_val('traj.' + phase + '.t_duration')
        p['traj.' + phase + '.t_initial']  = initial_guess.get_val('traj.' + phase + '.t_initial')
        for state in states:
            p['traj.' + phase + '.states:' + state] = initial_guess.get_val('traj.' + phase + '.states:' + state)
    return p


def readGuessFile (guess_file):
    "this function extracts the last iteration of a dymos.db file"
    # import the recorded cases

    cr = om.CaseReader(guess_file)
    
    
    # extract only the last iteration
    initial_guess = cr.get_cases()[-1]
    
    return initial_guess

def importInitialGuess_manualInput(p, phases, design_params, traj_phase_duration, states, randomize):
    "this function serves to define manually all of the initial values for the optimization."
    " if randomize == True the initial guess is set randomly for design parameters and time duration but manually for the states"
    
    if randomize == False:
        
        # parameters linked to massSizing and propulsion
        p['external_params.mp_1']               = 250951.47495719
        p['external_params.mp_2']               = 70174.74494154
        p['external_params.D_stage_1']          = 4.9973192
        p['external_params.thrust_vac_stage_1_nominal'] = 7639320.25181695
        p['external_params.thrust_vac_stage_2_nominal'] = 901591.67640696
        p['external_params.o_f_stage_1']        = 2.31333748
        p['external_params.o_f_stage_2']        = 2.35397704
        p['external_params.P_c_stage_1']        = 9999920.54073927
        p['external_params.P_c_stage_2']        = 9999987.96909449
        p['external_params.P_e_stage_2']        = 1733.98219325
        p['external_params.P_e_stage_1']        = 40546.157636
        p['external_params.max_n_f_1']          = 6.62949568
        p['external_params.max_q_dyn_1']        = 86566.288584
    
        # parameters for pitch angle guidance
        p['traj.design_parameters:xi']                      = -0.1997                
        p['traj.design_parameters:delta_theta_pitch_over']  = 0.0544            
        p['traj.design_parameters:delta_theta_exoatmos']    = -0.1734         
        p['traj.design_parameters:theta_f']                 = 0.0555  
        
        # duration of the phases        
        p['traj.lift_off.t_duration']                       = 10       
        p['traj.pitch_over_linear.t_duration']              = 10        
        p['traj.pitch_over_exponential.t_duration']         = 10           
        p['traj.gravity_turn.t_duration']                   = 40        
        p['traj.gravity_turn_b.t_duration']                 = 70
        p['traj.gravity_turn_c.t_duration']                 = 10
        p['traj.design_parameters:exoatmos_a_t_duration']   = 50         
        p['traj.design_parameters:exoatmos_b_t_duration']   = 230              
        
        # intial time of the phases
        p['traj.lift_off.t_initial']                        = 0.0
        p['traj.pitch_over_linear.t_initial']               = 10 
        p['traj.pitch_over_exponential.t_initial']          = 20
        p['traj.gravity_turn.t_initial']                    = 30
        p['traj.gravity_turn_b.t_initial']                  = 70
        p['traj.gravity_turn_c.t_initial']                  = 140
        p['traj.exoatmos_a.t_initial']                      = 150
        p['traj.exoatmos_b.t_initial']                      = 200
        
    elif randomize == True:
    
        # set random initialization for design parameters
        # ======================================================================================================
        # delete phase duration of exoatmos from design_params
        for key, values in traj_phase_duration.items():
            if '_t_duration' in key:
                design_params.pop(key)
        
        # set random initialization for design parameters
        for key, value in design_params.items():
            
            # define bounds for random number
            lower, upper = findBounds(value)
            
            # generate random number and set initial guess
            n = random.uniform(lower, upper)
            p[key] = n
            # print('lower: ' + str(np.round(lower,2)).ljust(15) + 'n: ' + str(np.round(n,2)).ljust(15) + 'upper: ' + str(np.round(upper,2)) )
            # print(key)
        
    
        # set random initializatio for time duration variables
        # ======================================================================================================
        # the time is defined as continous. One phase starts at the time the previous one finishes
        # traj_phase_duration must be in the right order of the phases. 
        time = 0.0
        
        for key, value in traj_phase_duration.items():
            
            # define bounds for random number
            lower, upper = findBounds(value)
            
            # generate random number and set initial guess for phase duration
            n = random.uniform(lower, upper)
            p[key] = n
            
            # clean the name of the phase
            phase = key.replace('traj.phases.','')
            phase = phase.replace('.time_extents.t_duration','')
            phase = phase.replace('traj.design_params.design_parameters:','')
            phase = phase.replace('_t_duration','')
            # set initial guess for initial time of the phases
            auxstr = 'traj.' + phase + '.t_initial'
            p[auxstr] = time
            
            # print('lower: ' + str(np.round(lower,2)).ljust(15) + 'n: ' + str(np.round(n,2)).ljust(15) + 'upper: ' + str(np.round(upper,2)).ljust(15) + 't_init:' + str(np.round(time,2)) ) 
            # print(key)
            # print(auxstr)
            # time increment
            time += n
    
    # parameters for the initial guess of each state discretization node at each state for each phase.
    # v, phi, lambda and r of the lift-off phase have fixed initial values, 
    # their inputs in this code will determine their value in the converged solution for the very first node.
    
    r_e = 6378135.0
    p['traj.lift_off.states:r']               = phases['lift_off'].interpolate(              ys=[r_e + 0.00000, r_e + 300.000], nodes='state_input')
    p['traj.pitch_over_linear.states:r']      = phases['pitch_over_linear'].interpolate(     ys=[r_e + 300.000, r_e + 1000.00], nodes='state_input')
    p['traj.pitch_over_exponential.states:r'] = phases['pitch_over_exponential'].interpolate(ys=[r_e + 1000.00, r_e + 2000.00], nodes='state_input')
    p['traj.gravity_turn.states:r']           = phases['gravity_turn'].interpolate(          ys=[r_e + 2000.00, r_e + 30000.0], nodes='state_input')
    p['traj.gravity_turn_b.states:r']         = phases['gravity_turn_b'].interpolate(        ys=[r_e + 30000.0, r_e + 60000.0], nodes='state_input')
    p['traj.gravity_turn_c.states:r']         = phases['gravity_turn_c'].interpolate(        ys=[r_e + 60000.0, r_e + 61000.0], nodes='state_input')
    p['traj.exoatmos_a.states:r']             = phases['exoatmos_a'].interpolate(            ys=[r_e + 61000.0, r_e + 90000.0], nodes='state_input')
    p['traj.exoatmos_b.states:r']             = phases['exoatmos_b'].interpolate(            ys=[r_e + 90000.0, r_e + 160000.], nodes='state_input')
    
    p['traj.lift_off.states:v']               = phases['lift_off'].interpolate(              ys=[ 1e-3, 6e1], nodes='state_input')
    p['traj.pitch_over_linear.states:v']      = phases['pitch_over_linear'].interpolate(     ys=[  6e1, 1e2], nodes='state_input')
    p['traj.pitch_over_exponential.states:v'] = phases['pitch_over_exponential'].interpolate(ys=[  1e2, 2e2], nodes='state_input')
    p['traj.gravity_turn.states:v']           = phases['gravity_turn'].interpolate(          ys=[  2e2, 2e3], nodes='state_input')
    p['traj.gravity_turn_b.states:v']         = phases['gravity_turn_b'].interpolate(        ys=[  2e3, 3e3], nodes='state_input')
    p['traj.gravity_turn_c.states:v']         = phases['gravity_turn_c'].interpolate(        ys=[  3e3, 3.1e3], nodes='state_input')
    p['traj.exoatmos_a.states:v']             = phases['exoatmos_a'].interpolate(            ys=[3.1e3, 5e3], nodes='state_input')
    p['traj.exoatmos_b.states:v']             = phases['exoatmos_b'].interpolate(            ys=[  5e3, 8e3], nodes='state_input')
    
    p['traj.lift_off.states:lambda']               = phases['lift_off'].interpolate(              ys=[  0.0, 1e-8], nodes='state_input')
    p['traj.pitch_over_linear.states:lambda']      = phases['pitch_over_linear'].interpolate(     ys=[  1e-8, 5e-8], nodes='state_input')
    p['traj.pitch_over_exponential.states:lambda'] = phases['pitch_over_exponential'].interpolate(ys=[  5e-8, 1e-7], nodes='state_input')
    p['traj.gravity_turn.states:lambda']           = phases['gravity_turn'].interpolate(          ys=[  1e-7, 1e-5], nodes='state_input')
    p['traj.gravity_turn_b.states:lambda']         = phases['gravity_turn_b'].interpolate(        ys=[  1e-5, 1e-4], nodes='state_input')
    p['traj.gravity_turn_c.states:lambda']         = phases['gravity_turn_c'].interpolate(        ys=[  1e-4, 1e-3], nodes='state_input')
    p['traj.exoatmos_a.states:lambda']             = phases['exoatmos_a'].interpolate(            ys=[  1e-3, 1e-2], nodes='state_input')
    p['traj.exoatmos_b.states:lambda']             = phases['exoatmos_b'].interpolate(            ys=[  1e-2, 2e-1], nodes='state_input')
    
    p['traj.lift_off.states:phi']               = phases['lift_off'].interpolate(              ys=[ np.radians(90), np.radians(90)], nodes='state_input')
    p['traj.pitch_over_linear.states:phi']      = phases['pitch_over_linear'].interpolate(     ys=[ np.radians(90), np.radians(88)], nodes='state_input')
    p['traj.pitch_over_exponential.states:phi'] = phases['pitch_over_exponential'].interpolate(ys=[ np.radians(88), np.radians(78)], nodes='state_input')
    p['traj.gravity_turn.states:phi']           = phases['gravity_turn'].interpolate(          ys=[ np.radians(78), np.radians(60)], nodes='state_input')
    p['traj.gravity_turn_b.states:phi']         = phases['gravity_turn_b'].interpolate(        ys=[ np.radians(60), np.radians(30)], nodes='state_input')
    p['traj.gravity_turn_c.states:phi']         = phases['gravity_turn_c'].interpolate(        ys=[ np.radians(30), np.radians(29)], nodes='state_input')
    p['traj.exoatmos_a.states:phi']             = phases['exoatmos_a'].interpolate(            ys=[ np.radians(29), np.radians(20)], nodes='state_input')
    p['traj.exoatmos_b.states:phi']             = phases['exoatmos_b'].interpolate(            ys=[ np.radians(20), np.radians(0)], nodes='state_input')
    
    p['traj.lift_off.states:m']               = phases['lift_off'].interpolate(              ys=[ 400e3, 380e3], nodes='state_input')
    p['traj.pitch_over_linear.states:m']      = phases['pitch_over_linear'].interpolate(     ys=[ 380e3, 360e3], nodes='state_input')
    p['traj.pitch_over_exponential.states:m'] = phases['pitch_over_exponential'].interpolate(ys=[ 360e3, 330e3], nodes='state_input')
    p['traj.gravity_turn.states:m']           = phases['gravity_turn'].interpolate(          ys=[ 330e3, 270e3], nodes='state_input')
    p['traj.gravity_turn_b.states:m']         = phases['gravity_turn_b'].interpolate(        ys=[ 270e3, 180e3], nodes='state_input')
    p['traj.gravity_turn_c.states:m']         = phases['gravity_turn_c'].interpolate(        ys=[ 180e3, 175e3], nodes='state_input')
    p['traj.exoatmos_a.states:m']             = phases['exoatmos_a'].interpolate(            ys=[ 150e3, 120e3], nodes='state_input')
    p['traj.exoatmos_b.states:m']             = phases['exoatmos_b'].interpolate(            ys=[ 118e3, 20e3], nodes='state_input')
    
    return p




def importInitialGuess_random(p, phases, design_params, traj_phase_duration, states):
    "This function sets ranmdomly the design parameters and phase durations in the domain defined by their bounds."
    " The states take a constant ramdon value in the whole span of each phase"
    
    # set random initialization for design parameters
    # ======================================================================================================
    # delete phase duration of exoatmos from design_params
    for key, values in traj_phase_duration.items():
        if '_t_duration' in key:
            design_params.pop(key)
    
    # set random initialization for design parameters
    for key, value in design_params.items():
        
        # define bounds for random number
        lower, upper = findBounds(value)
        
        # generate random number and set initial guess
        n = random.uniform(lower, upper)
        p[key] = n
        print('lower: ' + str(np.round(lower,2)).ljust(15) + 'n: ' + str(np.round(n,2)).ljust(15) + 'upper: ' + str(np.round(upper,2)) ) 
        
    # set random initializatio for time variables
    # ======================================================================================================
    # the time is defined as continous. One phase starts at the time the previous one finishes
    # traj_phase_duration must be in the right order of the phases
    time = 0.0
    
    for key, value in traj_phase_duration.items():
        
        # define bounds for random number
        lower, upper = findBounds(value)
        
        # generate random number and set initial guess for phase duration
        n = random.uniform(lower, upper)
        p[key] = n
        
        # clean the name of the phase
        phase = key.replace('traj.phases.','')
        phase = phase.replace('.time_extents.t_duration','')
        phase = phase.replace('traj.design_params.design_parameters:','')
        phase = phase.replace('_t_duration','')
        # set initial guess for initial time of the phases
        auxstr = 'traj.' + phase + '.t_initial'
        p[auxstr] = time
        
        print('lower: ' + str(np.round(lower,2)).ljust(15) + 'n: ' + str(np.round(n,2)).ljust(15) + 'upper: ' + str(np.round(upper,2)).ljust(15) + 't_init:' + str(np.round(time,2)) ) 
        # time increment
        time += n
        
    # set random initialization for states
    # =======================================================================================================
    
    # intial values for lift_off
    lift_off_initial = {'r':6378135.0,
                        'v':1e-3,
                        'lambda':0.0,
                        'm':random.uniform(states['m'][0], states['m'][1]),
                        'phi':np.pi/2}
                        
    for key_state, value_state in states.items():
        for key_phase, value_phase in phases.items():
            
            # set random initial and final value for each state at each phase 
            if key_phase == 'lift_off':
                initial = lift_off_initial[key_state]
            final   = random.uniform(value_state[0], value_state[1])
            
            # interpolate
            auxstr    = 'traj.' + key_phase + '.states:' + key_state
            p[auxstr] = value_phase.interpolate(  ys=[ initial, initial], nodes='state_input')
            print(len(value_phase.interpolate(  ys=[ initial, initial], nodes='state_input')))
            
            print(key_phase.ljust(25) + key_state.ljust(10) + 'initial: ' + str(np.round(initial,2)).ljust(15)  + 'final: ' + str(np.round(final,2)) ) 
            # update initial. This allows to have a continuous initial guess
            initial = final
        
    return p
    
def findBounds(dictValue):
    #  this functions finds the upper and lower bounds that were set for a variable
    adder = dictValue['adder']
    if adder == None:
        adder = 0.0
    lower = dictValue['lower'] / dictValue['scaler'] - adder
    upper = dictValue['upper'] / dictValue['scaler'] - adder
    
    return lower, upper
