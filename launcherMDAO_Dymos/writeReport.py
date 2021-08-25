# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:40:30 2020

@author: jorge
"""

import time
from numpy import ndarray
from math import isclose

def writeReport(Id, resultsPath, p, guess_file, vehicle, design_params_str, constraints_str, phase_duration_str, start_time, ha, hp_min, printToConsole):
    "This function writes the optimization report after the optimization is terminated"
    
    name_len = 40
    
    file1 = open(resultsPath + '/' + str(Id) + "_optReport.txt","w") 
    file1.write('============================================================================== \n')
    file1.write('============================ Optimization Report ============================= \n')
    file1.write('============================================================================== \n')
    file1.write('\n')
    file1.write('Design parameters: All values in standard SI units. Angles in radians. \n')
    file1.write('Design parameters marked with (***) are close to their bounds or violate them \n')
    file1.write('Name'.ljust(name_len) + 'lower'.ljust(15) + 'value'.ljust(15) + 'upper'.ljust(15) + '\n')
    file1.write('------------------------------------------------------------------------------- \n')
    
    for duration in phase_duration_str:
        aux = p.driver._designvars[duration]
        adder = aux['adder']
        if adder == None:
            adder = 0.0
        lower = aux['lower'] / aux['scaler'] - adder
        upper = aux['upper'] / aux['scaler'] - adder
        value = p.get_val(duration)
        name = duration.replace('traj.phases.','')
        name = name.replace('time_extents.','')
        name = name.replace('traj.design_params.design_parameters:','')
        if value <= lower or value >= upper or isclose(lower,value,rel_tol=1e-3) or isclose(upper,value,rel_tol=1e-3):
            name = name + ' (***)'
        if '_t_duration' in name:
            value = value[0]
        file1.write(name.ljust(name_len) + str(round(lower,4)).ljust(15) + str(round(value[0],4)).ljust(15) + str(round(upper,4)).ljust(15) + '\n')
    
    aux = ndarray([1])
    
    # # remove t_duration of exatmos phases
    # for param in design_params_str:
    #     print(param)
    #     if 't_duration' in param:
    #         design_params_str.remove(param)
            
    
    for param in design_params_str:
        if 't_duration' not in param:
            aux = p.driver._designvars[param]
            adder = aux['adder']
            if adder == None:
                adder = 0.0
            lower = aux['lower'] / aux['scaler'] - adder
            upper = aux['upper'] / aux['scaler'] - adder
            value = p.get_val(param)
            
                
            if value <= lower or value >= upper or isclose(lower,value,rel_tol=1e-3) or isclose(upper,value,rel_tol=1e-3):
                param = param + ' (***)'
            
            if 'external' in param:
                val_str = str(round(value[0],4)).ljust(15)
            else:
                val_str = str(round(value[0][0],4)).ljust(15)
                
            if 'traj.design_params' in param:
                param = param.replace('traj.design_params.design_parameters:','')
            elif 'external_params' in param:
                param = param.replace('external_params.','')
            
            file1.write(param.ljust(name_len) + str(round(lower,4)).ljust(15) + val_str + str(round(upper,4)).ljust(15) + '\n')
        
    file1.write('\n')    
    file1.write('Constraints:  All values in standard SI units. Angles in radians. \n')
    file1.write('Name'.ljust(name_len) + 'lower'.ljust(15) + 'value'.ljust(15) + 'upper'.ljust(15) + '\n')
    file1.write('------------------------------------------------------------------------------- \n')
    
    for con in constraints_str:
        aux = p.driver._cons[con]
        adder = aux['adder']
        scaler =  aux['scaler']
        if adder == None:
            adder = 0.0
            
        if scaler == None:
            scaler = 1.0
            
        lower = aux['lower'] / scaler - adder
        upper = aux['upper'] / scaler - adder
        value = p.get_val(con)
        
        # print(con)
        name = con.replace('traj.phases.','')
        name = name.replace('final_boundary_constraints.','')
        name = name.replace('propulsion.','')
        name = name.replace('propulsion_','')
        name = name.replace('cons_comp.','')
        name = name.replace('constraints','')

        file1.write(name.ljust(name_len) + str(round(lower,4)).ljust(15) + str(round(value[0],4)).ljust(15) + str(round(upper,4)).ljust(15) + '\n')
    
    
    file1.write('\n')
    file1.write('Vehicle paramaters \n')
    file1.write('------------------------------------------------------------------------------- \n')
    file1.write('Payload mass (kg):'.ljust(name_len) + str(round(vehicle.md,2)) + '\n')
    file1.write('Fairing mass (kg):'.ljust(name_len) + str(round(vehicle.mplf,2)) + '\n')
    file1.write('Aux mass first stage (kg):'.ljust(name_len) + str(round(vehicle.stage_1.mass_aux,2)) + '\n')
    file1.write('Propulsion type ():'.ljust(name_len) + 'LOx + RP_1' + '\n')
    file1.write('First stage:\n')
    file1.write('    Structural mass (kg):'.ljust(name_len) + str(round(vehicle.stage_1.ms,2)) + '\n')
    file1.write('    Propellants mass (kg):'.ljust(name_len) + str(round(vehicle.stage_1.mp,2)) + '\n')
    file1.write('    Structural coef ():'.ljust(name_len) + str(round(vehicle.structCoef_1,2)) + '\n')
    file1.write('    Thrust (N):'.ljust(name_len) + str(round(vehicle.stage_1.T,2)) + '\n')
    file1.write('    Isp (vac) (s):'.ljust(name_len) + str(round(vehicle.stage_1.Isp,2)) + '\n')
    file1.write('    number of engines ():'.ljust(name_len) + str(vehicle.stage_1.nb_e) + '\n')
    # file1.write('    S (m^2):'.ljust(name_len) + str(round(vehicle.stage_1.S,2)) + '\n')
    file1.write('    Ae_t (m^2):'.ljust(name_len) + str(round(vehicle.stage_1.Ae_t,2)) + '\n')
    file1.write('Second stage:\n')
    file1.write('    Structural mass (kg):'.ljust(name_len) + str(round(vehicle.stage_2.ms,2)) + '\n')
    file1.write('    Propellants mass (kg):'.ljust(name_len) + str(round(vehicle.stage_2.mp,2)) + '\n')
    file1.write('    Structural coef ():'.ljust(name_len) + str(round(vehicle.structCoef_2,2)) + '\n')
    file1.write('    Thrust (N):'.ljust(name_len) + str(round(vehicle.stage_2.T,2)) + '\n')
    file1.write('    Isp (vac) (s):'.ljust(name_len) + str(round(vehicle.stage_2.Isp,2)) + '\n')
    file1.write('    number of engines ():'.ljust(name_len) + str(vehicle.stage_2.nb_e) + '\n')
    # file1.write('    S (m^2):'.ljust(name_len) + str(round(vehicle.stage_2.S,2)) + '\n')
    file1.write('    Ae_t (m^2):'.ljust(name_len) + str(round(vehicle.stage_2.Ae_t,2)) + '\n')
    file1.write('First stage flight with fairing:\n')
    file1.write('    Tw_ratio ():'.ljust(name_len) + str(round(vehicle.TwRatio_a,2)) + '\n')
    file1.write('Second stage flight with fairing:\n')
    file1.write('    Tw_ratio ():'.ljust(name_len) + str(round(vehicle.TwRatio_b,2)) + '\n')
    
    file1.write('\n')
    file1.write('Mission paramaters \n')
    file1.write('------------------------------------------------------------------------------- \n')
    file1.write('Height at apogee (km):'.ljust(name_len) + str(ha/1e3) + '\n')
    file1.write('Min height at perigee(km):'.ljust(name_len) + str(hp_min/1e3) + '\n')
    
    m_i = p.get_val('traj.phases.lift_off.indep_states.states:m')[0] /1e3
    
    file1.write('\n')
    file1.write('Objective:'.ljust(name_len) + 'value' + '\n')
    file1.write('GLOW (ton):'.ljust(name_len) + str(round(m_i[0],3)) + '\n')
    
    file1.write('\n')
    file1.write('Initial guess:'.ljust(name_len) + guess_file + '\n')
    
    file1.write('\n')
    file1.write('Performance:'.ljust(name_len) + '\n')
    file1.write('Message:'.ljust(name_len) + p.driver.result.message + '\n')
    file1.write('Number of iterations:'.ljust(name_len) + str(p.driver.result.nit) + '\n')
    file1.write('Number of gradient evaluations:'.ljust(name_len) + str(p.driver.result.njev) + '\n')
    file1.write('Number of function evaluations:'.ljust(name_len) + str(p.driver.result.nfev) + '\n')
    file1.write('Optimization time (s):'.ljust(name_len) + str(round(time.time() - start_time,2)) + '\n')
    
    
    file1.close()
    if printToConsole == True:
        # print results to console
        file1 = open("results/" + str(Id) + "_optReport.txt","w")
        print(file1.read())
        file1.close()
        
def updateVehicle(p, vehicle):
    # update vehicle parameters with results from optimization
    thrust_vac_stage_1   = p.get_val('external_params.thrust_vac_stage_1_nominal')[0]
    Isp_1 = p.get_val('propulsion.propulsion_stage_1.Isp')[0]
    Ae_t_1  = p.get_val('propulsion.propulsion_stage_1.Ae_t')[0]
    ms_1  = p.get_val('massSizing.ms_1')[0]
    mp_1  = p.get_val('traj.lift_off.timeseries.states:m')[0][0] - p.get_val('traj.gravity_turn_c.timeseries.states:m')[0][0] - ms_1
    vehicle.write_stage_1_opt(thrust_vac_stage_1, Isp_1, Ae_t_1, mp_1, ms_1)
    
    thrust_vac_stage_2   = p.get_val('external_params.thrust_vac_stage_2_nominal')[0]
    Isp_2 = p.get_val('propulsion.propulsion_stage_2.Isp')[0]
    Ae_t_2  = p.get_val('propulsion.propulsion_stage_2.Ae_t')[0]
    mp_2  = p.get_val('traj.gravity_turn_c.timeseries.states:m')[0][0] - p.get_val('traj.exoatmos_b.timeseries.m_final')[-1][0] - vehicle.mplf
    ms_2  = p.get_val('massSizing.ms_2')[0]
    vehicle.write_stage_2_opt(thrust_vac_stage_2, Isp_2, Ae_t_2, mp_2, ms_2)
    
    #       calculateMassRatios(printRatios)
    vehicle.calculateMassRatios(False)
    
    return vehicle