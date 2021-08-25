# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:51:56 2020

@author: jorge
"""
import matplotlib
# matplotlib.use('agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import json

from .Propagate_coast_phase import propagate_coast

plt.rcParams.update({'font.size': 16})

def saveTraj(p, phases, exp_out, uncertainties, plot_coast, earth, Id, resultsPath):
    "This function saves the state and load history in a txt file"
    
    # merge results
    # ============================================================================================
    results, results_sim = mergeResults(p, phases, exp_out)
    
    # add coast phase results and define time limits
    # ============================================================================================
    if plot_coast == True:
        results_sim = addCoastResults(results_sim, earth)

    # convert np.arrays into lists
    for key, value in results.items():
        results[key] = value.tolist()
        
    # convert np.arrays into lists
    for key, value in results_sim.items():
        results_sim[key] = value.tolist()
        
    # Create dictionary with optimization info
    GLOW = p.get_val('traj.phases.lift_off.indep_states.states:m')[0][0]  # gross lift off weight
    hp   = (p.get_val('traj.phases.exoatmos_b.indep_states.states:r')[-1] - earth.r0 )[0]
    
    opt_info = {}
        
    opt_info['status']         = p.driver.result.status
    opt_info['uncertainties']  = np.array(uncertainties).tolist()
    opt_info['iterations']     = p.driver.result.nit
    opt_info['GLOW']           = GLOW
    opt_info['height_perigee'] = hp
    
    
    # write json file
    exDict = {'resultsDict': results,
              'results_simDict':results_sim,
              'opt_info': opt_info}
    
    with open(resultsPath + '/' + str(Id) + '_trajectory_state_history.txt', 'w') as file:
         file.write(json.dumps(exDict)) 
         
         
    # to read it use the following code
    # ===========================================================================================
    # obj_text = codecs.open('results/trajectory_state_history.txt', 'r', encoding='utf-8').read()
    # b_new = json.loads(obj_text)
    # a_new = np.array(b_new)

def plotState(p,earth,phases,exp_out, plot_coast, plot_loads, Id, resultsPath):
    "This function plots the states and the loads of the optimal trajectory"
    # if plot_coast == True: propagates the coast phase of the second stage and plots it
    # if plot_loads == True: plot Mach, q_heat, q_dyn and load factor
    
    # merge results
    # ============================================================================================
    results, results_sim = mergeResults(p, phases, exp_out)
        
    # add coast phase results and define time limits
    # ============================================================================================
    if plot_coast == True:
        results_sim = addCoastResults(results_sim, earth)
        t_coast = 3800 # define time to ploat coast phase
    elif plot_coast == False:
        t_coast = results['time'][-1] + 20 
    
    # define time for plots that don't change if coast_phase == True
    t   =  results['time'][-1] + 20 
        
    # find index and time for first stage jettison and fairing jettison
    # ============================================================================================
    stageJetisson_idx = np.where(results['r'] == p.get_val('traj.gravity_turn_b.timeseries.states:r')[-1])[0][0]
    plfJetisson_idx   = np.where(results['r'] == p.get_val('traj.exoatmos_a.timeseries.states:r')[-1])[0][0]
    stageJetisson_t   = p.get_val('traj.gravity_turn_b.timeseries.time')[-1]
    plfJetisson_t     = p.get_val('traj.exoatmos_a.timeseries.time')[-1]
        
    
    # plot states
    # ============================================================================================
    
    plt.figure(figsize=(12, 16))
    
    plt.subplot(3, 2, 1)
    plt.plot(results['time']    ,( (results['r']     - earth.r0)/ 1e3),"red",label='h',marker='o',ms=3)
    plt.plot(results_sim['time'],( (results_sim['r'] - earth.r0)/ 1e3),"black",label='h sim',lw=1)
    plt.plot(stageJetisson_t, (results['r'][stageJetisson_idx] - earth.r0)/1e3, "black",marker='o',ms=8)
    plt.plot(plfJetisson_t,   (results['r'][plfJetisson_idx]   - earth.r0)/1e3,   "black",marker='d',ms=8)
    plt.legend(loc='lower right')
    plt.xlabel('time (s)')
    plt.ylabel('height (km)')
    plt.xlim(0,t_coast)
    plt.grid()
    
    plt.subplot(3, 2, 2)
    plt.plot(results['time']    ,(results['v']     / 1e3),"red",label='v',marker='o',ms=3)
    plt.plot(results_sim['time'],(results_sim['v'] / 1e3),"black",label='v sim',lw=1)
    plt.plot(stageJetisson_t, results['v'][stageJetisson_idx]/1e3, "black",marker='o',ms=8)
    plt.plot(plfJetisson_t,   results['v'][plfJetisson_idx]/1e3,   "black",marker='d',ms=8)
    plt.legend(loc='lower right')
    plt.xlabel('time (s)')
    plt.ylabel('speed (km/s)')
    plt.xlim(0,t_coast)
    # plt.ylim(0,(11))
    plt.grid()
        
    plt.subplot(3, 2, 3)
    plt.plot(results['time']    ,    np.degrees(results['ph']),"red",label='$\phi$',marker='o',ms=3)
    plt.plot(results_sim['time'],np.degrees(results_sim['ph']),"black",label='$\phi$ sim',lw=1)
    plt.plot(results['time']    ,    np.degrees(results['th']),"orange",label=r'$\theta$',marker='o',ms=3)
    plt.plot(results_sim['time'],np.degrees(results_sim['th']),"blue",label=r'$\theta$ sim',lw=1)
    plt.plot(stageJetisson_t, np.degrees(results['ph'][stageJetisson_idx]), "black",marker='o',ms=8)
    plt.plot(plfJetisson_t,   np.degrees(results['ph'][plfJetisson_idx]),   "black",marker='d',ms=8)
    plt.legend(loc='upper right',fontsize='12',labelspacing=0.1)
    plt.xlabel('time (s)' )
    plt.ylabel('angle (Deg)')
    plt.xlim(0,t_coast)
    plt.grid()
    
    plt.subplot(3, 2, 4)
    plt.plot(results['time']    ,    np.degrees(results['la']),"red",label='$\lambda$',marker='o',ms=3)
    plt.plot(results_sim['time'],np.degrees(results_sim['la']),"black",label='$\lambda$ sim',lw=1)
    plt.plot(stageJetisson_t, np.degrees(results['la'][stageJetisson_idx]), "black",marker='o',ms=8)
    plt.plot(plfJetisson_t,   np.degrees(results['la'][plfJetisson_idx]),   "black",marker='d',ms=8)
    plt.legend(loc='lower right')
    plt.xlabel('time (s)')
    plt.ylabel('longitude (Deg)')
    plt.xlim(0,t_coast)
    plt.grid()
    

    plt.subplot(3, 2, 5)
    plt.plot(results['time']     , results['m']   / 1e3  , "red"  ,label='mass' ,marker='o',ms=3)
    plt.plot(results_sim['time'],results_sim['m'] / 1e3,"black",label='mass sim',lw=1)
    plt.plot(stageJetisson_t, results['m'][stageJetisson_idx]/1e3, "black",marker='o',ms=8)
    plt.plot(plfJetisson_t,   results['m'][plfJetisson_idx]/1e3,   "black",marker='d',ms=8)
    plt.legend(loc='upper right')
    plt.xlabel('time (s)')
    plt.ylabel('ton')
    plt.xlim(0,t)
    plt.grid()
    
    plt.subplot(3, 2, 6)
    plt.plot(results['time']     , results['m']*results['g']   / 1e6  , "red"  ,label='weight' ,marker='o',ms=3)
    plt.plot(results['time']     , results['thrust'] / 1e6 , "black", label='thrust',marker='o',ms=3)
    plt.legend(loc='upper right')
    plt.xlabel('time (s)')
    plt.ylabel('MN')
    plt.xlim(0,t)
    plt.grid()
    
    if plot_coast == False:
        plt.savefig(resultsPath + '/' + str(Id) + '_state_history.png') 
    elif plot_coast == True:
        plt.savefig(resultsPath + '/' + str(Id) + '_state_history_coast.png') 
    
    if plot_loads == True:
        # plot loads
        # ============================================================================================
        plt.figure(figsize=(12, 12))
        
        plt.subplot(2, 2, 1)
        plt.plot(results['time'],    results['q_dyn']/1e3  ,"r",marker='o',ms=3, label= 'q_dyn')
        plt.plot(stageJetisson_t, results['q_dyn'][stageJetisson_idx]/1e3, "black",marker='o',ms=8)
        plt.plot(plfJetisson_t,   results['q_dyn'][plfJetisson_idx]/1e3,   "black",marker='d',ms=8)
        plt.legend(loc='best')
        plt.xlabel('time (s)')
        plt.ylabel('q_dyn (kPa)')
        plt.xlim(0,t)
        plt.grid()
    
        plt.subplot(2, 2, 2)
        plt.plot(results['time'],    results['q_heat']/1e6 ,"b",marker='o',ms=3, label= 'q_heat')
        plt.plot(stageJetisson_t, results['q_heat'][stageJetisson_idx]/1e6, "black",marker='o',ms=8)
        plt.plot(plfJetisson_t,   results['q_heat'][plfJetisson_idx]/1e6,   "black",marker='d',ms=8)
        plt.legend(loc='best')
        plt.xlabel('time (s)')
        plt.ylabel('q_heat (MW/m^2)')
        plt.xlim(0,t)
        plt.grid()
    
        plt.subplot(2, 2, 3)
        plt.plot(results['time'],    results['mach'] ,"g",marker='o',ms=3, label = 'Mach')
        plt.plot(stageJetisson_t, results['mach'][stageJetisson_idx], "black",marker='o',ms=8)
        plt.plot(plfJetisson_t,   results['mach'][plfJetisson_idx],   "black",marker='d',ms=8)
        plt.legend(loc='best')
        plt.xlabel('time (s)')
        plt.xlim(0,t)
        plt.grid()    
        
        plt.subplot(2, 2, 4)
        plt.plot(results['time']     , results['n_f']  , "red"  ,label='axial load factor' ,marker='o',ms=3)
        plt.plot(stageJetisson_t, results['n_f'][stageJetisson_idx], "black",marker='o',ms=8)
        plt.plot(plfJetisson_t,   results['n_f'][plfJetisson_idx],   "black",marker='d',ms=8)
        plt.legend(loc='best')
        plt.xlabel('time (s)')
        plt.xlim(0,t)
        plt.grid()
        
        plt.savefig(resultsPath + '/' + str(Id) + '_load_history.png')
        
        # ============================================================================================
           

def mergeResults(p, phases, exp_out):
    "This function merges the results of each phase and returns them in dictionaries"
    
    # define empty numpy arrays to store variables
    # ===================================================================================================
    time  = time_sim = np.zeros(0)
    r     = r_sim    = np.zeros(0)
    la    = la_sim   = np.zeros(0)
    v     = v_sim    = np.zeros(0)
    ph    = ph_sim   = np.zeros(0)
    m     = m_sim    = np.zeros(0)
    th    = th_sim   = np.zeros(0)
    
    rho    = np.zeros(0)
    mach   = np.zeros(0)
    Cd     = np.zeros(0)
    q_dyn  = np.zeros(0)
    q_heat = np.zeros(0)
    n_f    = np.zeros(0)
    
    g      = np.zeros(0)
    thrust = np.zeros(0)
    
    throttle = np.zeros(0)
    
    # loop through phases to store values in the same array
    # ==================================================================================================
    for phase in phases:
        call_str = 'traj.' + phase + '.timeseries'
    
        time     = np.append( time     ,       p.get_val(call_str + '.time') )
        time_sim = np.append( time_sim , exp_out.get_val(call_str + '.time') )
        
        r        = np.append( r        ,       p.get_val(call_str + '.states:r') )
        r_sim    = np.append( r_sim    , exp_out.get_val(call_str + '.states:r') )
        
        la       = np.append( la       ,       p.get_val(call_str + '.states:lambda') )
        la_sim   = np.append( la_sim   , exp_out.get_val(call_str + '.states:lambda') )
        
        v        = np.append( v        ,       p.get_val(call_str + '.states:v') )
        v_sim    = np.append( v_sim    , exp_out.get_val(call_str + '.states:v') )
        
        ph       = np.append( ph       ,       p.get_val(call_str + '.states:phi') )
        ph_sim   = np.append( ph_sim   , exp_out.get_val(call_str + '.states:phi') )
        
        m        = np.append( m        ,       p.get_val(call_str + '.states:m') )
        m_sim    = np.append( m_sim    , exp_out.get_val(call_str + '.states:m') )
        
        g        = np.append( g        ,       p.get_val(call_str + '.g'))
        
        thrust   = np.append( thrust   ,       p.get_val(call_str + '.thrust'))
        
        throttle = np.append( throttle ,       p.get_val(call_str + '.controls:throttle'))
        
        if phase == 'lift_off':
            th     = np.append( th     ,       p.get_val(call_str + '.input_parameters:theta'))
            th_sim = np.append( th_sim , exp_out.get_val(call_str + '.input_parameters:theta'))
        else:
            th     = np.append( th     ,       p.get_val(call_str + '.theta'))
            th_sim = np.append( th_sim , exp_out.get_val(call_str + '.theta'))
            
        rho      = np.append( rho      ,       p.get_val(call_str + '.rho'))
        mach     = np.append( mach     ,       p.get_val(call_str + '.Mach'))
        Cd       = np.append( Cd       ,       p.get_val(call_str + '.Cd'))
        q_dyn    = np.append( q_dyn    ,       p.get_val(call_str + '.q_dyn'))
        q_heat   = np.append( q_heat   ,       p.get_val(call_str + '.q_heat'))
        n_f      = np.append( n_f      ,       p.get_val(call_str + '.n_f'))
    
    # build dictionaries
    # ==================================================================================================
        
    results = {'time'     : time,
              'r'        : r,
              'la'       : la,
              'v'        : v,
              'ph'       : ph,
              'm'        : m,
              'g'        : g,
              'thrust'   : thrust,
              'throttle' : throttle,
              'th'       : th,
              'rho'      : rho,
              'mach'     : mach,
              'Cd'       : Cd,
              'q_dyn'    : q_dyn,
              'q_heat'   : q_heat,
              'n_f'      : n_f}
    
    results_sim = {'time'     : time_sim,
                  'r'        : r_sim,
                  'la'       : la_sim,
                  'v'        : v_sim,
                  'ph'       : ph_sim,
                  'm'        : m_sim,
                  'th'       : th_sim}
    
    return results, results_sim

def addCoastResults(results_sim, earth):
    " This function uses a Runge Kutta method to progate the second state  coast phase and appends the results"
    
    r_0   = results_sim['r'][-1]
    v_rel = results_sim['v'][-1]
    v_0   = v_rel - (r_0 - earth.r0) * earth.angularSpeed
    m_0   = results_sim['m'][-1]
    la_0  = results_sim['la'][-1]
    ph_0  = results_sim['ph'][-1]
    
    y0 = [r_0 , v_0 , m_0 , la_0 , ph_0]
    
    # Time parameters
    ti = results_sim['time'][-1]                                     
    tf_max = 60*100  + ti  
    tf_max = 100  + ti                     
    t_span = (ti,tf_max)
    
    # simulation parameters
    max_step = 0.1

    # Solve ODE Systemd
    coast_sol = solve_ivp( lambda t,y:propagate_coast(t,y,earth), t_span , y0, method='RK45', dense_output=True, max_step = max_step, rtol=1e-4 )  

    results_sim['r']    = np.append(results_sim['r'] ,   coast_sol.y[0,:])
    results_sim['v']    = np.append(results_sim['v'] ,   coast_sol.y[1,:])
    results_sim['m']    = np.append(results_sim['m'] ,   coast_sol.y[2,:])
    results_sim['la']   = np.append(results_sim['la'],   coast_sol.y[3,:])
    results_sim['ph']   = np.append(results_sim['ph'],   coast_sol.y[4,:])
    results_sim['time'] = np.append(results_sim['time'], coast_sol.t)
    results_sim['th']   = np.append(results_sim['th'],   coast_sol.y[4,:])
    
    return results_sim
