# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:15:35 2021

@author: jorge
"""

# function to verify that the mass of the converged solution is behaving correclty

def checkMass(p, vehicle):
    
    # mass at the instant before first stage jettison
    me_a = p.get_val('traj.gravity_turn_b.timeseries.states:m')[-1]
    # mass at the instant after first stage jettison
    mf_b = p.get_val('traj.gravity_turn_c.timeseries.states:m')[0]
    # residul of first stage jettison must be zero
    residualStageJettison = me_a - mf_b - vehicle.stage_1.ms
    print(' ========== Mass checks ===================================')
    print('residual Stage Jettison (kg): %.6f'%residualStageJettison)
    
    
    # mass at the instant before fairing jettison
    mi_b = p.get_val('traj.exoatmos_a.timeseries.states:m')[-1]
    # mass at the instant after fairing jettison
    mi_c = p.get_val('traj.exoatmos_b.timeseries.states:m')[0]
    # resiudal of payload fairing jettison must be zero
    residualFairingJettison = mi_b - mi_c - vehicle.mplf
    print('residual Fairing Jettison (kg): %.6f'%residualFairingJettison)
    
    
    # print the mass necessarry for orbit ciruclarization
    # mass at the instant after first SECO
    m1SECO = p.get_val('traj.exoatmos_b.timeseries.states:m')[-1]
    m2SECO = p.get_val('traj.exoatmos_b.timeseries.m_final')[-1]
    print('mass necessarry for orbit circularization (kg): %.6f' %(m1SECO - m2SECO))
    
    # verify that mass after orbit circularization adds up to the strucutralm ass of the second 
    # stage plus the payload mass. resiudal should be zero
    residualMassFinal = m2SECO - vehicle.stage_2.ms - vehicle.md
    print('residualMassFinal (kg): %.6f'%residualMassFinal)
    
    # check that GLOW corresponds to the sum of masses
    GLOW_LGL    = p.get_val('traj.lift_off.timeseries.states:m')[0][0]
    
    GLOW_add_up = p.get_val('massSizing.ms_1') + \
                  p.get_val('massSizing.ms_2') + \
                  p.get_val('external_params.mp_1') + \
                  p.get_val('external_params.mp_2') + \
                  vehicle.mplf +\
                  vehicle.md 
                  
    print('GLOW as mass of the first node (kg): %.6f'% GLOW_LGL)
    print('GLOW as sum of ms, mp ,md, mplf (kg): %.6f'% GLOW_add_up)
    print(' ============================================================')
    
    