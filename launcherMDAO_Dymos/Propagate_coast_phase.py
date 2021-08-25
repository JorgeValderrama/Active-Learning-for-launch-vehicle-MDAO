# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:27:39 2020

@author: jorge
"""
import numpy as np


def propagate_coast(t,y,celestialBody,*args):
    
    "this function defines the equations of motion to be used when propagating the coast phase with a Runge Kutta solver"
    
    # extract omega from object celestialBody
    om = celestialBody.angularSpeed
    
    # extrect thrust from object vehicle
    T = 0
       
    # state vector
    r, v, m, la, ph = y
    
    
    D= 0
    L = 0
    
    # Select theta according to phase
    th = ph
    
    # call to gravity model
    g = celestialBody.gravitySpheric(r)

    #================  Differential equations =====================================================
    # distance change to earth's center
    r_dot = v * np.sin(ph)
    # velocity change compared to fix point on Earth
    v_dot = (T*np.cos(th-ph) - D)/m + (om**2*r - g) * np.sin(ph)
    # mass change
    m_dot = 0
    # lambda. CHange in longitude
    la_dot = v/r * np.cos(ph)
    # phi. Change in flight path angle
    ph_dot = L/(m*v) + T*np.sin(th-ph)/(m*v) + (om**2*r - g) * np.cos(ph) / v + 2*om + (v/r) * np.cos(ph) 
    # =============================================================================================
    
    # derivatives of state vector
    dydt = [r_dot,v_dot,m_dot,la_dot,ph_dot]
    
    return dydt
