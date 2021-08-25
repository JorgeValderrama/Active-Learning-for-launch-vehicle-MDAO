# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:57:27 2020

This function uses tables and equations to calculate atmospheric values at a give height and speed.
Taken from Tewari.
Standard Atmosphere Derived from 1976 and 1962 U.S. Standard Atmospheres.
This model provides data up to 2 000 km height.
Tewari - ATMOSPHERIC AND SPACE FLIGHT DYNAMICS
 
inputs:
    - h : altitude in m
    - speed: speed of the vehicle in m/s
    
outputs:
    - mach: mach number
    - rhoE: air density in kg/m**3
    - Pr  : atmospheric pressure in Pa
    - TM  : atmospheric temperature in K
    
@author: jorge
"""

import numpy as np

# %% 
def atmosphere (h, speed) :
        
        
        R = 287             # gas constant                                                                             
        go = 9.806          # standard gravity
        To = 288.15         # sea level temp
        Po = 1.01325e5      # sea leve press
        re = 6378.14e3      # radius of earth at sea level. Tewari uses 
        # h = r - re          # height above sea level
        B = 2 / re
        gamma = 1.405       # adiab contant
        layers = 21
        
        # Print alert if height goes above 2000 km and stop execution of this fucntion
        if h >= 2000e3:
            print('h = '+ str(h/1e3) + ' km. Atmosphere is only defined for heights under 2 000 km.')
            return
        
        # Print alert if velocity is negative
        if speed < 0:
            print('v = ' + str(speed) + 'm/s. Atmosphere received a negative speed as input. Returning negative Mach number.')
        
        # height in meters
        Z = [0 , 11019.1 , 20063.1 , 32161.9 , 47350.1 , 51412.5 , 71802.0 , 86000 , 100000 , 110000 , 120000 , 150000 ,  160000 , 170000 , 190000 , 230000 , 300000 , 400000 , 500000 , 600000 , 700000 , 2000000]
        # Temperature in kelvin
        T = [288.15 , 216.65 , 216.65 , 228.65 , 270.65 , 270.65 , 214.65 , 186.946 , 210.02 , 260.65 , 360.65 , 960.65 , 1110.60 , 1210.65 , 1350.65 , 1550.65 , 1830.65 , 2160.65 , 2420.65 , 2590.65 , 2700.00 , 2700.0]      
        # Slope K / m
        LR   = [-6.5e-3 , 0 , 1e-3 , 2.8e-3 , 0 , -2.8e-3 , -2e-3 , 1.693e-3 , 5.00e-3 , 1e-2 , 2e-2 , 1.5e-2 , 1e-2 , 7e-3 , 5e-3 , 4e-3 , 3.3e-3 , 2.6e-3 , 1.7e-3 , 1.1e-3 , 0]                                 
        
        rho0 = Po / (R * To)
        P = [Po]
        rho = [rho0]
        
        # Complete data chart with density and pressure
        for i in range(0, layers):
            if LR[i] != 0:
                C1 = 1 + B * ( T[i]/LR[i] - Z[i] )
                C2 = C1 * go / (R * LR[i])
                C3 = T[i+1]/T[i]
                C4 = C3**(-C2)
                C5 = np.exp( go * B * (Z[i+1]-Z[i]) / (R * LR[i]) )
                P.append(P[i] * C4 * C5)
                C7 = C2 + 1
                rho.append(rho[i] * C5 * C3**(-C7))
            else:       
                C8 = -go * (Z[i + 1] - Z[i]) * (1 - B * (Z[i + 1] + Z[i]) / 2) / (R * T[i])
                P.append (P[i] * np.exp(C8))
                rho.append (rho[i] * np.exp(C8))
            
        # Look for the correct range in the chart and interpolate
        for i in range(0, layers):
            if h < Z[i+1]:
                if LR[i] != 0:
                    C1 = 1 + B * ( T[i]/LR[i] - Z[i] )
                    TM = T[i] + LR[i]*(h - Z[i])
                    C2 = C1 * go / (R * LR[i])
                    C3 = TM/T[i]
                    C4 = C3**(-C2)
                    C5 = np.exp( B * go * (h - Z[i]) / (R * LR[i]) )
                    C7 = C2 + 1
                    rhoE = C5 * rho[i] * C3**(-C7) 
                    Pr = P[i] * C4 * C5
                else:
                    TM = T[i]
                    C8 = -go * (h - Z[i]) * (1 - (h + Z[i]) * B/2) / (R * T[i])
                    rhoE = rho[i] * np.exp(C8)
                    Pr = P[i] * np.exp(C8)
                sos = (gamma * R * TM)**(1/2)
                mach = speed / sos
                return mach , rhoE, Pr, TM
    