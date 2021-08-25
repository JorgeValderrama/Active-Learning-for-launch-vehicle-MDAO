# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 21:04:44 2020

@author: jorge
"""
import numpy as np

class Earth():
    # General paramaters of Earth including, gravity, angular speed and atmosphere
    def __init__(self): 
        self.angularSpeed = 7.2921159e-5                                        # (rad/s)
        self.r0 = 6378.135e3                                                    # (m) radius at sea level
        self.mu = 3.986004e14                                                   # (m3 s^{−2})mu=GMe
        self.armonics = {'J2':1.08263e-3, 'J3':2.532153e-7 , 'J4':1.6109876e-7} # First Jefferey constants or armonics
        self.g0 = 9.80665                                                       # (m/s^2) standard gravity 
        self.P0 = 101325.0                                                      # (Pa) Pressure at r0
        self.rho0 = 1.225                                                       # (kg/m^3) Air density at r0
        self.h_scale = 8.44E3                                                   # reference height for exponential atmos model
    
    def gravitySpheric(self,r):
        # Print alert in case r is under the sea level
        # if r < self.r0:
        #     print('r = ' + str(r/1e3) + 'km. gravitySpheric received as input an r value less than earth.r0')
        # Newton's gravity model
        # Input, distance from the center of the Earth to the point at which the gravity value is to be calcualted
        g = self.mu / (r**2)
        return g
    
    def orbitalSpeed(self,h):
        # Orbital speed according to Newton's gravity model in inertial ref frame
        # Input h (m), height above sea level.
        
        # Print alert in case h is negative
        if h < 0:
            print('h = ' + str(h) + ' m. orbitalSpeed received as input a negative h.')
        os = (self.gravitySpheric(self.r0 + h) * (self.r0 + h))**0.5
        return os
    
    def relativeOrbitalSpeed(self,h,r_ref):
        # Orbital speed according to Newton's gravity model in relative ref frame with distance h_ref to the center of the planet
        # Input h (m), height above sea level. r_ref is the is the refernece point at wich v_0 was specified..usually r_0
        
        # Print alert in case h is negative
        if h < 0:
            print('h = ' + str(h) + ' m. relativeOrbitalSpeed received as input a negative h.')
            
        if r_ref < self.r0:
            print('r_ref = ' + str(r_ref/1e3) + 'km. relativeOrbitalSpeed received as input an r_ref value less than earth.r0')
        
        ros = self.orbitalSpeed(h) - self.angularSpeed * r_ref
        return ros
        
    def gravityOblate(self,d,r):
        # Im not using this method yet in Dymos.
        # gravity model for oblate planet Earth as found in Tewari 
        phi = np.pi/2 - d
        mu = self.mu
        J2 = self.armonics['J2']
        J3 = self.armonics['J3']
        J4 = self.armonics['J4']
        Re = self.r0
        gc = mu * (1-1.5*J2*(3*np.cos(phi)**2-1)*(Re/r)**2-2*J3*np.cos(phi)*(5*np.cos(phi)**2-3)*(Re/r)**3-(5/8)*J4*(35*np.cos(phi)**4-30*np.cos(phi)**2+3)*(Re/r)**4)/r**2 
        gd = -3*mu*np.sin(phi)*np.cos(phi)*(Re/r)*(Re/r)*(J2+0.5*J3*(5*np.cos(phi)**2-1)*(Re/r)/np.cos(phi)+(5/6)*J4*(7*np.cos(phi)**2-1)*(Re/r)**2)/r**2
        return gc , gd
    