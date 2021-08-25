# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:02:20 2020

The orbit injection that is considered is of type Hohmann ascent.
This implies two separate burns of the second stage. The first second-stage burn is included in the state history
to be solved using the pseudospectral method in Dymos .
The mass of fuel necessary to circularize the orbit during the second second-stage burn is calculated analytically.
This allows to exclude the long duration coast phase between second-stage burns of the pseudospectral transcription procedure.

This group finds the values of the radius at perigee and apogee (To be constrained) by transforming state values
in polar corrdinats to orbital elements.

@author: jorge
"""

import openmdao.api as om

from .components.speed_inertial import Speed_inertial
from .components.energyAndMomentum import EnergyAndMomentum
from .components.eccentricityAndMajorAxis import EccentricityAndMajorAxis
from .components.apogeeAndPerigee import ApogeeAndPerigee
from .components.delta_v2 import Delta_v2
from .components.final_mass import Final_mass

class OrbitalParameters(om.Group):
    
    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('central_body', desc = 'object of class Earth')
        
    def setup(self):
        nn      = self.options['num_nodes']
        cb      = self.options['central_body']
        
        self.add_subsystem('speed_inertial', Speed_inertial(num_nodes=nn, omega=cb.angularSpeed, r_ref=cb.r0),
                           promotes_inputs=['v'],
                           promotes_outputs=['v_i'])
        
        self.add_subsystem('energyAndMomentum', EnergyAndMomentum(num_nodes=nn, mu = cb.mu),
                           promotes_inputs=['v_i','r','phi'],
                           promotes_outputs=['E', 'H'])
        
        self.add_subsystem('eccentricityAndMajorAxis', EccentricityAndMajorAxis(num_nodes=nn, mu = cb.mu),
                          promotes_inputs=['E','H'],
                          promotes_outputs=['e','a'])
        
        self.add_subsystem('apogeeAndPerigee', ApogeeAndPerigee(num_nodes=nn),
                           promotes_inputs=['e','a'],
                           promotes_outputs=['ra','rp'])
        
        self.add_subsystem('delta_v2', Delta_v2(num_nodes=nn, mu = cb.mu),
                           promotes_inputs=['ra','rp'],
                           promotes_outputs=['delta_v2'])
        
        self.add_subsystem('final_mass', Final_mass(num_nodes=nn, g0 = cb.g0),
                           promotes_inputs=['delta_v2','m_0', 'Isp'],
                           promotes_outputs=['m_final'])
        