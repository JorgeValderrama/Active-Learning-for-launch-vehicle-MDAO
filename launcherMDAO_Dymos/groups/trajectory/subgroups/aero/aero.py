# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 09:53:56 2020

This group calculates the drag force "D", the dynamic pressure "q_dyn" and heat flux "q_heat" as
functions of the radius "r" and speed "v". 

It uses Interpolators for an atmospheric model that spans heights of [0, 2000) km 
and for the modelling of drag coeff as a function of mach number.

@author: jorge
"""

import openmdao.api as om

from .components.altitude import Altitude
from .components.us_atmos import USatm
from .components.mach_number import Mach_number
from .components.drag_coefficient import Drag_coefficient
from .components.heatFluxAndDynamicPressure import HeatFluxAndDynamicPressure
from .components.dragForce import DragForce

class Aero(om.Group):
    
    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('central_body',desc = 'object of class Earth')
        # self.options.declare('S', types=float, desc='aerodynamic reference area (m**2)')
        self.options.declare('uncertainty_cd', types=float, desc='uncertainty about drag coefficient (-)')

    def setup(self):
        nn      = self.options['num_nodes']
        cb      = self.options['central_body']
        # S       = self.options['S']
        
        self.add_subsystem('altitude', Altitude(num_nodes=nn, r0 = cb.r0),
                           promotes_inputs=['r'],
                           promotes_outputs=['h'])
        
        self.add_subsystem('atmos', USatm(num_nodes=nn),
                           promotes_inputs=['h'],
                           promotes_outputs=['rho','P_a','sos', 'd_rho_wrt_h'])
        
        self.add_subsystem('mach_number', Mach_number(num_nodes=nn),
                           promotes_inputs=['v','sos'],
                           promotes_outputs=['Mach'])
        
        self.add_subsystem('drag_coefficient', Drag_coefficient(num_nodes=nn),
                           promotes_inputs=['Mach'],
                           promotes_outputs=['Cd'])
        
        self.add_subsystem('heatFluxAndDynP', HeatFluxAndDynamicPressure(num_nodes=nn),
                           promotes_inputs=['v','rho'],
                           promotes_outputs=['q_dyn','q_heat'])
        
        self.add_subsystem('dragForce', DragForce(num_nodes = nn,uncertainty_cd = self.options['uncertainty_cd']),
                           promotes_inputs=['q_dyn', 'Cd', 'diameter'],
                           promotes_outputs=['Drag'])
        
        
        