# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:25:57 2020


The structure of this group  replicates the one  in LAST (Launcher Analysis and Sizing Tool), although some models differ.
It calculates propulsion variables bases on interpolation of Rocket CEA data.

@author: jorge
"""

import openmdao.api as om

from .rocket_cea             import Rocket_cea
from .characteristicVelocity import CharacteristicVelocity
from .areaRatio              import AreaRatio
from .thrustCoefficient      import ThrustCoefficient
from .specificImpulse        import SpecificImpulse



class Chemistry (om.Group):
    
    def initialize(self):
        
        self.options.declare('g0', types=float,
                              desc='gravity at r0 (m/s**2)')
        
        self.options.declare('eta_C_f', types=float,
                             desc = 'C_f efficiency')
        
        self.options.declare('eta_cStar', types=float,
                             desc = 'cStar efficiency')
        
        self.options.declare('Rmc', types=float,
                             desc = 'gass constant times molecular weight (mJ/K/mol )')
        
        self.options.declare('P_a', types = float, desc= 'atmospheric pressure')
        
        self.options.declare('uncertainty_Isp', types = float, desc= 'uncertainty about Isp (s)',
                             default = 0.0)    
        
    def setup(self):
        
        g0         = self.options['g0']
        eta_C_f      = self.options['eta_C_f']
        eta_cStar = self.options['eta_cStar']
        Rmc        = self.options['Rmc']
        P_a        = self.options['P_a']
        
        self.add_subsystem('rocket_cea', Rocket_cea(),
                           promotes_inputs=['P_c', 'o_f'],
                           promotes_outputs=['gamma_t', 'tc', 'mc'])
        
        self.add_subsystem('characteristicVelocity', CharacteristicVelocity(Rmc = Rmc, eta_cStar = eta_cStar),
                           promotes_inputs=['gamma_t', 'tc', 'mc'],
                           promotes_outputs=['cStar'])
        
        self.add_subsystem('areaRatio', AreaRatio(),
                           promotes_inputs=['P_c', 'P_e', 'gamma_t'],
                           promotes_outputs=['epsilon'])
        
        self.add_subsystem('thrustCoefficient', ThrustCoefficient(P_a = P_a, eta_C_f = eta_C_f),
                           promotes_inputs=['P_c', 'P_e', 'epsilon', 'gamma_t'],
                           promotes_outputs=['C_f'])
        
        self.add_subsystem('specificImpulse', SpecificImpulse( g0 = g0, uncertainty_Isp = self.options['uncertainty_Isp']),
                           promotes_inputs=['cStar', 'C_f'],
                           promotes_outputs=['Isp'])
        