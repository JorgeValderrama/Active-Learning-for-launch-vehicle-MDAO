# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:31:11 2018

This group calculates the mass of the first stage as in FELIN
The models are based on Castellini.

It only considers the models in  FELIN for the following inputs

type_fuel    = 'RP1'
type_prop    = 'Cryostorable'
feed         = 'GG'

material     = 'Al'          ---> thrust frame mass
config       = 'intertank'   ---> tank_mass
type_stage   = 'lower'       ---> tank_mass
type_struct  = 'Al'          ---> tank_mass

techno       = 'hydraulic'   ---> TVC_mass
RL           = 3             ---> EPS_avio_mass
U_L          = 'lower'       ---> mass interstages
S_interstage = 30

@author: lbrevaul
@modified: jorge
"""
from __future__ import print_function

from openmdao.api import  Group

from .Mass_models import Sizing, SingleEngineThrust, Engine_mass, Thrust_frame_mass, Tank_volume, Tank_mass, TVC_mass
from .Mass_models import EPS_avio_mass, Interstage_mass, AddUpMass

class Dry_Mass_stage_1_Comp(Group):
    
    
    def initialize(self):
        self.options.declare('nb_e', 
                             types = int, 
                             desc = 'number of engines')
        
        self.options.declare('mass_aux_1', 
                             types = float, 
                             desc = 'auxiliary mass for the first stage in kg')
        
        self.options.declare('mu_F',
                              types = float,
                              default = 810.0,
                              desc ='fuel density (kg/m**3)')
        
        self.options.declare('mu_LOX',
                              types = float,
                              default = 1141.0,
                              desc ='liquid oxygen density (kg/m**3)')
        
        self.options.declare('S_interstage',
                              default = 30.0,
                              types = float,
                              desc = 'surface area of interstage')
        
        self.options.declare('P_tanks_Ox',
                              default = 3.0,
                              types = float,
                              desc = '')
        
        self.options.declare('P_tanks_F',
                              default = 3.0,
                              types = float,
                              desc = '')

        self.options.declare('uncertainty_dry_mass_1', 
                             types = float, 
                             desc = 'uncertainty of dry mass for the first stage in  kg')
    def setup(self):
        
        nb_e         = self.options['nb_e']
        mass_aux_1   = self.options['mass_aux_1']
        mu_F         = self.options['mu_F']
        mu_LOX       = self.options['mu_LOX']
        S_interstage = self.options['S_interstage']
        P_tanks_Ox   = self.options['P_tanks_Ox']
        P_tanks_F    = self.options['P_tanks_F']
        uncertainty_dry_mass_1 = self.options['uncertainty_dry_mass_1']
        
        self.add_subsystem('sizing', Sizing( mu_F = mu_F, mu_LOX = mu_LOX ),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        self.add_subsystem('singleEngineThrust', SingleEngineThrust(nb_e = nb_e),
                           promotes_inputs=[('thrust','thrust_vac_stage_1')],
                           promotes_outputs=['*'])
        
        self.add_subsystem('engine_mass', Engine_mass(),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        self.add_subsystem('thrust_frame_mass', Thrust_frame_mass (nb_e = nb_e),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        self.add_subsystem('tank_volume', Tank_volume ( mu_F = mu_F, mu_LOX = mu_LOX),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        self.add_subsystem('tank_mass', Tank_mass( P_tanks_Ox = P_tanks_Ox, P_tanks_F = P_tanks_F),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        self.add_subsystem('TVC_mass', TVC_mass(),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        self.add_subsystem('EPS_avio_mass', EPS_avio_mass(),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        self.add_subsystem('interstage_mass', Interstage_mass( S_interstage = S_interstage),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        self.add_subsystem('addUpMass', AddUpMass( nb_e = nb_e , mass_aux_1 = mass_aux_1, uncertainty_dry_mass_1 = uncertainty_dry_mass_1),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        