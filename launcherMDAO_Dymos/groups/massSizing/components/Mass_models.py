# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:31:11 2018

This script contains all the models used for the calculation of the dry mass of the first stage as in FELIN.
The models from FELIN were taken and adapted to the Dymos environment.

@author: lbrevaul
@modified: jorge
"""
import openmdao.api as om
import numpy as np

class AddUpMass (om.ExplicitComponent):
    "This class adds up the masses of all the vehicle components to calculate the dry mass of the first stage"
    
    def initialize(self):
        
        self.options.declare('nb_e', types = int, desc = 'number of engines')
        
        self.options.declare('mass_aux_1', types = float, desc = 'auxliary mass for the first stage in kg')
        
        self.options.declare('uncertainty_dry_mass_1', 
                             types = float, 
                             desc = 'uncertainty of dry mass for the first stage in  kg')
        self.options.declare('uncertainty_dry_mass_2', 
                             types = float, 
                             desc = 'uncertainty of dry mass for the second stage in  kg')        
    def setup(self):
        
        nb_e = self.options['nb_e']
        
        self.add_input('engine_mass',
                        val = 0.0,
                        units = 'kg',
                        desc = 'mass of one engine')
        
        self.add_input('thrust_frame_mass',
                        val = 0.0,
                        units = 'kg',
                        desc ='')
        
        self.add_input('M_FT',
                        val = 0.0,
                        units = 'kg',
                        desc = 'mass of fuel tank')
        
        self.add_input('M_OxT',
                        val = 0.0,
                        units = 'kg',
                        desc = 'mass of oxygen tank')
        
        self.add_input('M_inter_tank',
                        val = 0.0,
                        units = 'kg',
                        desc = 'mass of intertank')
        
        self.add_input('M_TPS_OxT',
                        val=0.0,
                        units = 'kg',
                        desc = 'mass of thermal protection oxygen tank')
        
        self.add_input('tvc_mass',
                        val = 0.0,
                        units = 'kg',
                        desc = 'mass of tvc. Hydraulic type')
        
        self.add_input('M_avio',
                        val = 0.0,
                        units = 'kg',
                        desc = '')
        
        self.add_input('M_EPS',
                        val = 0.0,
                        units = 'kg',
                        desc = '')
        
        self.add_input('mass_interstage',
                        val = 0.0,
                        units = 'kg',
                        desc = '')
        
        # --------------------------------------------------------------------
        
        self.add_output('Dry_mass_stage_1',
                        val = 0.0,
                        units = 'kg',
                        desc = '')
        
        
        # --------------------------------------------------------------------
        
        self.declare_partials(of = 'Dry_mass_stage_1', wrt = 'engine_mass', val = nb_e)
        self.declare_partials(of = 'Dry_mass_stage_1', wrt = 'thrust_frame_mass', val = 1.0)
        self.declare_partials(of = 'Dry_mass_stage_1', wrt = 'M_FT', val = 1.0)
        self.declare_partials(of = 'Dry_mass_stage_1', wrt = 'M_OxT', val = 1.0)
        self.declare_partials(of = 'Dry_mass_stage_1', wrt = 'M_inter_tank', val = 1.0)
        self.declare_partials(of = 'Dry_mass_stage_1', wrt = 'M_TPS_OxT', val = 1.0)
        self.declare_partials(of = 'Dry_mass_stage_1', wrt = 'tvc_mass', val = nb_e)
        self.declare_partials(of = 'Dry_mass_stage_1', wrt = 'M_avio', val = 1.0)
        self.declare_partials(of = 'Dry_mass_stage_1', wrt = 'M_EPS', val = 1.0)
        self.declare_partials(of = 'Dry_mass_stage_1', wrt = 'mass_interstage', val =1.0)
        
    def compute(self, inputs, outputs):
        
        nb_e      = self.options['nb_e']
        mass_aux_1 = self.options['mass_aux_1']
        
        outputs['Dry_mass_stage_1'] = inputs['engine_mass'] * nb_e +\
                                      inputs['thrust_frame_mass'] +\
                                      inputs['M_FT'] +\
                                      inputs['M_OxT'] +\
                                      inputs['M_inter_tank'] +\
                                      inputs['M_TPS_OxT'] +\
                                      inputs['tvc_mass'] * nb_e +\
                                      inputs['M_avio'] +\
                                      inputs['M_EPS'] +\
                                      inputs['mass_interstage'] +\
                                      mass_aux_1 + self.options['uncertainty_dry_mass_1']

class Tank_volume(om.ExplicitComponent):
    
    "This class calculates the volume of the oxidizer and fuel tanks based on the mass of propellants and o/f ratio"
    
    def initialize(self):
        
        self.options.declare('mu_F',
                              types = float,
                              desc ='fuel density (kg/m**3)')
        
        self.options.declare('mu_LOX',
                              types = float,
                              desc ='liquid oxygen density (kg/m**3)')
        
    def setup(self):
        
        self.add_input('o_f',
                        val = 0.0,
                        units = None,
                        desc='mass ratio oxidizer / fuel')
        
        self.add_input('mp',
                        val = 0.0,
                        units = 'kg',
                        desc='mass of propellants')
        
        # -------------------------------------------------
        
        self.add_output('V_F',
                        val =0.0,
                        units = 'm**3',
                        desc = 'volume of fuel')
        
        self.add_output('V_LOX',
                        val =0.0,
                        units = 'm**3',
                        desc = 'volume of liquid oxygen')
        
        # --------------------------------------------------
        
        self.declare_partials( of = 'V_F', wrt = 'o_f')
        self.declare_partials( of = 'V_F', wrt = 'mp')
        
        self.declare_partials( of = 'V_LOX', wrt = 'o_f')
        self.declare_partials( of = 'V_LOX', wrt = 'mp')
        
    def compute(self, inputs, outputs):
        
        mu_F = self.options['mu_F']
        mu_LOX = self.options['mu_LOX']
        
        o_f = inputs['o_f']
        mp = inputs['mp']
        
        outputs['V_F'] = mp / (mu_F * (o_f +1))
        outputs['V_LOX'] = o_f * mp / (mu_LOX * (1 + o_f) )
        
    def compute_partials(self, inputs, jacobian):
        
        mu_F = self.options['mu_F']
        mu_LOX = self.options['mu_LOX']
        
        o_f = inputs['o_f']
        mp = inputs['mp']
        
        jacobian['V_F', 'mp']  = 1 / (mu_F * (o_f +1))
        jacobian['V_F', 'o_f']  = mp * mu_F * -1 / (mu_F*(o_f + 1))**2
        
        jacobian['V_LOX', 'mp'] = o_f * 1 / (mu_LOX * (1 + o_f) )
        jacobian['V_LOX', 'o_f'] = mp / (mu_LOX * (1 + o_f) ) + (o_f * mp) * -mu_LOX / (mu_LOX * (1 + o_f))**2


class SingleEngineThrust(om.ExplicitComponent):
    
    "This class calculates the thrust of one engine by dividing the total thrust by the number of engines"
    
    def initialize(self):
        
        self.options.declare('nb_e', types = int, desc = 'number of engines')
    
    def setup(self):
        
        nb_e = self.options['nb_e']
        
        self.add_input('thrust',
                       val = 0.0,
                       units = 'N',
                       desc = 'launchers total thrust at vacuum')

        self.add_output('thrust_single',
                        val = 0.0,
                        units = 'N',
                        desc = 'single engine thrust at vacuum')
        
        self.declare_partials(of = 'thrust_single', wrt = 'thrust', val = 1/nb_e)
        
    def compute(self, inputs, outputs):
        
        nb_e = self.options['nb_e']
        
        outputs['thrust_single'] = inputs['thrust'] / nb_e
        

class Sizing(om.ExplicitComponent):
    
    "This class calculates the surface of the oxygen and fuel tanks. Exterior surface and total length of the vehicle"
    
    def initialize(self):
        self.options.declare('mu_F',
                              types = float,
                              desc ='fuel density (kg/m**3)')
        
        self.options.declare('mu_LOX',
                              types = float,
                              desc ='liquid oxygen density (kg/m**3)')
        
        self.options.declare('p',
                              types = float,
                              default = 1.6075,
                              desc = 'parameter of the Knud Thomsen formula for the Surface area of an ellipsoid')
        
    def setup(self):
        
        self.add_input('mp',
                        val = 0.0,
                        units = 'kg',
                        desc='mass of propellants')
        
        self.add_input('o_f',
                        val = 0.0,
                        units = None,
                        desc='mass ratio oxidizer / fuel')
        
        self.add_input('D_stage_1',
                        val = 0.0,
                        units = 'm',
                        desc = 'stage diameter')
        
        # --------------------------------------------------
        
        self.add_output('S_OX',
                        val = 0.0,
                        units = 'm**2',
                        desc = 'surface of LOx tanks')
        
        self.add_output('S_F',
                        val = 0.0,
                        units = 'm**2',
                        desc = 'surface of fuel tanks')
        
        self.add_output('S_totale',
                        val = 0.0,
                        units = 'm**2',
                        desc = 'total surface')
        
        self.add_output('S_dome',
                        val = 0.0,
                        units = 'm**2',
                        desc = 'surface of the tank dome')
        
        self.add_output('S_exterieur',
                        val = 0.0,
                        units = 'm**2',
                        desc = '')
        
        self.add_output('L_total',
                        val = 0.0,
                        units = 'm',
                        desc = 'total length')
        
        # -------------- partials ------------------------------
        
        self.declare_partials(of = 'S_OX', wrt = 'mp')
        self.declare_partials(of = 'S_OX', wrt = 'o_f')
        self.declare_partials(of = 'S_OX', wrt = 'D_stage_1')
        
        self.declare_partials(of = 'S_F', wrt = 'mp')
        self.declare_partials(of = 'S_F', wrt = 'o_f')
        self.declare_partials(of = 'S_F', wrt = 'D_stage_1')
        
        self.declare_partials(of = 'S_totale', wrt = 'mp')
        self.declare_partials(of = 'S_totale', wrt = 'o_f')
        self.declare_partials(of = 'S_totale', wrt = 'D_stage_1')
        
        self.declare_partials(of = 'S_dome', wrt = 'mp')
        self.declare_partials(of = 'S_dome', wrt = 'o_f')
        self.declare_partials(of = 'S_dome', wrt = 'D_stage_1')
        
        self.declare_partials(of = 'S_exterieur', wrt = 'mp')
        self.declare_partials(of = 'S_exterieur', wrt = 'o_f')
        self.declare_partials(of = 'S_exterieur', wrt = 'D_stage_1')
        
        self.declare_partials(of = 'L_total', wrt = 'mp')
        self.declare_partials(of = 'L_total', wrt = 'o_f')
        self.declare_partials(of = 'L_total', wrt = 'D_stage_1')
        
        
    def compute(self, inputs, outputs):

        mu_F   = self.options['mu_F']      
        mu_LOX = self.options['mu_LOX']
        p      = self.options['p']
        
        mp  = inputs['mp']
        o_f = inputs['o_f']
        D   = inputs['D_stage_1']
        
        # Volume calculations
        M_F = mp /(1+o_f)
        M_OX = mp - M_F
 	
        #Volume of LOX and fuel
        V_OX             = M_OX / mu_LOX
        V_F              = M_F / mu_F
       	# V_tot            = V_F+V_OX #volume total
       	h_dome           = 0.3*D #height of the dome part of the tanks
       	volume_domes     = 4.0/3.0 * np.pi * (D/2)**2*h_dome #volume of the tank domes
       	S_dome           = (4.0*np.pi* (( (D/2.0)**(2.0*p) + 2* (D/2.0)**(p) * h_dome**p )/3.0)**(1./p)) #surface of the tank dome. Knud Thomsen's formula for The Surface Area of an Ellipsoid.
       	volume_virole_OX = V_OX - volume_domes #volume of the cylindrical part of the tanks
       	L_virole_OX      = volume_virole_OX / np.pi /(D/2)**2 #height of the cylindrical part of the tanks
       	S_OX             = L_virole_OX *(D/2)**2 * np.pi + S_dome #surface of LOX tank
       	volume_virole_F  = V_F - volume_domes #volume fuel cylindrical tank part
       	L_virole_F       = volume_virole_F / np.pi /(D/2)**2 #height fuel cylindrical tank part
       	S_F              = L_virole_F *(D/2.)**2 * np.pi + S_dome #surface fuel tank
       	S_totale         = S_OX+S_F #total surface
       	S_exterieur      = 2* np.pi*(L_virole_OX+L_virole_F+4*h_dome+0.5)*(D/2.)    
       	L_total          = L_virole_OX + L_virole_F + 4*h_dome + 0.5 #total length
        
        
        outputs['S_OX']        = S_OX
        outputs['S_F']         = S_F
        outputs['S_totale']    = S_totale
        outputs['S_dome']      = S_dome
        outputs['S_exterieur'] = S_exterieur
        outputs['L_total']     = L_total
        
        
    def compute_partials(self, inputs, jacobian):
        
        mu_F   = self.options['mu_F']      
        mu_LOX = self.options['mu_LOX']
        p      = self.options['p']
        
        mp  = inputs['mp']
        o_f = inputs['o_f']
        D   = inputs['D_stage_1']
    
        jacobian['S_OX', 'mp']               = o_f/(mu_LOX*(o_f + 1))
        jacobian['S_OX', 'o_f']              = mp/(mu_LOX*(o_f + 1)**2)
        jacobian['S_OX', 'D_stage_1']        = np.pi*(-3*D**3 + 80*(2**(-2*p)*20**(-p)*D**(2*p)*(2**(2*p + 1)*3**p + 20**p)/3)**(1/p))/(10*D)
        
        jacobian['S_F', 'mp']                = 1/(mu_F*(o_f + 1))
        jacobian['S_F', 'o_f']               = -mp/(mu_F*(o_f + 1)**2)
        jacobian['S_F', 'D_stage_1']         = np.pi*(-3*D**3 + 80*(2**(-2*p)*20**(-p)*D**(2*p)*(2**(2*p + 1)*3**p + 20**p)/3)**(1/p))/(10*D)
		
        jacobian['S_totale', 'mp']           = (mu_F*o_f + mu_LOX)/(mu_F*mu_LOX*(o_f + 1))
        jacobian['S_totale', 'o_f']          = mp*(mu_F - mu_LOX)/(mu_F*mu_LOX*(o_f + 1)**2)
        jacobian['S_totale', 'D_stage_1']    = np.pi*(-3*D**3 + 80*(2**(-2*p)*20**(-p)*D**(2*p)*(2**(2*p + 1)*3**p + 20**p)/3)**(1/p))/(5*D)
        
        jacobian['S_dome', 'mp']             = 0
        jacobian['S_dome', 'o_f']            = 0 
        jacobian['S_dome', 'D_stage_1']      = 8*np.pi*(D**(2*p)*(2*20**(-p)*3**p + 2**(-2*p))/3)**(1/p)/D
        
        jacobian['S_exterieur', 'mp']        = 4*(mu_F*o_f + mu_LOX)/(D*mu_F*mu_LOX*(o_f + 1))
        jacobian['S_exterieur', 'o_f']       = 4*mp*(mu_F - mu_LOX)/(D*mu_F*mu_LOX*(o_f + 1)**2)
        jacobian['S_exterieur', 'D_stage_1'] = (8*np.pi*D**3*mu_F*mu_LOX*o_f + 8*np.pi*D**3*mu_F*mu_LOX + 5*np.pi*D**2*mu_F*mu_LOX*o_f + 5*np.pi*D**2*mu_F*mu_LOX - 40*mp*mu_F*o_f - 40*mp*mu_LOX)/(10*D**2*mu_F*mu_LOX*(o_f + 1))
        
        jacobian['L_total', 'mp']            = 4*(mu_F*o_f + mu_LOX)/(np.pi*D**2*mu_F*mu_LOX*(o_f + 1))
        jacobian['L_total', 'o_f']           = 4*mp*(mu_F - mu_LOX)/(np.pi*D**2*mu_F*mu_LOX*(o_f + 1)**2)
        jacobian['L_total', 'D_stage_1']     = 2*(np.pi*D**3*mu_F*mu_LOX*o_f + np.pi*D**3*mu_F*mu_LOX - 20*mp*mu_F*o_f - 20*mp*mu_LOX)/(5*np.pi*D**3*mu_F*mu_LOX*(o_f + 1))

	
class Engine_mass(om.ExplicitComponent):
    
    " Taken from FELIN and based on Castellini"
    "This class computes the mass of a single engine for a Cryo-Storable propellants and Gas generator cycle"
    " It accounts for the mass of Gas-generator(s) & turbopump(s), Valves and piping, Injector and igniter, Thrust chamber"
    
    def initialize(self):
        
        self.options.declare('a',
                              types = float,
                              default = 3.75407e03,
                              desc = '')
        
        self.options.declare('b',
                              types = float,
                              default = 7.05627e-02,
                              desc = '')
        
        self.options.declare('c',
                              types = float,
                              default = -8.8479e03,
                              desc = '')
        
    def setup(self):
        
        self.add_input('thrust_single',
                        val = 0.0,
                        units = 'N',
                        desc = 'thrust in vacuum for one engine')
        
        self.add_output('engine_mass',
                        val = 0.0,
                        units = 'kg',
                        desc = 'mass of one engine')
        
        self.declare_partials(of = 'engine_mass', wrt = 'thrust_single')
        
    def compute(self, inputs, outputs):
        
        a = self.options['a']
        b = self.options['b']
        c = self. options['c']
        
        T = inputs['thrust_single']
        
        outputs['engine_mass'] = a*(T)**b+c
        
    def compute_partials(self, inputs, jacobian):
        a = self.options['a']
        b = self.options['b']
        # c = self. options['c']
        
        T = inputs['thrust_single']
        
        jacobian['engine_mass', 'thrust_single'] = a*b*T**(b-1)
    
        
class Thrust_frame_mass(om.ExplicitComponent):
    
    " This class computes the thrust frame mass using aluminum"
    
    def initialize(self):
        
        self.options.declare('k_SM',
                              default = 1.0,
                              types = float,
                              desc = 'default value of 1 is for Al as material')
        
        self.options.declare('nb_e', types=int,
                              desc='number of engines')
        
        self.options.declare('SSM_TF_1st_stage', 
                             types = float, 
                             desc ='structural safety margin',
                             default = 1.25)
        
    def setup(self):
        
        self.add_input('thrust_single',
                        val = 0.0,
                        units = 'kN',
                        desc = 'thrust of one engine in kN')
        
        self.add_input('engine_mass',
                        val = 0.0,
                        units = 'kg',
                        desc = 'mass of one engine')
        
        self.add_input('n_ax_max',
                        val = 0.0,
                        units = None,
                        desc = 'max axial load factor')
        
        # ----------------------------------------------------------
        self.add_output('thrust_frame_mass',
                        val = 0.0,
                        units = 'kg',
                        desc ='')
        
        self.declare_partials(of = 'thrust_frame_mass', wrt = 'thrust_single')
        self.declare_partials(of = 'thrust_frame_mass', wrt = 'engine_mass')
        self.declare_partials(of = 'thrust_frame_mass', wrt = 'n_ax_max')
        
    def compute(self, inputs, outputs):
        
        T        = inputs['thrust_single']
        M_eng    = inputs['engine_mass']
        n_ax_max = inputs['n_ax_max']
        
        SSM      = self.options['SSM_TF_1st_stage']
        
        k_SM = self.options['k_SM']
        N_eng = self.options['nb_e']
        
        
        outputs['thrust_frame_mass'] = (0.013*N_eng**0.795*(224.81*T)**0.579+0.01*N_eng*(M_eng/(0.45))**0.717)*0.45*(1.5*SSM*n_ax_max*9.80665)*k_SM
        if outputs['thrust_frame_mass'] == np.NaN:
            print('Problem in thrust_frame_mass. thrust :' + str(T).l_just(15) + 'M_eng:' + str(M_eng).l_just(15) + 'n_ax_max:' + str( n_ax_max))
        
        
    def compute_partials(self, inputs, jacobian):
        
        T        = inputs['thrust_single']
        M_eng    = inputs['engine_mass']
        n_ax_max = inputs['n_ax_max']
        
        SSM      = self.options['SSM_TF_1st_stage']
        
        k_SM = self.options['k_SM']
        N_eng = self.options['nb_e']
        
        jacobian['thrust_frame_mass', 'thrust_single']  = 1.14590235422899*N_eng**0.795*SSM*T**(-0.421)*k_SM*n_ax_max
        jacobian['thrust_frame_mass', 'engine_mass']    = 0.0841376216909171*M_eng**(-0.283)*N_eng*SSM*k_SM*n_ax_max
        jacobian['thrust_frame_mass', 'n_ax_max']       = SSM*k_SM*(0.117346752706997*M_eng**0.717*N_eng + 1.97910596585319*N_eng**0.795*T**0.579)
        
    
class Tank_mass(om.ExplicitComponent):
    
    "This class calculates the mass of the tanks, thermal protection of Lox tank and mass of intertank"
    
    def initialize(self):
        
        self.options.declare('k1',
                              default = 1.15,
                              types = float,
                              desc = 'material constant')
        
        self.options.declare('k2',
                              default = 1.0,
                              types = float,
                              desc = 'intertank configuration - space between tank domes')
        
        self.options.declare('k3',
                              default = 1.3,
                              types = float,
                              desc = 'vertical integration')
        
        self.options.declare('k4_ref',
                              default = 5.76404,
                              types = float,
                              desc = '')
        
        self.options.declare('k5_ref',
                              default = 1.29134,
                              types = float,
                              desc = '')

        self.options.declare('k6_ref',
                              default = 2.7862,
                              types = float,
                              desc = '')    
        
        self.options.declare('SSM',
                              default = 1.25,
                              types = float,
                              desc = 'Structural - Safety Margin')
        
        self.options.declare('k_ins_Ox',
                              default =  0.9765,
                              types = float,
                              desc = 'Mass thermal protection')
        
        self.options.declare('M_TPS_FT',
                              default = 0.0,
                              types = float,
                              desc = 'For cryo/storable propulsion type')
        
        self.options.declare('k_1',
                              default = 5.4015,
                              types = float,
                              desc = 'intertank cnfiguration - lower stage')
        
        self.options.declare('k_2',
                              default = 0.5169,
                              types = float,
                              desc = 'intertank cnfiguration - lower stage')
        
        self.options.declare('k_it',
                              default = 0.3,
                              types = float,
                              desc = 'intertank cnfiguration - lower stage')
        
        self.options.declare('k_SM',
                              default = 1.0,
                              types = float,
                              desc = 'type structure = Al')
        
        self.options.declare('P_tanks_Ox',
                              types = float,
                              desc = '')
        
        self.options.declare('P_tanks_F',
                              types = float,
                              desc = '')
    
        
    def setup(self):
        
        self.add_input('P_dyn_max',
                        val = 0.0,
                        units='Pa',
                        desc = 'maximum dynamic pressure')
        
        self.add_input('n_ax_max',
                        val = 0.0,
                        units = None,
                        desc = 'maximum axial load factor')
        
        self.add_input('V_F',
                        val = 0.0,
                        units = 'm**3',
                        desc = 'volume of fuel tank')
        
        self.add_input('V_LOX',
                        val = 0.0,
                        units = 'm**3',
                        desc = 'volume of oxygen tank')
        
        self.add_input('D_stage_1',
                        val = 0.0,
                        units = 'm',
                        desc = 'stage diameter')
        
        self.add_input('S_OX',
                        val = 0.0,
                        units = 'm**2',
                        desc = 'surface of LOx tanks')
        
        # ---------------------------------------------
        
        self.add_output('M_FT',
                        val = 0.0,
                        units = 'kg',
                        desc = 'mass of fuel tank')
        
        self.add_output('M_OxT',
                        val = 0.0,
                        units = 'kg',
                        desc = 'mass of oxygen tank')
        
        self.add_output('M_inter_tank',
                        val = 0.0,
                        units = 'kg',
                        desc = 'mass of intertank')
        
        self.add_output('M_TPS_OxT',
                        val=0.0,
                        units = 'kg',
                        desc = 'mass of thermal protection oxygen tank')
        
        # ---------------------------------------
        
        self.declare_partials(of = 'M_FT', wrt = 'P_dyn_max')
        self.declare_partials(of = 'M_FT', wrt = 'n_ax_max')
        self.declare_partials(of = 'M_FT', wrt = 'V_F')
        
        self.declare_partials(of = 'M_OxT', wrt = 'P_dyn_max')
        self.declare_partials(of = 'M_OxT', wrt = 'n_ax_max')
        self.declare_partials(of = 'M_OxT', wrt = 'V_LOX')
        
        self.declare_partials(of = 'M_inter_tank', wrt = 'D_stage_1')
        
        self.declare_partials(of = 'M_TPS_OxT', wrt = 'S_OX')
        
    def compute(self, inputs, outputs):
        
        k1        = self.options['k1']
        k2        = self.options['k2']
        k3        = self.options['k3']
        k4_ref    = self.options['k4_ref']
        k5_ref    = self.options['k5_ref']
        k6_ref    = self.options['k6_ref']   
        SSM       = self.options['SSM']
        k_ins_Ox  = self.options['k_ins_Ox']
        # M_TPS_FT  = self.options['M_TPS_FT']
        k_1       = self.options['k_1']
        k_2       = self.options['k_2']
        k_it      = self.options['k_it']
        k_SM      = self.options['k_SM']
        P_tanks_Ox = self.options['P_tanks_Ox']
        P_tanks_F  = self.options['P_tanks_F']
        
        P_dyn_max = np.max(inputs['P_dyn_max'])
        
        n_ax_max  = inputs['n_ax_max']
        V_F       = inputs['V_F']
        V_LOX     = inputs['V_LOX']
        D         = inputs['D_stage_1']
        S_OX      = inputs['S_OX']
        
        
        k4=P_dyn_max**0.16/k4_ref
        k5=(SSM*n_ax_max)**0.15/k5_ref
        k6_Ox=1.3012+1.4359*10**-6*P_tanks_Ox/k6_ref
        k6_F=1.3012+1.4359*10**-6*P_tanks_F/k6_ref
                
        ### Mass Fuel tank
        M_FT=k1*k2*k3*k4*k5*k6_F*((V_F*35.315)*0.4856+800)*0.4536
        # Mass Ox tank
        M_OxT = k1*k2*k3*k4*k5*k6_Ox*((V_LOX*35.315)*0.4856+700)*0.4536
        
        # Mass of thermal protection
        M_TPS_OxT=k_ins_Ox*S_OX
        
        
        #
        l_it = 2*k_it * D+0.5
        M_inter_tank = k_SM*k_1*D*np.pi*l_it*(D*3.2808)**k_2
        
        outputs['M_FT']         = M_FT
        outputs['M_OxT']        = M_OxT
        outputs['M_inter_tank'] = M_inter_tank
        outputs['M_TPS_OxT']    = M_TPS_OxT
        
    def compute_partials(self, inputs, jacobian):
        
        k1        = self.options['k1']
        k2        = self.options['k2']
        k3        = self.options['k3']
        k4_ref    = self.options['k4_ref']
        k5_ref    = self.options['k5_ref']
        k6_ref    = self.options['k6_ref']   
        SSM       = self.options['SSM']
        k_ins_Ox  = self.options['k_ins_Ox']
        # M_TPS_FT  = self.options['M_TPS_FT']
        k_1       = self.options['k_1']
        k_2       = self.options['k_2']
        k_it      = self.options['k_it']
        k_SM      = self.options['k_SM']
        P_tanks_Ox = self.options['P_tanks_Ox']
        P_tanks_F  = self.options['P_tanks_F']
        
        P_dyn_max = inputs['P_dyn_max']
        n_ax_max  = inputs['n_ax_max']
        V_F       = inputs['V_F']
        V_LOX     = inputs['V_LOX']
        D         = inputs['D_stage_1']
        # S_OX      = inputs['S_OX']
        
        jacobian[ 'M_FT', 'P_dyn_max']           = 0.072576*P_dyn_max**(-0.84)*k1*k2*k3*(SSM*n_ax_max)**0.15*(1.4359e-6*P_tanks_F + 1.3012*k6_ref)*(17.148964*V_F + 800)/(k4_ref*k5_ref*k6_ref)
        jacobian[ 'M_FT', 'n_ax_max']            = 0.06804*P_dyn_max**0.16*k1*k2*k3*(SSM*n_ax_max)**0.15*(1.4359e-6*P_tanks_F + 1.3012*k6_ref)*(17.148964*V_F + 800)/(k4_ref*k5_ref*k6_ref*n_ax_max)
        jacobian[ 'M_FT', 'V_F']                 = 7.7787700704*P_dyn_max**0.16*k1*k2*k3*(SSM*n_ax_max)**0.15*(1.4359e-6*P_tanks_F + 1.3012*k6_ref)/(k4_ref*k5_ref*k6_ref)
        
        jacobian[ 'M_OxT',  'P_dyn_max']         = 0.072576*P_dyn_max**(-0.84)*k1*k2*k3*(SSM*n_ax_max)**0.15*(1.4359e-6*P_tanks_Ox + 1.3012*k6_ref)*(17.148964*V_LOX + 700)/(k4_ref*k5_ref*k6_ref)
        jacobian[ 'M_OxT',  'n_ax_max']          = 0.06804*P_dyn_max**0.16*k1*k2*k3*(SSM*n_ax_max)**0.15*(1.4359e-6*P_tanks_Ox + 1.3012*k6_ref)*(17.148964*V_LOX + 700)/(k4_ref*k5_ref*k6_ref*n_ax_max)
        jacobian[ 'M_OxT',  'V_LOX']             = 7.7787700704*P_dyn_max**0.16*k1*k2*k3*(SSM*n_ax_max)**0.15*(1.4359e-6*P_tanks_Ox + 1.3012*k6_ref)/(k4_ref*k5_ref*k6_ref)
        
        jacobian[ 'M_inter_tank',  'D_stage_1']  = np.pi*k_1*k_SM*(3.2808*D)**k_2*(4*D*k_it + 1.0*k_2*(2*D*k_it + 0.5) + 0.5)
        
        jacobian[ 'M_TPS_OxT',  'S_OX']          = k_ins_Ox
        
        
class TVC_mass(om.ExplicitComponent):
    
    " This class calculates the mass of the trhust vectoring control of hydraulic type"
    def setup(self):
        
        self.add_input('thrust_single',
                        val = 0.0,
                        units = 'N',
                        desc = 'thrust of one engine')
        
        self.add_output('tvc_mass',
                        val = 0.0,
                        units = 'kg',
                        desc = 'mass of tvc. Hydraulic type')
        
        self.declare_partials(of = 'tvc_mass', wrt = 'thrust_single', val = 0.1976*10**-3)
        
    def compute(self, inputs, outputs):
        
        T = inputs['thrust_single']
        
        outputs['tvc_mass'] = 0.1976*(T*10**-3)+20.922
        
		
class EPS_avio_mass(om.ExplicitComponent):
    
    "This class calculates the mass of avionics and Electrical and power supply system"
    
    def initialize(self):
        
        self.options.declare('K_RL',
                              default = 1.3,
                              types = float,
                              desc ='Redundancy level. Default for level 3')
        
    def setup(self):
        
        self.add_input('S_exterieur',
                        val = 0.0,
                        units = 'm**2',
                        desc = '')
        
        self.add_output('M_avio',
                        val = 0.0,
                        units = 'kg',
                        desc = '')
        
        self.add_output('M_EPS',
                        val = 0.0,
                        units = 'kg',
                        desc = '')
        
        self.declare_partials(of = 'M_avio', wrt = 'S_exterieur')
        
        self.declare_partials(of = 'M_EPS', wrt = 'S_exterieur')
        
    def compute(self, inputs, outputs):
        
        K_RL = self.options['K_RL']
        
        S_exterieur = inputs['S_exterieur']
        
        outputs['M_avio'] = K_RL*(246.76 + 1.3183*S_exterieur)
        outputs['M_EPS']  = K_RL*0.405*outputs['M_avio']
        
    def compute_partials(self, inputs, jacobian):
        
        K_RL = self.options['K_RL']
        
        # S_exterieur = inputs['S_exterieur']
        
        jacobian['M_avio', 'S_exterieur'] = K_RL * 1.3183
        
        jacobian['M_EPS', 'S_exterieur'] = K_RL**2 * 0.405 * 1.3183
        
class Interstage_mass(om.ExplicitComponent):
    
    "This class calculates the mass of the lower interstage"
    
    def initialize(self):
        
        self.options.declare('K_SM',
                              default = 1.0,
                              types = float,
                              desc = 'Safety coefficient')
        
        self.options.declare('k_1_1',
                              default = 7.7165,
                              types = float,
                              desc = '')
        
        self.options.declare('k_2_1',
                              default = 0.4856,
                              types = float,
                              desc = '')
        
        self.options.declare('S_interstage',
                              types = float,
                              desc = 'surface area of interstage')
        
        # **************  Mind that if S_interstage is 0, the mass of the interstage will be zero *****
        
    def setup(self):
        
        self.add_input('D_stage_1',
                        val = 0.0,
                        units = 'm',
                        desc ='')
        
        self.add_input('D_stage_2',
                        val = 0.0,
                        units = 'm',
                        desc = '')
        
        self.add_output('mass_interstage',
                        val = 0.0,
                        units = 'kg',
                        desc = '')
        
        self.declare_partials(of = 'mass_interstage', wrt = 'D_stage_1')
        self.declare_partials(of = 'mass_interstage', wrt = 'D_stage_2')
        
    def compute(self, inputs, outputs):
        
        K_SM = self.options['K_SM']
        k_1_1 = self.options['k_1_1']
        k_2_1 = self.options['k_2_1']
        S_interstage = self.options['S_interstage']
        
        D_inf = inputs['D_stage_1']
        D_up  = inputs['D_stage_2']
        
        D=(D_inf+D_up)/2
        
        outputs['mass_interstage'] = K_SM * k_1_1 * S_interstage * (D*3.2808)**k_2_1 
        
    def compute_partials(self, inputs, jacobian):
        
        K_SM = self.options['K_SM']
        k_1_1 = self.options['k_1_1']
        k_2_1 = self.options['k_2_1']
        S_interstage = self.options['S_interstage']
        
        D_inf = inputs['D_stage_1']
        D_up  = inputs['D_stage_2']
        
        D=(D_inf+D_up)/2
        
        jacobian['mass_interstage', 'D_stage_1']    = K_SM * k_1_1 * S_interstage * k_2_1 * (D*3.2808)**(k_2_1 - 1) * 3.2808 * 0.5
        jacobian['mass_interstage', 'D_stage_2']  = K_SM * k_1_1 * S_interstage * k_2_1 * (D*3.2808)**(k_2_1 - 1) * 3.2808 * 0.5
        