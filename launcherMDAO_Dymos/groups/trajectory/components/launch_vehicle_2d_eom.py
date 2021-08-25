# -*- coding: utf-8 -*-
"""
Created on 

This class calculates the EOM in 2D polar coordinates considering a rotating planet with angular speed "omega".
This is a set of 5 coupled ODEs.

@author: jorge
"""
import numpy as np

import openmdao.api as om

class LaunchVehicle2DEOM(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        
        self.options.declare('omega', types=float,
                             desc='Earth angular speed (rad/s)')
        
        self.options.declare('g0', types=float,
                              desc='gravity at r0 (m/s**2)')

    def setup(self):
        nn     = self.options['num_nodes']

        # Inputs by JLVZ
        self.add_input('r',
                       val = np.zeros(nn),
                       desc = 'radius',
                       units = 'm')
        
        self.add_input('lambda',
                       val = np.zeros(nn),
                       desc = 'longitude',
                       units = 'rad')
        
        self.add_input('v',
                       val = np.zeros(nn),
                       desc = 'speed',
                       units = 'm/s')
        
        self.add_input('phi',
                       val = np.zeros(nn),
                       desc = 'flight path angle',
                       units = 'rad')
        
        
        self.add_input('m',
                       val = np.zeros(nn),
                       desc = 'mass',
                       units = 'kg')
        
        self.add_input('thrust',
                       val= np.zeros(nn),
                       desc='thrust after throttle and losses are considered',
                       units='N')
        
        self.add_input('mfr',
                        val= np.zeros(nn),
                        desc= 'mass flow rate after throttle is considered',
                        units= 'kg/s')
        
        self.add_input('theta',
                       val = np.zeros(nn),
                       desc='pitch angle',
                       units='rad')
        
        self.add_input('g',
                       val = np.zeros(nn),
                       desc='gravity acceleration',
                       units='m/s**2')
        
        self.add_input('Drag',
                       val = np.zeros(nn),
                       desc='drag force',
                       units='N')
        
        # Outputs by JLVZ
        
        self.add_output('rdot',
                       val = np.zeros(nn),
                       desc = 'speed',
                       units = 'm/s')
        
        self.add_output('lambdadot',
                        val = np.zeros(nn),
                        desc = 'angular speed',
                        units = 'rad/s')
        
        self.add_output('vdot',
                        val = np.zeros(nn),
                        desc = 'acceleration magnitude',
                        units = 'm/s**2')
        
        self.add_output('phidot',
                        val = np.zeros(nn),
                        desc = 'flight path angle rate of change',
                        units = 'rad/s')
        
        self.add_output('mdot',
                        val = np.zeros(nn),
                        desc = 'mass rate of change',
                        units = 'kg/s')
        
        self.add_output('n_f',
                        val = np.zeros(nn),
                        desc = 'axial load factor',
                        units = None)


        # Setup partials by JLVZ
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of = 'rdot', wrt = 'v', rows=ar, cols=ar)
        self.declare_partials(of = 'rdot', wrt = 'phi', rows=ar, cols=ar)
        
        self.declare_partials(of = 'lambdadot', wrt = 'v', rows=ar, cols=ar)
        self.declare_partials(of = 'lambdadot', wrt = 'r', rows=ar, cols=ar)
        self.declare_partials(of = 'lambdadot', wrt = 'phi', rows=ar, cols=ar)
        
        self.declare_partials(of = 'mdot', wrt = 'mfr', rows=ar, cols=ar)
        
        self.declare_partials(of = 'vdot', wrt = 'thrust', rows=ar, cols=ar)
        self.declare_partials(of = 'vdot', wrt = 'phi', rows=ar, cols=ar)
        self.declare_partials(of = 'vdot', wrt = 'm', rows=ar, cols=ar)
        self.declare_partials(of = 'vdot', wrt = 'r', rows=ar, cols=ar)
        self.declare_partials(of = 'vdot', wrt = 'theta', rows=ar, cols=ar)
        self.declare_partials(of = 'vdot', wrt = 'g', rows=ar, cols=ar)
        self.declare_partials(of = 'vdot', wrt = 'Drag', rows=ar, cols=ar)
        
        self.declare_partials(of = 'phidot', wrt = 'thrust', rows=ar, cols=ar)
        self.declare_partials(of = 'phidot', wrt = 'phi', rows=ar, cols=ar)
        self.declare_partials(of = 'phidot', wrt = 'm', rows=ar, cols=ar)
        self.declare_partials(of = 'phidot', wrt = 'v', rows=ar, cols=ar)
        self.declare_partials(of = 'phidot', wrt = 'r', rows=ar, cols=ar)
        self.declare_partials(of = 'phidot', wrt = 'theta', rows=ar, cols=ar)
        self.declare_partials(of = 'phidot', wrt = 'g', rows=ar, cols=ar)
        
        self.declare_partials(of = 'n_f', wrt = 'thrust', rows=ar, cols=ar)
        self.declare_partials(of = 'n_f', wrt = 'Drag', rows=ar, cols=ar)
        self.declare_partials(of = 'n_f', wrt = 'theta', rows=ar, cols=ar)
        self.declare_partials(of = 'n_f', wrt = 'phi', rows=ar, cols=ar)
        self.declare_partials(of = 'n_f', wrt = 'm', rows=ar, cols=ar)
        
        
    def compute(self, inputs, outputs):
        
        r       = inputs['r']
        v       = inputs['v']
        phi     = inputs['phi']
        m       = inputs['m']
        thrust  = inputs['thrust']
        mfr     = inputs['mfr']
        theta   = inputs['theta']
        g       = inputs['g']
        D       = inputs['Drag']
                
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        cos_theta_phi = np.cos(theta-phi)
        sin_theta_phi = np.sin(theta-phi)
        
        omega = self.options['omega']
        g0    = self.options['g0']
        L = 0
        
        outputs['rdot']      = v * sin_phi
        outputs['lambdadot'] = v * cos_phi / r
        outputs['vdot']      = (-D + thrust * cos_theta_phi) / m + (omega**2 *r - g) * sin_phi
        outputs['phidot']    = L/(m*v) + (thrust * sin_theta_phi) / (m*v) + ((omega**2 *r - g)*cos_phi) / v + 2*omega + v*cos_phi / r 
        outputs['mdot']      = -mfr
        
        outputs['n_f']       = (thrust - D * cos_theta_phi) / (m * g0)

    def compute_partials(self, inputs, jacobian):
        
        r       = inputs['r']
        v       = inputs['v']
        phi     = inputs['phi']
        m       = inputs['m']
        thrust  = inputs['thrust']
        theta   = inputs['theta']
        g       = inputs['g']
        D       = inputs['Drag']
        
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        cos_theta_phi = np.cos(theta-phi)
        sin_theta_phi = np.sin(theta-phi)
        
        omega = self.options['omega']
        g0    = self.options['g0']
        L = 0
        
        jacobian['rdot', 'v']           = sin_phi
        jacobian['rdot', 'phi']         = v * cos_phi 
        
        jacobian['lambdadot', 'v']      = cos_phi / r
        jacobian['lambdadot', 'r']      = v * cos_phi * (-1/r**2)
        jacobian['lambdadot', 'phi']    = - v / r * sin_phi
        
        jacobian['vdot', 'thrust']      = cos_theta_phi / m
        jacobian['vdot', 'phi']         = thrust/m * sin_theta_phi + (omega**2 * r - g) * cos_phi
        jacobian['vdot', 'm']           = (-D + thrust * cos_theta_phi) * (-1/m**2)
        jacobian['vdot', 'r']           = omega**2 * sin_phi
        jacobian['vdot', 'theta']       = - (thrust / m ) * sin_theta_phi
        jacobian['vdot', 'g']           = - sin_phi
        jacobian['vdot', 'Drag']           = -1 / m
        
        jacobian['phidot', 'thrust']    = sin_theta_phi / (m*v)
        jacobian['phidot', 'phi']       = -thrust/(m*v) * cos_theta_phi - (omega**2 *r -g)/v * sin_phi - v/r * sin_phi
        jacobian['phidot', 'm']         = L/v * (-1/m**2) + thrust * sin_theta_phi / v * (-1/m**2)
        jacobian['phidot', 'v']         = L/m * (-1/v**2) + thrust * sin_theta_phi/m * (-1/v**2) + (omega**2 * r - g) * cos_phi * (-1/v**2) + cos_phi/r                                  
        jacobian['phidot', 'r']         = omega**2 / v * cos_phi + v * cos_phi * (-1/r**2)
        jacobian['phidot', 'theta']     = thrust / (m * v) * cos_theta_phi
        jacobian['phidot', 'g']         = - cos_phi/v
        
        jacobian['mdot', 'mfr']         = -1 
        
        jacobian['n_f', 'thrust']       = 1 / (m * g0)
        jacobian['n_f', 'Drag']         = (- cos_theta_phi) / (m * g0)
        jacobian['n_f', 'theta']        = D / (m * g0) * sin_theta_phi
        jacobian['n_f', 'phi']          = -D / (m * g0) * sin_theta_phi 
        jacobian['n_f', 'm']            = -(thrust - D * cos_theta_phi) / (m**2 * g0)
        
        