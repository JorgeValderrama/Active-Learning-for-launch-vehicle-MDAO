# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:36:49 2020

Different classes to calculate the pitch guidance are defined in this script.
They calculate the pitch angle for the launcher ascent trajectory as in Castellini and Pagano.

@author: jorge
"""

import numpy as np
import openmdao.api as om

class Guidance_pitch_over_linear(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('num_nodes', types=int)
        
    def setup(self):
        
        nn = self.options['num_nodes']
        
        self.add_input('delta_theta',
                       val = np.zeros(nn),
                       desc = 'change in pitch angle during phase',
                       units = 'rad')
        
        self.add_input('phase_time',
                        val = np.zeros(nn),
                        desc = 'elapsed time of the phase. Starts at t_p = 0 ends at t_p = phase duration',
                        units = 's')
        
        self.add_input('phi',
                       val = np.zeros(nn),
                       desc = 'flight path angle',
                       units = 'rad')
        
        self.add_input('phase_duration',
                       val = 0,
                       desc = 'duration of phase',
                       units = 's'
                       )
        
        # outputs -------------
        self.add_output('theta',
                        val = np.zeros(nn),
                        desc='pitch angle',
                        units='rad'
                        )
        
        # declare partials
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of='theta', wrt='delta_theta', rows=ar, cols=ar)
        self.declare_partials(of='theta', wrt='phase_time', rows=ar, cols=ar)
        self.declare_partials(of='theta', wrt='phi', rows=ar, cols=ar)
        
        self.declare_partials(of='theta', wrt='phase_duration') ####### attention! non declared size ########
    
    def compute(self, inputs, outputs):

        delta_theta = inputs['delta_theta']
        phase_time  = inputs['phase_time']
        phi = inputs['phi']
        
        phase_duration = inputs['phase_duration']
        
        outputs['theta'] = phi - phase_time / phase_duration * delta_theta
        
    def compute_partials(self, inputs, jacobian):
        delta_theta = inputs['delta_theta']
        phase_time  = inputs['phase_time']
        
        # phase_duration = phase_time[-1]
        phase_duration = inputs['phase_duration']
        
        jacobian['theta','delta_theta']     = - phase_time / phase_duration
        jacobian['theta','phase_time']      = - delta_theta / phase_duration
        jacobian['theta','phi']             = 1
        
        jacobian['theta','phase_duration']  =  phase_time * delta_theta / phase_duration**2
        

class Guidance_pitch_over_exponential(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        
        nn = self.options['num_nodes']
        
        self.add_input('delta_theta',
                       val = np.zeros(nn),
                       desc = 'change in pitch angle during phase',
                       units = 'rad')
        
        self.add_input('phase_time',
                        val = np.zeros(nn),
                        desc = 'elapsed time of the phase. Starts at t_p = 0 ends at t_p = phase duration',
                        units = 's')
        
        self.add_input('phi',
                       val = np.zeros(nn),
                       desc = 'flight path angle',
                       units = 'rad')
        
        self.add_input('phase_duration',
                       val = 0,
                       desc = 'duration of phase',
                       units = 's'
                       )
        
        # outputs -------------
        self.add_output('theta',
                        val = np.zeros(nn),
                        desc='pitch angle',
                        units='rad'
                        )
        
        # declare partials
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of='theta', wrt='delta_theta', rows=ar, cols=ar)
        self.declare_partials(of='theta', wrt='phase_time', rows=ar, cols=ar)
        self.declare_partials(of='theta', wrt='phi', rows=ar, cols=ar)
        
        self.declare_partials(of='theta', wrt='phase_duration') ####### attention! non declared size ########
        
    def compute(self, inputs, outputs):
        delta_theta = inputs['delta_theta']
        phase_time  = inputs['phase_time']
        phi = inputs['phi']
        
        # phase_duration = phase_time[-1]
        phase_duration = inputs['phase_duration']
        
        outputs['theta'] = phi - (delta_theta * np.exp(-3 * phase_time / phase_duration))
        
    def compute_partials(self, inputs, jacobian):
        delta_theta = inputs['delta_theta']
        phase_time  = inputs['phase_time']
        
        # phase_duration = phase_time[-1]
        phase_duration = inputs['phase_duration']
        
        jacobian['theta','delta_theta']     = - np.exp(-3 * phase_time / phase_duration)
        jacobian['theta','phase_time']      = delta_theta * np.exp(-3 * phase_time / phase_duration) * (3/phase_duration)
        jacobian['theta','phi']             = 1
        
        jacobian['theta','phase_duration']  = - (delta_theta * np.exp(-3 * phase_time / phase_duration)) * (3 * phase_time / phase_duration**2)
        
        
class Guidance_gravity_turn(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('num_nodes', types=int)
        
    def setup(self):
        
        nn = self.options['num_nodes']
        
        self.add_input('phi',
                       val = np.zeros(nn),
                       desc = 'flight path angle',
                       units = 'rad')
        
        # outputs -------------
        self.add_output('theta',
                        val = np.zeros(nn),
                        desc='pitch angle',
                        units='rad'
                        )
        
        # declare partials
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of='theta', wrt='phi', rows=ar, cols=ar)
        
    def compute(self, inputs, outputs):
        
        phi = inputs['phi']
        
        outputs['theta'] = phi
        
    def compute_partials(self, inputs, jacobian):
        
        jacobian['theta','phi']  = 1
        

class Guidance_exoatmos(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('num_nodes', types=int)
        
    def setup(self):
        
        nn = self.options['num_nodes']
        
        self.add_input('theta_gt',
                       val = np.zeros(1),
                       desc = 'theta at the end of gravity turn',
                       units = 'rad'
                       )
        
        self.add_input('delta_theta',
                       val = np.zeros(nn),
                       desc = 'change in pitch from gravity turn to begining of exoatmos',
                       units = 'rad')
        
        self.add_input('phase_time',
                        val = np.zeros(nn),
                        desc = 'elapsed time of the phase. Starts at t_p = 0 ends at t_p = phase duration',
                        units = 's')
        
        self.add_input('theta_f',
                       val = np.zeros(nn),
                       desc = 'theta at the end of exoatmos',
                       units = 'rad')
        
        self.add_input('xi',
                       val = np.zeros(nn),
                       desc = 'parameter controlling shape of BLTL',
                       units = None)
        
        self.add_input('phase_duration',
                       val = 0,
                       desc = 'duration of phase',
                       units = 's'
                       )
        
        # outputs -------------
        self.add_output('theta',
                        val = np.zeros(nn),
                        desc='pitch angle',
                        units='rad'
                        )
        
        # declare partials
        ar = np.arange(self.options['num_nodes'])
        
        
        self.declare_partials(of='theta', wrt='theta_gt') ####### attention! non declared size ########
        self.declare_partials(of='theta', wrt='delta_theta', rows=ar, cols=ar)
        self.declare_partials(of='theta', wrt='phase_time', rows=ar, cols=ar)
        self.declare_partials(of='theta', wrt='theta_f', rows=ar, cols=ar)
        self.declare_partials(of='theta', wrt='xi', rows=ar, cols=ar)
        
        self.declare_partials(of='theta', wrt='phase_duration') ####### attention! non declared size ########
        
    def compute(self, inputs, outputs):
        theta_gt    = inputs['theta_gt'] # the first elemtn of phi corresponds to the final pitch angle of the gravity turn phase
        delta_theta = inputs['delta_theta']
        phase_time  = inputs['phase_time']
        theta_f     = inputs['theta_f']
        xi          = inputs['xi']
        
        phase_duration = inputs['phase_duration']
        
        a   = 100.0 # this parameter is constant in Castellini. It could be a good idea to try different values.
        
        num = a**xi * np.tan(theta_gt+delta_theta) + (np.tan(theta_f) - a**xi * np.tan(theta_gt+delta_theta)) * (phase_time / phase_duration)
        den = a**xi + ( 1 - a**xi) * (phase_time / phase_duration)
        outputs['theta'] = np.arctan ( num/den )
        
    def compute_partials(self, inputs, jacobian):
        
        theta_gt    = inputs['theta_gt']
        delta_theta = inputs['delta_theta']
        phase_time  = inputs['phase_time']
        theta_f     = inputs['theta_f']
        xi          = inputs['xi']
        
        phase_duration = inputs['phase_duration']
        
        a   = 100.0 # this parameter is constant in Castellini. It could be a good idea to try different values.
        
        num = a**xi * np.tan(theta_gt+delta_theta) + (np.tan(theta_f) - a**xi * np.tan(theta_gt+delta_theta)) * (phase_time / phase_duration)
        den = a**xi + ( 1 - a**xi) * (phase_time / phase_duration)
        
        jacobian['theta','theta_gt']       = 1 / (1 + (num/den)**2) * 1 / den * (a**xi*(np.tan(delta_theta + theta_gt)**2 + 1) - a**xi*phase_time*(np.tan(delta_theta + theta_gt)**2 + 1)/phase_duration)
        jacobian['theta','delta_theta']    = 1 / (1 + (num/den)**2) * 1 / den * (a**xi*(np.tan(delta_theta + theta_gt)**2 + 1) - a**xi*phase_time*(np.tan(delta_theta + theta_gt)**2 + 1)/phase_duration)
        jacobian['theta','phase_time']     = (-(1 - a**xi)*(a**xi*np.tan(delta_theta + theta_gt) + phase_time*(-a**xi*np.tan(delta_theta + theta_gt) + np.tan(theta_f))/phase_duration)/(phase_duration*(a**xi + phase_time*(1 - a**xi)/phase_duration)**2) + (-a**xi*np.tan(delta_theta + theta_gt) + np.tan(theta_f))/(phase_duration*(a**xi + phase_time*(1 - a**xi)/phase_duration)))/(1 + (a**xi*np.tan(delta_theta + theta_gt) + phase_time*(-a**xi*np.tan(delta_theta + theta_gt) + np.tan(theta_f))/phase_duration)**2/(a**xi + phase_time*(1 - a**xi)/phase_duration)**2)
        jacobian['theta','theta_f']        = 1 / (1 + (num/den)**2) * 1 / den * (phase_time*(np.tan(theta_f)**2 + 1)/phase_duration)
        jacobian['theta','xi']             = 1 / (1 + (num/den)**2) * ((a**xi*np.log(a)*np.tan(delta_theta + theta_gt) - a**xi*phase_time*np.log(a)*np.tan(delta_theta + theta_gt)/phase_duration)/(a**xi + phase_time*(1 - a**xi)/phase_duration) + (-a**xi*np.log(a) + a**xi*phase_time*np.log(a)/phase_duration)*(a**xi*np.tan(delta_theta + theta_gt) + phase_time*(-a**xi*np.tan(delta_theta + theta_gt) + np.tan(theta_f))/phase_duration)/(a**xi + phase_time*(1 - a**xi)/phase_duration)**2)
        jacobian['theta','phase_duration'] = (phase_time*(1 - a**xi)*(a**xi*np.tan(delta_theta + theta_gt) + phase_time*(-a**xi*np.tan(delta_theta + theta_gt) + np.tan(theta_f))/phase_duration)/(phase_duration**2*(a**xi + phase_time*(1 - a**xi)/phase_duration)**2) - phase_time*(-a**xi*np.tan(delta_theta + theta_gt) + np.tan(theta_f))/(phase_duration**2*(a**xi + phase_time*(1 - a**xi)/phase_duration)))/(1 + (a**xi*np.tan(delta_theta + theta_gt) + phase_time*(-a**xi*np.tan(delta_theta + theta_gt) + np.tan(theta_f))/phase_duration)**2/(a**xi + phase_time*(1 - a**xi)/phase_duration)**2)