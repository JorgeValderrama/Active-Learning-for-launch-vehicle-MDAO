# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:34:44 2020

This class uses Tsiolkovsky's equation to calculate the mass of the vehicle
at the circularized orbit.

@author: jorge
"""

import numpy as np

import openmdao.api as om

class Final_mass(om.ExplicitComponent):
    
    def initialize(self):
    
        self.options.declare('g0', types=float,
                              desc='gravity at r0 (m/s**2)')
        
        
        self.options.declare('num_nodes', types=int, 
                             desc='Number of nodes to be evaluated in the RHS')
        
    def setup(self):
        
        nn     = self.options['num_nodes']
        
        self.add_input('m_0',
                       val = np.zeros(nn),
                       units = 'kg',
                       desc = 'initial mass of Hohmann circularization')
        
        self.add_input('delta_v2',
                       val = np.zeros(nn),
                       units = 'm/s',
                       desc = 'delta v necessary for circularization')
        
        self.add_input('Isp',
                        val = 0.0,
                        desc='specific impulse',
                        units = 's')
        
        # ----------------------------------------------------------------
        
        self.add_output('m_final',
                        val =np.zeros(nn),
                        units = 'kg',
                        desc = 'final mass at circularized orbit. Equivalent to ms_2 + md ')
        
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials( of = 'm_final', wrt = 'm_0', rows = ar, cols = ar)
        self.declare_partials( of = 'm_final', wrt = 'delta_v2', rows = ar, cols = ar)
        self.declare_partials( of = 'm_final', wrt = 'Isp')
        
    def compute(self, inputs, outputs):
        
        m_0     = inputs['m_0']
        dv      = inputs['delta_v2']
        Isp     = inputs['Isp']
        
        g0   = self.options['g0']
        
        outputs['m_final'] = m_0 / ( np.exp( dv / (Isp*g0) ) )
        
    def compute_partials(self, inputs, jacobian):
        
        m_0 = inputs['m_0']
        dv  = inputs['delta_v2']
        Isp = inputs['Isp']
        
        g0   = self.options['g0']
        
        jacobian['m_final', 'm_0']       = 1 / ( np.exp( dv / (Isp*g0) ) )
        
        jacobian['m_final', 'delta_v2']  =  - (m_0/(Isp*g0)) * np.exp(-dv / (Isp*g0))
        
        jacobian['m_final', 'Isp']       = dv*m_0*np.exp(-dv/(Isp*g0))/(Isp**2*g0)