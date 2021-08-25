# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:35:02 2020

@author: jorge
"""

import openmdao.api as om

class ConstraintsPropellants(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('mplf', types = float)
    
    def setup(self):
        
        self.add_input('mp_1_propulsion',
                       val   = 0.0,
                       units = 'kg',
                       desc  = 'mass of propellants from propulsion module. First stage.')
        
        self.add_input('mp_2_propulsion',
                       val   = 0.0,
                       units = 'kg',
                       desc  = 'mass of propellants from propulsion module. Second stage.')
        
        self.add_input(name  = 'me_a',
                       val   = 0.0,
                       units = 'kg',
                       desc  = 'me_a read from state. empty mass for first stage flight with fairing')
        
        self.add_input(name  = 'm_final',
                        val   = 0.0,
                        units = 'kg',
                        desc  = 'final mass at circularized orbit. Equivalent to ms_2 + md')
        
        self.add_input(name  = 'mf_b',
                       val   = 0.0,
                       units = 'kg',
                       desc  = 'mf_b read from state. full mass for second flight with fairing')
        
        self.add_input(name  = 'mf_a',
                       val   = 0.0,
                       units = 'kg',
                       desc  = 'me_a read from state. full mass for first stage flight with fairing')
        
        self.add_output(name  = 'residual_mp_1',
                        val   = 0.0,
                        units = 'kg',
                        desc  = 'residual of mass of propellants operation')
        
        self.add_output(name  = 'residual_mp_2',
                        val   = 0.0,
                        units = 'kg',
                        desc  = 'residual of mass of propellants operation')
        
        # declare partials
        
        self.declare_partials(of = 'residual_mp_1', wrt = 'mp_1_propulsion', val = 1.0)
        self.declare_partials(of = 'residual_mp_1', wrt = 'me_a', val = 1.0)
        self.declare_partials(of = 'residual_mp_1', wrt = 'mf_a', val = -1.0)
        
        self.declare_partials(of = 'residual_mp_2', wrt = 'mp_2_propulsion', val =1.0)
        self.declare_partials(of = 'residual_mp_2', wrt = 'mf_b', val = -1.0)
        self.declare_partials(of = 'residual_mp_2', wrt = 'm_final', val = 1.0)
        
    def compute(self, inputs, outputs):
        
        mp_1_propulsion = inputs['mp_1_propulsion']
        me_a            = inputs['me_a']
        mf_a            = inputs['mf_a']
        
        mp_2_propulsion = inputs['mp_2_propulsion']
        mf_b            = inputs['mf_b']
        m_final         = inputs['m_final']
        
        mplf            = self.options['mplf']
        
        
        outputs['residual_mp_1'] = mp_1_propulsion - mf_a + me_a
        outputs['residual_mp_2'] = mp_2_propulsion - mf_b + m_final + mplf
        
        # print('--------------------------')
        # print(mp_2_propulsion)
        # print(mf_b - m_final - mplf)