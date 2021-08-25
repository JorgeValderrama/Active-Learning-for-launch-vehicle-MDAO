# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:35:18 2020

@author: jorge
"""

import openmdao.api as om

class ConstraintsLoadFactor(om.ExplicitComponent):
    
    def setup(self):
        
        self.add_input('max_n_f_1_p',
                        val = 0.0,
                        units = None,
                        desc = 'maximum load factor for the frist stage. taken from propulsion')
        
        # self.add_input('max_n_f_2_p',
        #                 val = 0.0,
        #                 units = None,
        #                 desc = 'maximum load factor for the second stage. taken from propulsion')
        
        self.add_input('max_n_f_1_t',
                        val = 0.0,
                        units = None,
                        desc = 'maximum load factor for the frist stage. taken from trajectory')
        
        # self.add_input('max_n_f_2_t',
        #                 val = 0.0,
        #                 units = None,
        #                 desc = 'maximum load factor for the second stage. taken from trajectory')
        
        self.add_output('residual_max_n_f_1',
                        val = 0.0,
                        units = None,
                        desc = 'residual of operation of max_n_f_1')
        
        # self.add_output('residual_max_n_f_2',
        #                 val = 0.0,
        #                 units = None,
        #                 desc = 'residual of operation of max_n_f_2')
        
        # declare partials --------------------------------------------
        
        self.declare_partials(of = 'residual_max_n_f_1', wrt = 'max_n_f_1_p', val = 1)
        self.declare_partials(of = 'residual_max_n_f_1', wrt = 'max_n_f_1_t', val = -1)
        
        # self.declare_partials(of = 'residual_max_n_f_2', wrt = 'max_n_f_2_p', val = 1)
        # self.declare_partials(of = 'residual_max_n_f_2', wrt = 'max_n_f_2_t', val = -1)
        
    def compute(self, inputs, outputs):
        
        max_n_f_1_p = inputs['max_n_f_1_p']
        # max_n_f_2_p = inputs['max_n_f_2_p']
        
        max_n_f_1_t = inputs['max_n_f_1_t']
        # max_n_f_2_t = inputs['max_n_f_2_t']
        
        outputs['residual_max_n_f_1'] = max_n_f_1_p - max_n_f_1_t
        
        # outputs['residual_max_n_f_2'] = max_n_f_2_p - max_n_f_2_t