# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:20:43 2020

ms_1, ms_2 = structural mass of first and second stage
mp_1, mp_2 = mass of propellants of first and second stage
mplf       = mass of payload fairing
md         = mass of payload

mf_a       = ms_1 + ms_2 + mplf + md + mp_1 + mp_2    = full  mass for first stage flight with fairing
me_a       = ms_1 + ms_2 + mplf + md + 0.0  + mp_2    = empty mass for first stage flight with fairing

mf_b       = 0.0  + ms_2 + mplf + md + 0.0  + mp_2    = full  mass for second stage flight with fairing
mi_b       = 0.0  + ms_2 + mplf + md + 0.0  + ????    = mass for second stage flight at the instant before fairing jettison

mi_c       = 0.0  + ms_2 + 0.0  + md + 0.0  + ????    = mass for second stage flight at the instant after fairing jettison
m_final    = 0.0  + ms_2 + 0.0  + md + 0.0  + 0.0     = mass at the cricularized orbit. 

@author: jorge
"""


import openmdao.api as om

class ConstraintsJettison(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('mplf', types = float)
        self.options.declare('md', types = float)
                 
    def setup(self):
        
        # self.add_input(name  = 'me_a',
        #                val   = 0.0,
        #                units = 'kg',
        #                desc  = 'me_a read from state. empty mass for first stage flight with fairing')
        
        # self.add_input(name  = 'mf_b',
        #                val   = 0.0,
        #                units = 'kg',
        #                desc  = 'mf_b read from state. full mass for second stage fligh with fairing')
        
        # self.add_input(name  = 'mi_b',
        #                val   = 0.0,
        #                units = 'kg',
        #                desc  = 'mi_b read from state. mass at the end of second stage flight with fairing')
        
        # self.add_input(name  = 'mi_c',
        #                val   = 0.0,
        #                units = 'kg',
        #                desc  = 'mi_c read from state. mass at the beggining of second stage flight without fairing')
        
        
        self.add_input(name  = 'massjettison_first_stage',
                        val   = 0.0,
                        units = 'kg',
                        desc  = 'mass being jetissoned during first stage jettison')
        
        self.add_input(name  = 'massjettison_plf',
                        val   = 0.0,
                        units = 'kg',
                        desc  = 'mass being jetissoned during payload fairing jettison')
        
        self.add_input(name  = 'ms_1',
                       val   = 0.0,
                       units = 'kg',
                       desc  = 'structural mass of first stage')
        
        self.add_input(name  = 'ms_2',
                        val   = 0.0,
                        units = 'kg',
                        desc  = 'structural mass of second stage')
        
        self.add_input(name  = 'm_final',
                        val   = 0.0,
                        units = 'kg',
                        desc  = 'final mass at circularized orbit. Equivalent to ms_2 + md')
        
        # -------------------------------------------------------------------------------------
        self.add_output(name   = 'residual_ms_1',
                        val    = 0.0,
                        units  = 'kg',
                        desc   = 'residual of jettison of first stage structural mass')
        
        self.add_output(name   = 'residual_mplf',
                        val    = 0.0,
                        units  = 'kg',
                        desc   = 'residual of jettison of payload fairing mass')
        
        self.add_output(name   = 'residual_m_final',
                        val    = 0.0,
                        units  = 'kg',
                        desc   = 'residual of final mass of launcher')
        
        # self.declare_partials(of = 'residual_ms_1', wrt = 'me_a', val =  1.0)
        # self.declare_partials(of = 'residual_ms_1', wrt = 'mf_b', val = -1.0)
        self.declare_partials(of = 'residual_ms_1', wrt = 'massjettison_first_stage', val = 1.0)
        self.declare_partials(of = 'residual_ms_1', wrt = 'ms_1', val = -1.0)
        
        # self.declare_partials(of = 'residual_mplf', wrt = 'mi_b', val =  1.0)
        # self.declare_partials(of = 'residual_mplf', wrt = 'mi_c', val = -1.0)
        self.declare_partials(of = 'residual_mplf', wrt = 'massjettison_plf', val = 1.0 )
        
        self.declare_partials(of = 'residual_m_final', wrt = 'm_final', val = 1.0)
        self.declare_partials(of = 'residual_m_final', wrt = 'ms_2', val = -1.0)
        
    def compute(self, inputs, outputs):
        
        # outputs['residual_ms_1']    = inputs['me_a'] - inputs['mf_b'] - inputs['ms_1']
        outputs['residual_ms_1']    = inputs['massjettison_first_stage'] - inputs['ms_1']
        
        # outputs['residual_mplf']    = inputs['mi_b'] - inputs['mi_c'] - self.options['mplf']
        outputs['residual_mplf']    = inputs['massjettison_plf'] - self.options['mplf']
        
        outputs['residual_m_final'] = inputs['m_final'] - inputs['ms_2'] - self.options['md']