# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:59:17 2020


This group calculates the throat area of one engine.
It the divided the massflow rate by the number of engines.

@author: jorge
"""

import openmdao.api as om

class ThroatArea(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('nb_e', types=int,
                              desc='number of engines')
    
    def setup(self):
        
        self.add_input('cStar',
                       val = 0,
                       desc='characteristic velocity of the propellants',
                       units = 'm/s')
        
        self.add_input('mfr_max',
                       val = 0,
                       desc='maximum mass flow rate of all engines',
                       units = 'kg/s')
        
        self.add_input('P_c',
                       val = 0,
                       desc='pressure at the chamber',
                       units='Pa')
        
        self.add_output('At',
                        val = 0,
                        desc='throat area of one engine',
                        units = 'm**2')
        
        
        # Setup partials by JLVZ
        self.declare_partials(of = 'At', wrt = 'cStar')
        self.declare_partials(of = 'At', wrt = 'mfr_max')
        self.declare_partials(of = 'At', wrt = 'P_c')
        
    def compute(self, inputs, outputs):
        
        cStar   = inputs['cStar']
        mfr_max = inputs['mfr_max']
        P_c     = inputs['P_c']
        
        nb_e = self.options['nb_e']
        
        outputs['At']   = cStar * mfr_max / nb_e  / P_c
        
        # print( 'Ae (m^2)'.ljust(20) + str( outputs['Ae']) )
        # print ('epsilon ()'.ljust(20) + str( epsilon))
        
    def compute_partials(self, inputs, jacobian):
        
        cStar   = inputs['cStar']
        mfr_max = inputs['mfr_max']
        P_c     = inputs['P_c']
        
        nb_e = self.options['nb_e']
        
        jacobian['At', 'cStar']   = mfr_max / nb_e  / P_c
        jacobian['At', 'mfr_max'] = cStar  / P_c / nb_e
        jacobian['At', 'P_c']     = - cStar * mfr_max / nb_e / P_c**2

        
    