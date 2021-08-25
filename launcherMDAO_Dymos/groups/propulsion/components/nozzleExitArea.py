# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:59:17 2020

This class calculates hte the nozzle exit area of engine "Ae"
and the total nozzle exit area of all the engines "Ae_t".

@author: jorge
"""

import openmdao.api as om

class NozzleExitArea(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('nb_e', types=int,
                              desc='number of engines')
    
    def setup(self):
        
        
        self.add_input('epsilon',
                       val = 0,
                       desc='expansion ratio',
                       units = None)
        
        self.add_input('At',
                        val = 0,
                        desc='throat area of one engine',
                        units = 'm**2')
        
        self.add_output('Ae',
                        val = 0,
                        desc='nozzle exit area of one engine',
                        units = 'm**2')
        
        self.add_output('Ae_t',
                        val = 0,
                        desc='total nozzle exit area',
                        units = 'm**2')
        
        # Setup partials 
        self.declare_partials(of = 'Ae', wrt = 'At')
        self.declare_partials(of = 'Ae', wrt = 'epsilon')
        
        self.declare_partials(of = 'Ae_t', wrt = 'At')
        self.declare_partials(of = 'Ae_t', wrt = 'epsilon')
        
    def compute(self, inputs, outputs):
        
        At   = inputs['At']
        epsilon = inputs['epsilon']
        
        nb_e = self.options['nb_e']
        
        outputs['Ae']   = At * epsilon
        outputs['Ae_t'] = At * epsilon * nb_e
        
        # print( 'Ae (m^2)'.ljust(20) + str( outputs['Ae']) )
        # print ('epsilon ()'.ljust(20) + str( epsilon))
        
    def compute_partials(self, inputs, jacobian):
        
        At      = inputs['At']
        epsilon = inputs['epsilon']
        
        nb_e    = self.options['nb_e']
        
        jacobian['Ae', 'At']        = epsilon
        jacobian['Ae', 'epsilon']   = At
        
        jacobian['Ae_t', 'At']      = epsilon * nb_e
        jacobian['Ae_t', 'epsilon'] = At * nb_e
        
    