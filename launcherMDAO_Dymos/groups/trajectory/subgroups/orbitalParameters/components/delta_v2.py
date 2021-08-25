# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:12:18 2020

This class uses the Vis-Viva equation to calculate the delta v
necessary to circularize the elliptical transfer orbit.

@author: jorge
"""

import numpy as np

import openmdao.api as om

class Delta_v2(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('mu', desc = 'mu=GMe (m**3/s**2)')
    
    def setup(self):
        
        nn     = self.options['num_nodes']
        
        self.add_input('ra',
                        val = np.zeros(nn),
                        desc = 'radius at apogee',
                        units = 'm')
        
        self.add_input('rp',
                        val = np.zeros(nn),
                        desc = 'radius at perigee',
                        units = 'm')
        
        # ------------------------------------------
        
        self.add_output('delta_v2',
                        val = np.zeros(nn),
                        desc = 'delta v necessary for orbit circularization',
                        units = 'm/s')
        
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of = 'delta_v2', wrt = 'ra', rows = ar, cols = ar)
        
        self.declare_partials(of = 'delta_v2', wrt = 'rp', rows = ar, cols = ar)
        
    def compute(self, inputs, outputs):
        
        ra = inputs['ra']
        rp = inputs['rp']
        
        mu = self.options['mu']
        
        
        outputs['delta_v2'] = - np.sqrt( (2 * mu * rp) / (ra * (ra+rp)) ) + np.sqrt(mu/ra)
        if np.NaN in outputs['delta_v2'] :
            print('problem with delta_v2. ra: ' + str(ra).l_just(15) + 'rp:' + str(rp) )
            
    def compute_partials(self, inputs, jacobian):

        ra = inputs['ra']
        rp = inputs['rp']
        
        mu = self.options['mu']
        
        jacobian['delta_v2','rp'] = -np.sqrt(2)*ra*np.sqrt(mu*rp/(ra*(ra + rp)))*(ra + rp)*(-mu*rp/(2*ra*(ra + rp)**2) + mu/(2*ra*(ra + rp)))/(mu*rp)
        jacobian['delta_v2','ra'] = -np.sqrt(mu/ra)/(2*ra) - np.sqrt(2)*ra*np.sqrt(mu*rp/(ra*(ra + rp)))*(ra + rp)*(-mu*rp/(2*ra*(ra + rp)**2) - mu*rp/(2*ra**2*(ra + rp)))/(mu*rp)
    