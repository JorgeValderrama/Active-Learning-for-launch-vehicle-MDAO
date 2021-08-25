# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 22:44:54 2021

This class calculates the time rate of the dynamic pressure q_dyn.

This helps to find the maximum dynamic pressure and ensures that it happens at a known node.
Thus, the maximum dynamic pressure (where qdot = 0) is the final boundary constraint of the  first part
of the gravity turn phase.

@author: jorge
"""
import numpy as np
import openmdao.api as om


class QDot(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        
        nn     = self.options['num_nodes']
        
        self.add_input('vdot',
                        val = np.zeros(nn),
                        desc = 'derivative of speed w.r.t. time',
                        units = 'm/s**2')
        
        self.add_input('rdot',
                       val = np.zeros(nn),
                       desc = 'derivative of radius w.r.t. time',
                       units = 'm/s')
        
        self.add_input('v',
                       val = np.zeros(nn),
                       desc = 'speed',
                       units = 'm/s')
        
        self.add_input('rho', 
                        val=np.zeros(nn), 
                        units='kg/m**3')
        
        self.add_input('d_rho_wrt_h', 
                        val=np.zeros(nn), 
                        units='kg/m**4',
                        desc = 'partial of density wrt to height')
        
        # ----------------------------------------------------------
        
        self.add_output('qDot', 
                        val = np.zeros(nn),
                        units = 'kg/m/s**3',
                        desc = 'partial of dynamic pressure w.r.t. time')
        
        # -------------------------------------------------------------------
        
        ar = np.arange(nn)
         
        self.declare_partials( of = 'qDot', wrt = 'vdot', rows = ar, cols = ar)
        self.declare_partials( of = 'qDot', wrt = 'rdot', rows = ar, cols = ar)
        self.declare_partials( of = 'qDot', wrt = 'v', rows = ar, cols = ar)
        self.declare_partials( of = 'qDot', wrt = 'rho', rows = ar, cols = ar)
        self.declare_partials( of = 'qDot', wrt = 'd_rho_wrt_h', rows = ar, cols = ar)
        
    def compute(self, inputs, outputs):
        
        vdot = inputs['vdot']
        rdot = inputs['rdot']
        v    = inputs['v']
        rho  = inputs['rho']
        drdh = inputs['d_rho_wrt_h']
        
        outputs['qDot'] = 1/2 * drdh * rdot * v**2  + rho * v * vdot
        
        
    def compute_partials (self, inputs, jacobian):
        
        vdot = inputs['vdot']
        rdot = inputs['rdot']
        v    = inputs['v']
        rho  = inputs['rho']
        drdh = inputs['d_rho_wrt_h']
        
        jacobian['qDot', 'vdot']        = rho * v
        jacobian['qDot', 'rdot']        = 1/2 * drdh * v**2
        jacobian['qDot', 'v']           = drdh * rdot * v + rho * vdot
        jacobian['qDot', 'rho']         = v * vdot
        jacobian['qDot', 'd_rho_wrt_h'] = 1/2 * rdot * v**2
        
        
        