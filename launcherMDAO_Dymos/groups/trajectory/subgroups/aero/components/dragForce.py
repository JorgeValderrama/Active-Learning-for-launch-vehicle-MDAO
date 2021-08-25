# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:34:29 2020

This class calculates the drag force along the trajectory.

@author: jorge
"""

import openmdao.api as om
import numpy as np

class DragForce(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('uncertainty_cd', types=float, desc='uncertainty about drag coefficient (-)')
        
        # self.options.declare('S', types=float,
        #                      desc='aerodynamic reference area (m**2)')
        
    def setup(self):
    
        nn = self.options['num_nodes']
        
        self.add_input('q_dyn',
                        val=np.zeros(nn),
                        desc='dynamic pressure',
                        units='Pa')
        
        self.add_input('Cd',
                        val=np.zeros(nn),
                        desc='Drag coefficient',
                        units=None)
        
        self.add_input('diameter',
                        val=0.0,
                        desc='diameter of stage',
                        units='m')
        
        self.add_output('Drag',
                        val=np.zeros(nn),
                        desc='drag force',
                        units = 'N')
        
        # Setup partials 
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of = 'Drag', wrt = 'q_dyn' , rows=ar, cols=ar)
        self.declare_partials(of = 'Drag', wrt = 'Cd' , rows=ar, cols=ar)
        self.declare_partials(of = 'Drag', wrt = 'diameter')
        
    def compute(self, inputs, outputs):
        q_dyn    = inputs['q_dyn']
        Cd       = inputs['Cd']
        diameter = inputs['diameter']
        # S     = self.options['S']
        
        S = np.pi/4 * diameter**2
        
        outputs['Drag'] = q_dyn * Cd* (1 + self.options['uncertainty_cd']) * S
        
        
    def compute_partials(self, inputs, jacobian):
        
        q_dyn    = inputs['q_dyn']
        Cd       = inputs['Cd']
        diameter = inputs['diameter']
        # S     = self.options['S']
        
        S = np.pi/4 * diameter**2
        
        jacobian['Drag','q_dyn'] = Cd* (1 + self.options['uncertainty_cd'])* S
        jacobian['Drag', 'Cd']   = q_dyn * S* (1 + self.options['uncertainty_cd'])
        
        jacobian['Drag', 'diameter']   = q_dyn * Cd* (1 + self.options['uncertainty_cd']) * np.pi/2 * diameter