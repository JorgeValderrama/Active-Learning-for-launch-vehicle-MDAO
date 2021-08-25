# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:15:05 2020

The bilinear tangent law used during the exoatmospheric phase is a function of
the phase normalized time. As the phase is splitted into 3 parts (a, b and c) due to jettison events, it's 
necessary to use the following classes to asseses the time variables.

As partial derivatives take constant values of 1. These are defined with the "val" parameter
of the "declare_partials" method to avoid creating a new method  to compute the partials.
@author: jorge
"""
import numpy as np
import openmdao.api as om

class Time_exoatmos_a(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('num_nodes', types=int)
        
    def setup(self):
        
        
        self.add_input('phase_duration_a',
                       val = 0,
                       desc = 'Duration of exoatmos_a phase',
                       units = 's')
        
        self.add_input('phase_duration_b',
                       val = 0,
                       desc = 'Duration of exoatmos_b phase',
                       units = 's')
        
        # self.add_input('phase_duration_c',
        #                val = 0,
        #                desc = 'Duration of exoatmos_b phase',
        #                units = 's')
        
        self.add_output('phase_duration_total',
                        val = 0,
                        desc = 'Total duration of exoatmos phase',
                        units = 's')
        
        # declare partials
        
        self.declare_partials(of='phase_duration_total', wrt='phase_duration_a', val=1)
        self.declare_partials(of='phase_duration_total', wrt='phase_duration_b', val=1)
        # self.declare_partials(of='phase_duration_total', wrt='phase_duration_c', val=1)
        
    def compute(self, inputs, outputs):
        
        outputs['phase_duration_total'] = inputs['phase_duration_a'] + inputs['phase_duration_b'] 
        
class Time_exoatmos_b(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('num_nodes', types=int)
        
    def setup(self):
        
        nn = self.options['num_nodes']
        
        self.add_input('phase_duration_a',
                       val = 0,
                       desc = 'Duration of exoatmos_a phase',
                       units = 's')
        
        self.add_input('phase_duration_b',
                       val = 0,
                       desc = 'Duration of exoatmos_b phase',
                       units = 's')
        
        # self.add_input('phase_duration_c',
        #                val = 0,
        #                desc = 'Duration of exoatmos_c phase',
        #                units = 's')
        
        self.add_input('phase_time_b',
                        val = np.zeros(nn),
                        desc = "Phase time of exoatmos_b phase to be shifted",
                        units = 's')
        
        self.add_output('phase_duration_total',
                        val = 0,
                        desc = 'Total duration of exoatmos phase',
                        units = 's')
        
        self.add_output('phase_time_b_shifted',
                        val = np.zeros(nn),
                        desc = 'Shifted time of exoatmos_b phase',
                        units = 's')
        
        # declare partials
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of='phase_duration_total', wrt='phase_duration_a', val=1.0) # attention! non declared size
        self.declare_partials(of='phase_duration_total', wrt='phase_duration_b', val=1.0) # attention! non declared size
        # self.declare_partials(of='phase_duration_total', wrt='phase_duration_c', val=1.0) # attention! non declared size
        
        self.declare_partials(of='phase_time_b_shifted', wrt='phase_time_b', rows= ar, cols= ar, val=1.0)     # attention! non declared size
        self.declare_partials(of='phase_time_b_shifted', wrt='phase_duration_a', val=1.0) # attention! non declared size
        
    def compute(self, inputs, outputs):
        
        outputs['phase_duration_total'] = inputs['phase_duration_a'] + inputs['phase_duration_b'] 
        
        outputs['phase_time_b_shifted'] = inputs['phase_duration_a'] + inputs['phase_time_b']
        
    def compute_partials(self, inputs, jacobian):
        
        jacobian['phase_duration_total','phase_duration_a'] = 1.0
        jacobian['phase_duration_total','phase_duration_b'] = 1.0
        # jacobian['phase_duration_total','phase_duration_c'] = 1.0
        
        jacobian['phase_time_b_shifted','phase_duration_a'] = 1.0
        jacobian['phase_time_b_shifted','phase_time_b']     = 1.0
        
        
# class Time_exoatmos_c(om.ExplicitComponent):
    
#     def initialize(self):
        
#         self.options.declare('num_nodes', types=int)
        
#     def setup(self):
        
#         nn = self.options['num_nodes']
        
#         self.add_input('phase_duration_a',
#                        val = 0,
#                        desc = 'Duration of exoatmos_a phase',
#                        units = 's')
        
#         self.add_input('phase_duration_b',
#                        val = 0,
#                        desc = 'Duration of exoatmos_b phase',
#                        units = 's')
        
#         self.add_input('phase_duration_c',
#                        val = 0,
#                        desc = 'Duration of exoatmos_c phase',
#                        units = 's')
        
#         self.add_input('phase_time_c',
#                         val = np.zeros(nn),
#                         desc = "Phase time of exoatmos_c phase to be shifted",
#                         units = 's')
        
#         self.add_output('phase_duration_total',
#                         val = 0,
#                         desc = 'Total duration of exoatmos phase',
#                         units = 's')
        
#         self.add_output('phase_time_c_shifted',
#                         val = np.zeros(nn),
#                         desc = 'Shifted time of exoatmos_c phase',
#                         units = 's')
        
#         # declare partials
#         ar = np.arange(self.options['num_nodes'])
        
#         self.declare_partials(of='phase_duration_total', wrt='phase_duration_a', val=1.0) # attention! non declared size
#         self.declare_partials(of='phase_duration_total', wrt='phase_duration_b', val=1.0) # attention! non declared size
#         self.declare_partials(of='phase_duration_total', wrt='phase_duration_c', val=1.0) # attention! non declared size
        
#         self.declare_partials(of='phase_time_c_shifted', wrt='phase_time_c', rows= ar, cols= ar, val=1.0)     # attention! non declared size
#         self.declare_partials(of='phase_time_c_shifted', wrt='phase_duration_b', val=1.0) # attention! non declared size
#         self.declare_partials(of='phase_time_c_shifted', wrt='phase_duration_a', val=1.0) # attention! non declared size
        
#     def compute(self, inputs, outputs):
        
#         outputs['phase_duration_total'] = inputs['phase_duration_a'] + inputs['phase_duration_b'] + inputs['phase_duration_c']
        
#         outputs['phase_time_c_shifted'] = inputs['phase_duration_a'] + inputs['phase_duration_b'] + inputs['phase_time_c']
        
        