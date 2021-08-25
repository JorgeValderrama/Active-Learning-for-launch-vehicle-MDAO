# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:48:32 2020

This script calls the atmospheric table generator to build tables of data.
Such tables are then interpolated in order to have access to continuous funtions
and their derivatives.
The tables are generated and interpolated only 1 time per optimization procedure.
The interpolator is used iteratively through out the optimization.

Tewari's atmospheric model is valid for heights [0 , 2000) km
Tewari - ATMOSPHERIC AND SPACE FLIGHT DYNAMICS

@author: jorge
"""

# Standard Atmosphere Derived from 1976 and 1962 U.S. Standard Atmospheres. Reference Tewari
from .Usatm_table_generation import atmosphere
from collections import namedtuple

import numpy as np
from scipy.interpolate import Akima1DInterpolator as Akima

import openmdao.api as om
import matplotlib.pyplot as plt

USatmData = namedtuple('USatmData', ['T', 'P' , 'rho'])

# Define sampling space to call the atmosphere table generation function 
# Tewari's atmospheric model  is valid for heights [0 , 2000) km
height_initial  = 0       # m
height_final    = 1999e3  # m
height_sampling = 1000    # m
numberSamples = int(((height_final -  height_initial )/height_sampling ) + 1)

# define vector containing every altitude at which the atmospheric model is to be evaluated
# to generate the table
USatmData.alt = np.linspace(height_initial , height_final, numberSamples) 

#  define variables and allocate memory
USatmData.T   = np.zeros(numberSamples);
USatmData.P   = np.zeros(numberSamples);
USatmData.rho = np.zeros(numberSamples);

# iteratively call atmosphere function to fill table
for i in range(numberSamples):
    _, USatmData.rho[i], USatmData.P[i], USatmData.T[i] = atmosphere(USatmData.alt[i],0)

# interpolate the date from the tables
T_interp = Akima(USatmData.alt, USatmData.T)
P_interp = Akima(USatmData.alt, USatmData.P)
rho_interp = Akima(USatmData.alt, USatmData.rho)

# create function to evaluate derivatives of the interpolator
T_interp_deriv    = T_interp.derivative(1)
P_interp_deriv    = P_interp.derivative(1)
rho_interp_deriv  = rho_interp.derivative(1)
rho_interp_deriv2 = rho_interp.derivative(2)

# % plot with height_final = 50e3 to check continuity
# plt.figure()   
# plt.plot(USatmData.alt/1e3,USatmData.T,'o')
# plt.figure()
# plt.plot(USatmData.alt/1e3,USatmData.rho,'o')
# plt.figure()
# plt.plot(USatmData.alt/1e3,rho_interp_deriv(USatmData.alt),'o')
# plt.figure()
# plt.plot(USatmData.alt/1e3,rho_interp_deriv2(USatmData.alt),'o')
# plt.figure()
# plt.plot(USatmData.alt/1e3,USatmData.P/1e3,'o')

class USatm(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

        gamma = 1.4  # Ratio of specific heats
        gas_c = 287  # Gas constant J/(kg K)
        self._K = gamma * gas_c

    def setup(self):
        
        nn = self.options['num_nodes']
        
        self.add_input('h', 
                       val=np.zeros(nn), 
                       units='m',
                       desc = ' altitude')

        self.add_output('P_a', 
                        val=np.zeros(nn), 
                        units='Pa',
                        desc = 'local atmospheric pressure')
        
        self.add_output('rho', 
                        val=np.zeros(nn), 
                        units='kg/m**3',
                        desc = 'local air density')
        
        self.add_output('d_rho_wrt_h', 
                        val=np.zeros(nn), 
                        units='kg/m**4',
                        desc = 'partial derivative of rho w.r.t. h')
        
        self.add_output('sos', 
                        val=np.zeros(nn), 
                        units='m/s',
                        desc = 'speed of sound')

        ar = np.arange(nn)
        
        self.declare_partials( of='rho',         wrt='h',rows=ar, cols=ar)
        self.declare_partials( of='d_rho_wrt_h', wrt='h',rows=ar, cols=ar)
        self.declare_partials( of='sos',         wrt='h',rows=ar, cols=ar)
        self.declare_partials( of='P_a',         wrt='h',rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        T = T_interp(inputs['h'], extrapolate=True)
        outputs['P_a']         = P_interp(inputs['h'], extrapolate=True)
        outputs['rho']         = rho_interp(inputs['h'], extrapolate=True)
        outputs['d_rho_wrt_h'] = rho_interp_deriv(inputs['h'], extrapolate=True)
        outputs['sos']         = np.sqrt(self._K * T)

    def compute_partials(self, inputs, partials):
        H   = inputs['h']
        T   = T_interp(H, extrapolate=True)
        T_p = T_interp_deriv(H, extrapolate=True)
        
        partials['P_a', 'h']         = P_interp_deriv(H, extrapolate=True)
        partials['rho', 'h']         = rho_interp_deriv(H, extrapolate=True)
        partials['d_rho_wrt_h', 'h'] = rho_interp_deriv2(H, extrapolate=True)
        partials['sos', 'h']         = 0.5/np.sqrt(self._K * T) * T_p * self._K
