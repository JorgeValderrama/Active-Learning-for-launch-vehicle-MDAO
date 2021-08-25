# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:44:40 2020

This script intepolates data in 2D from Rocket CEA to obtain continuous functions and derivatives
for isentropic coef, flame temperature and molecular mass as functions of chamber pressure and mass ratio of 
oxidizer / fuel.

@author: jorge
"""

from numpy import genfromtxt
import numpy as np
from scipy.interpolate import RectBivariateSpline as interp

import openmdao.api as om
import os
dirname = os.path.dirname(__file__)

#%% 
# Step by step to generate data

#1. Go to: https://cearun.grc.nasa.gov/
#2. Choose Chemical Equilibrium Problem Types = rocket
#3. Enter low, high and interval values for pressure, also its units. This is equivalent to chamber pressure.
# be careful in CEA when using different inputs, I've got bad outputs outside of this range.
P_lv = 60
P_hv = 170
P_int = 10
P_units = 'bar'
#4. Choose fuel = RP-1
#5. Choose oxidizer = O2(L)
#6. Enter low, high and interval values for o_f
o_f_lv = 1.6
o_f_hv = 4.0
o_f_int = 0.2
#7. Choose Pc/Pe = 1. I'm not planning to use exit parameters so this value does not have any influence
#8. Choose What do you want to do upon clicking 'Submit'? = Tabulate results
#9. Select equilibrium adn frozen coompositions. Enter parameters in the folling order: 
#       gamfz, mwfz, tfz, p , ispfz
#10. Save input, output and tabulation files

#%%

# import chemistry data from rocket CEA csv file
rocket_cea = genfromtxt(os.path.join(dirname,'cea_table.csv'), delimiter='', encoding="utf8", skip_header=True)

# definition of vectors
P_c = np.arange(P_lv, P_hv + P_int, P_int)
o_f = np.arange(o_f_lv, o_f_hv + o_f_int, o_f_int)

rows = len(o_f)
cols = len(P_c)

# empty matrix to be filled with the values extracted from the csv output from rocket CEA
m_gamma_t = np.zeros((rows,cols))
m_tc      = np.zeros((rows,cols))
m_mc      = np.zeros((rows,cols)) 

# Our interest is gamma at the throat
# rocket cea first writes inputs at the chamber, followed by throat and finally the nozzle exit
# We want: 
# - gamma at the throat: shift = 1
# - tc at the chamber  : shift = 0
# - mc at the chamber  : shift = 0

# transform data from csv file into matrix form
# Frozen eq understimates performance by 1 - 4 %.  Refer to Sutton pag 167
for i in range(0, rows):
    for j in range(0, cols): 
        idx =  (i * cols  + j) * 3 
        m_gamma_t[i,j] = rocket_cea[idx + 1, 0]
        m_mc[i,j]      = rocket_cea[idx + 0, 1]
        m_tc[i,j]      = rocket_cea[idx + 0, 2]
        
        # print(idx, i, j)
# fit 2d interpolator for each variable      
# it is better to use the plotting function to analyze the effect of the smoothing factor for derivatives = 0,1,2     
# high frequency peaks in first derivative plots are obtained when using RectBivariateSpline for apparantly smooth functions
        
gamma_t_interp = interp( o_f, P_c, m_gamma_t) 
tc_interp      = interp( o_f, P_c, m_tc) 
mc_interp      = interp( o_f, P_c, m_mc) 

class Rocket_cea(om.ExplicitComponent):
    
        
    def setup(self):
        
        
        self.add_input('P_c',
                       val=0.0,
                       desc='pressure at the combustion chamber',
                       units= P_units)
        
        self.add_input('o_f',
                       val=0.0,
                       desc='mass ratio oxidizer/fuel',
                       units=None)
        
        
        self.add_output('gamma_t',
                        val=0.0,
                        desc='Isentropic coefficient at the throat',
                        units=None)
        
        self.add_output('tc',
                        val=0.0,
                        desc='flame temperature at the combustion chamber',
                        units='K')
        
        self.add_output('mc',
                        val=0.0,
                        desc='molecular mass at combustion chamber',
                        units='g/mol')
        
        # Setup partials 
        # ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of = 'gamma_t', wrt = 'P_c')
        self.declare_partials(of = 'gamma_t', wrt = 'o_f')
        
        self.declare_partials(of = 'tc',      wrt = 'P_c')
        self.declare_partials(of = 'tc',      wrt = 'o_f')
        
        self.declare_partials(of = 'mc',      wrt = 'P_c')
        self.declare_partials(of = 'mc',      wrt = 'o_f')
        
    def compute(self, inputs, outputs):
        
        P_c  = inputs['P_c']
        o_f = inputs['o_f']
        
        outputs['gamma_t'] = gamma_t_interp( o_f, P_c)[0][0]
        outputs['tc']      =       tc_interp(o_f, P_c)[0][0]
        outputs['mc']      =       mc_interp(o_f, P_c)[0][0]
        
        # print('------------------------------------------')
        # print('P_c (bar)'.ljust(20) + str(P_c))
        # print('o_f ()'.ljust(20) + str(o_f))
        # print('tc (K)'.ljust(20) + str(outputs['tc']))
        # print('gamma_t ()'.ljust(20) + str(outputs['gamma_t']))
        # print('mc (mol)'.ljust(20) + str(outputs['mc']))
        
    def compute_partials(self, inputs, jacobian):
        
        P_c  = inputs['P_c']
        o_f = inputs['o_f']
        
        jacobian['gamma_t', 'o_f']  = gamma_t_interp(o_f, P_c, dx=1, dy=0)
        jacobian['gamma_t', 'P_c']  = gamma_t_interp(o_f, P_c, dx=0, dy=1)

        jacobian['tc'   , 'o_f']    =    tc_interp(o_f, P_c, dx=1, dy=0)
        jacobian['tc'   , 'P_c']    =    tc_interp(o_f, P_c, dx=0, dy=1)
        
        jacobian['mc'   , 'o_f']    =    mc_interp(o_f, P_c, dx=1, dy=0)
        jacobian['mc'   , 'P_c']    =    mc_interp(o_f, P_c, dx=0, dy=1)

