# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 18:04:04 2020

Reference values taken from Sutton - Rocket propulsion Elements pag 180
# TABLE 5–5. Theoretical Chamber Performance of Liquid Rocket Propellant Combinations
Comparisson made for Frozen composition

@author: jorge
"""

import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from .groups.propulsion.propulsionStage import PropulsionStage


class TestPropulsion(unittest.TestCase):

    def test_value(self):
        
        p = om.Problem()
        p.model.add_subsystem(name='propulsion', subsys=PropulsionStage(g0 = 9.80665, 
                                                                        nb_e = 1, 
                                                                        eta_C_f = 1.0,
                                                                        eta_cStar = 1.0,
                                                                        Rmc = 8314.0, 
                                                                        P_a = 101325.0))
        
        external_params = p.model.add_subsystem('external_params', om.IndepVarComp())
         
        # --------------- Inouts from Sutton ----------------------
        external_params.add_output('P_c', val=1000.0, units='psi')
        external_params.add_output('P_e', val=14.7, units='psi')
        external_params.add_output('o_f', val=2.24, units=None)
        external_params.add_output('thrust', val=1e6, units='N')
        # ---------------------------------------------------------
        
        # external_params.add_output('P_c', val=1e7, units='Pa')
        # external_params.add_output('P_e', val=40530., units='Pa')
        # external_params.add_output('o_f', val=2.30691248, units=None)
        # external_params.add_output('thrust', val=1e6, units='N')
        
        p.model.connect('external_params.P_c', 'propulsion.P_c')
        p.model.connect('external_params.P_e', 'propulsion.P_e')
        p.model.connect('external_params.o_f', 'propulsion.o_f')
        p.model.connect('external_params.thrust', 'propulsion.thrust')
        p.setup()
        
        p.run_model()

        Ae       = p.get_val('propulsion.Ae')
        cStar    = p.get_val('propulsion.cStar')
        Isp      = p.get_val('propulsion.Isp')
        epsilon  = p.get_val('propulsion.epsilon')
        tc       = p.get_val('propulsion.chemistry.tc')
        mc       = p.get_val('propulsion.chemistry.mc')
        gamma_t  = p.get_val('propulsion.chemistry.gamma_t')
        
        print('Ae      = ' + str(Ae[0]))
        print('cStar   = ' + str(cStar[0])) 
        print('Isp     = ' + str(Isp[0]))
        print('epsilon = ' + str(epsilon[0]))
        print('tc      = ' + str(tc[0]))
        print('mc      = ' + str(mc[0]))
        print('gamma_t = ' + str(gamma_t[0]))

        
        # ---- Comparisson against Sutton values ---------------
        tol = 2e-2 # errors are below 4% when compared to Sutton
        
        assert_near_equal(mc, 21.9, tol)
        assert_near_equal(cStar, 1774, tol)
        assert_near_equal(tc, 3571, tol)
        assert_near_equal(Isp, 285.4, tol)
        assert_near_equal(gamma_t, 1.24, tol)
        # -------------------------------------------------------

    def test_partials(self):
        p = om.Problem(model=om.Group())
        p.model.add_subsystem(name='propulsion', subsys=PropulsionStage(g0 = 9.80665, 
                                                                        nb_e = 1, 
                                                                        eta_C_f = 1.0,
                                                                        eta_cStar = 1.0,
                                                                        Rmc = 8314.0, 
                                                                        P_a = 101325.0))
        
        external_params = p.model.add_subsystem('external_params', om.IndepVarComp())
            
        external_params.add_output('P_c', val=1000.0, units='psi')
        external_params.add_output('P_e', val=14.7, units='psi')
        external_params.add_output('o_f', val=2.24, units=None)
        external_params.add_output('thrust', val=1e6, units='N')
        
        p.model.connect('external_params.P_c', 'propulsion.P_c')
        p.model.connect('external_params.P_e', 'propulsion.P_e')
        p.model.connect('external_params.o_f', 'propulsion.o_f')
        p.model.connect('external_params.thrust', 'propulsion.thrust')
        p.setup()
        p.run_model()
        cpd = p.check_partials(compact_print=True, show_only_incorrect=False, out_stream=None)
        assert_check_partials(cpd, atol=1.0E-2, rtol=1.0E-2) # poor tolerance
        


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
    
    