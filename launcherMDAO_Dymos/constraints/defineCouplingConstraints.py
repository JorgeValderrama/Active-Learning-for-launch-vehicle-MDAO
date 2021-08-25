# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 00:05:49 2021

@author: jorge
"""

from .constraintsJettison import ConstraintsJettison
from .constraintsPropellants import ConstraintsPropellants
from .constraintsLoadFactor import ConstraintsLoadFactor
from .constraintsDynamicPressure import ConstraintsDynamicPressure
from .constraintsExitArea import ConstraintsExitArea

from .massJetisson import MassJettison


def defineCouplingConstraints(p, vehicle):
    
    # add components related to constraint couplings
    # ==========================================================================================================
    
    # component to calculate the mass being jettisoned at first jettison and fairing jettison
    p.model.add_subsystem('massJettison', subsys = MassJettison(mplf = vehicle.mplf, md = vehicle.md))
    
    # component to enforce that the mass being jettisoned during fairing jettison corresponds to the mass of the fairing
    constraintsJettison = p.model.add_subsystem('constraintsJettison', 
                                                ConstraintsJettison(mplf = vehicle.mplf, md = vehicle.md))
    
    # component to enforce that the total area of the fairings stays below the cross-sectional area of the stage
    constraintsExitArea = p.model.add_subsystem('constraintsExitArea', 
                                                ConstraintsExitArea(areaFactor_1 = 0.64, areaFactor_2 = 0.64))
    
    # component to enforce the coupling for the maximum dynamic pressure of the trajectory and the massSizing module
    constraintsDynamicPressure = p.model.add_subsystem('constraintsDynamicPressure', ConstraintsDynamicPressure())
    
    # component to enforce the coupling for the mass of propellants of both stages between trajectory and the massSizing module
    constraintsPropellants = p.model.add_subsystem('constraintsPropellants', ConstraintsPropellants(mplf = vehicle.mplf))
    
    # component to enforce the coupling between the load factor at the end of first stage flight (max load factor)
    # from the trajectory and the massSizing module. Be careful as the max load factor can happen at a point
    # different from the end of first stage flight.
    constraintsLoadFactor = p.model.add_subsystem('constraintsLoadFactor', ConstraintsLoadFactor())
    
    # define constraints
    # ===========================================================================================================
    
    # contraints Jettison
    constraintsJettison.add_constraint('residual_ms_1', lower = 0.0, ref = 1.1)
    constraintsJettison.add_constraint('residual_mplf', lower = 0.0, ref = 1.1)
    constraintsJettison.add_constraint('residual_m_final', lower = 0.0, ref = 1.1)
    
    # # constraints exit area
    # constraintsExitArea.add_constraint('residual_area_1', lower = 0.0, ref = 1.1)
    # constraintsExitArea.add_constraint('residual_area_2', lower = 0.0, ref = 1.1)
    
    # constraint dynamic pressure
    #constraintsDynamicPressure.add_constraint('residual_max_q_dyn', lower = 0.0, ref = 1.1)
    
    # constraints propellants
    constraintsPropellants.add_constraint('residual_mp_1', lower = 0.0, ref = 1.1)
    constraintsPropellants.add_constraint('residual_mp_2', lower = 0.0, ref = 1.1)
    
    # constraint max load factor
    #constraintsLoadFactor.add_constraint('residual_max_n_f_1', lower = 0.0, ref = 1.1)
    
    # create dictionary with constraint components to facilitate the reading of constraint values
    # ===========================================================================================================
    constraintComponents = {'constraintsJettison' : constraintsJettison,
                            'constraintsPropellants' : constraintsPropellants}
                            # 'constraintsExitArea' : constraintsExitArea,
                            # 'constraintsDynamicPressure' : constraintsDynamicPressure,

                            # 'constraintsLoadFactor' : constraintsLoadFactor}
    
    return p, constraintComponents