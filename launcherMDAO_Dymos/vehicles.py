# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 21:52:16 2020

@author: jorge
"""

class TSTO_stage():
    def __init__(self, nb_e, mass_aux, areaFactor):
        self.nb_e = nb_e                                                                # ()     number of engines
        self.mass_aux = mass_aux                                                        # ()     auxiliary mass 
        self.areaFactor = areaFactor                                                    # ()     ratio between maximum total exit area of the engines and stage diameter
        
    def write_stage_opt(self, T, Isp, Ae_t, mp, ms, g0):
        self.ms = ms                                                                    # (kg)   stage structural mass
        self.Isp = Isp                                                                  # (s)    stage Specific impulse
        self.T = T                                                                      # (N)    stage thrust at vacuum
        self.Ae_t = Ae_t                                                                # (m^2)  stage Engine nozzle exit area
        self.mp = mp                                                                    # (kg)   stage Initial mass of propellants
        self.mfrMax = T / (g0 * Isp)                                                    # (kg/s) stage maximum mass flow rate
        self.burnTime = mp / self.mfrMax                                                # (s)    stage burning time for propellants at constant mfr (100%)
        
class TSTO():
    def __init__(self, md, mplf, mass_aux_1, nb_e_first_stage, nb_e_second_stage, areaFactor_1, areaFactor_2, centralBody,uncertainties):
        "Initialize Two Stage to orbit vehicle"
        self.md = md                                                                    # (kg)    payload mass
        self.g0 = centralBody.g0                                                        # (m/s^2) gravity at r0
        self.mplf = mplf                                                                # (kg)    payload fairing mass
        self.stage_1 = TSTO_stage(nb_e_first_stage, mass_aux_1, areaFactor_1)
        self.stage_2 = TSTO_stage(nb_e_second_stage, 0, areaFactor_2)
        
        self.u_Isp1 = uncertainties[0]                                                  # (s)    uncertainty about first stage Isp (additive)
        self.u_Isp2 = uncertainties[1]                                                  # (s)    uncertainty about second stage Isp (additive)
        self.u_m1 = uncertainties[2]                                                    # (kg)   uncertainty about first stage mass (additive)
        self.u_m2 = uncertainties[3]                                                    # (kg)   uncertainty about second stage mass (additive)
        self.u_mfr1 = uncertainties[4]                                                  # (kg/s) uncertainty about first stage mfr (additive)
        self.u_mfr2 = uncertainties[5]                                                  # (kg/s) uncertainty about second stage mfr (additive)
        self.u_cd = uncertainties[6]                                                    # (NA)   uncertainty about draf coefficient (multiplicative)
        
        
        
    def write_stage_1_opt(self, T, Isp, Ae_t, mp, ms):
        "Update vehicle information after optimization is terminated"
        self.stage_1.write_stage_opt(T, Isp, Ae_t, mp, ms, self.g0)
        
    def write_stage_2_opt(self, T, Isp, Ae_t, mp, ms):
        "Update vehicle information after optimization is terminated"
        self.stage_2.write_stage_opt(T, Isp, Ae_t, mp, ms, self.g0)
        
    def calculateMassRatios(self,printRatios):
        "calculate useful ratios after optimization is terminated"
        ms_1 = self.stage_1.ms
        ms_2 = self.stage_2.ms
        mp_1 = self.stage_1.mp
        mp_2 = self.stage_2.mp
        md   = self.md
        mplf = self.mplf
        
        self.mf_a          = ms_1 + ms_2 + mplf + md + mp_1 + mp_2                      # (kg)   full mass for 1st stage flight with fairing
        self.mf_b          = 0.0  + ms_2 + mplf + md + 0.0  + mp_2                      # (kg)   full mass for 2nd stage flight with fairing
        
        self.me_a          = ms_1 + ms_2 + mplf + md + 0.0  + mp_2                      # (kg)   empty mass for 1st stage flight with fairing
        self.me_c          = 0.0  + ms_2 + 0.0  + md + 0.0  + 0.0                       # (kg)   empty mass for 2nd stage flight without fairing
        
        self.TwRatio_a = self.stage_1.T / (self.mf_a * self.g0)                         # ()     thrust to weight ratio at vacuum for 1st stage flight
        self.TwRatio_b = self.stage_2.T / (self.mf_b * self.g0)                         # ()     thrust to weight ratio at vacuum for 2nd stage flight
        
        self.structCoef_1 = ms_1 / (mp_1 + ms_1)                                        # ()     1st stage structural coefficient 
        self.structCoef_2 = ms_2 / (mp_2 + ms_2)                                        # ()     structural coefficient for 2nd stage fligh with fairing
        
        
        if printRatios == True :
            space = 40
            # print('============= Input data =====================')
            
            # print('==============================================')
            print('=========== TSTO Mass ratios =================')
            print('1st stage struct coef : '.ljust(space) + str(round(self.structCoef_1,3)))
            print('2nd stage struct coef : '.ljust(space) + str(round(self.structCoef_2,3)))
            
            print('1st stage flight Thrust/ weight : '.ljust(space) + str(round(self.TmRatio_a,2)))
            print('2nd stage flight Thrust/ weight : '.ljust(space) + str(round(self.TmRatio_b,2)))
            
            print('1st stage flight full mass: '.ljust(space) + str(round(self.mf_a/1e3,1)) + ' (ton)')
            print('2nd stage flight full mass: '.ljust(space) + str(round(self.mf_b/1e3,1)) + ' (ton)')
            
            print('==============================================')