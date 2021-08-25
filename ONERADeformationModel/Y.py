import numpy as np


def p_pos(x):
        
    return 1/2* (x +np.abs(x))
    
def endo(params,eps, ymax): #### 
    
    y_thermo = 1/2* params[0]* p_pos(eps) **2
                    
    y_maxk = np.maximum(np.maximum.accumulate(y_thermo), ymax)
                
    return params[3] * (1- np.exp(-p_pos((np.sqrt(y_maxk) - np.sqrt(params[1]))/np.sqrt(params[2]))**1.2)),y_maxk,y_thermo


def calc_déf_res(eps,K_eff,Xi, data_endo):

   y_thermo =  data_endo[2]
   
   endo_eff = data_endo[0]
    
   y_maxk = data_endo[1]
            
   déf_rés = np.zeros(len(eps))            
        
   D_endo = np.diff(endo_eff)
    
   for k in range(1,len(eps)):
       
       if not D_endo[k-1]:
           
           D_déf_rés = 0
            
       else:
            
            if y_thermo[k-1] < y_maxk[k-1] and y_thermo[k] > y_maxk[k-1]:
                
                alpha_y = np.sqrt((y_maxk[k-1]- y_thermo[k-1])/(y_thermo[k]- y_thermo[k-1]))
                
                def_a =  eps[k-1] + alpha_y*(eps[k] - eps[k-1])          
                
            else:
                
                def_a = eps[k-1]
            
            if y_thermo[k-1] > y_maxk[k-1] and y_thermo[k] < y_maxk[k-1]:
                
                alpha_y = np.sqrt((y_maxk[k-1]- y_thermo[k-1])/(y_thermo[k-1]- y_thermo[k]))
                
                def_b =  eps[k] + alpha_y*(eps[k-1] - eps[k])    
                
            else:
                
                def_b = eps[k]                         
                
            F_a = Xi * K_eff[k-1] * def_a                
            
            F_b = Xi * K_eff[k] * def_b
            
            D_déf_rés = 1/2 * D_endo[k-1] * (F_a + F_b)
                                       
       déf_rés[k] = D_déf_rés + déf_rés[k-1]
                
   return déf_rés


def calcul_comportement(eps,params, inc_= 0, ymax = 0):
    
    eta = 1

    
    endo_d = endo(params,eps, ymax)
    
    Rig_eff =  np.divide(params[0],np.ones(len(endo_d[0]))+ eta*endo_d[0])        
    
    endo_d[1][0] =  ymax
    
    if not inc_:           
                    
        return Rig_eff * eps
    
    else :            
    
         
        K_eff =  eta * (Rig_eff/params[0]) ** 2
        
        déf_rés = calc_déf_res(eps,K_eff,params[5],endo_d)
                                
        return Rig_eff * eps - params[0] * déf_rés
    


def gg(x,c):

   return np.divide(x,c) 
