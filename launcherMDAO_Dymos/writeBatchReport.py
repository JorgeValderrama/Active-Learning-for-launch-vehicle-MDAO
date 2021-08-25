# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 19:35:17 2020

@author: jorge
"""

import time

def writeBatchReport(Id, p,  start_time):
    
    if Id ==0:
        file1 = open("results/BatchOptReport.txt","w") 
        file1.write('Id, cost, opt_time, njev, nfev, status \n')
    
    file1 = open("results/BatchOptReport.txt","a") 
    
    cost     = round( p.get_val('traj.lift_off.timeseries.states:m')[0][0] / 1e3 , 2)
    opt_time = round(time.time() - start_time,2)
    njev     = p.driver.result.njev
    nfev     = p.driver.result.nfev
        
    my_str = str ([Id, cost, opt_time, njev, nfev ]) +  ',' + str(p.driver.result.status)
    
    my_str = my_str.replace('[','')
    my_str = my_str.replace(']','')
    my_str = my_str.replace(')','')
    my_str = my_str.replace('(','')
    
    file1.write( my_str + '\n')
    
    file1.close()
    