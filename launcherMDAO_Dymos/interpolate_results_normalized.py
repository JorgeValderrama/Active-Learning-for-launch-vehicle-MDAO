# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:30:07 2021

@author: jorge
"""
import json
import codecs
import numpy as np
from scipy import interpolate
import sys


# read json files with results
# ===========================================================================================
# read results from Dymos

# dfine function to convert elemtns of list into lists
def extractElements(lst):
    return [[el] for el in lst]

def interpolateResultsToGrid(tnew, results_dict, kind):
    # This function interpoaltes the timeseries results contained in "results_dict"
    # using the new time grid "tnew"

        # calculate acceleration using finite differences. ignore t_0 = 0
        acc = np.gradient(results_dict['v'][1:], results_dict['time'][1:])
        
        # add value of t_0 = 0 and append to dictionary
        results_dict['acceleration'] = np.append(acc[0],acc)
        
        # create emty dictionray to store results of interpolation
        results_intepolation = {}
        # save the standard time
        results_intepolation['time_original'] = results_dict['time']
        # extract time and normalize it
        tf = results_dict['time'][-1]
        time = [t / tf for t in results_dict.pop('time') ]
        
        # normalize tnew
        # save interpolated time and then normalize time
        results_intepolation['time'] = tnew.tolist()
        tnew = tnew/ tnew[-1]
            
        # loop through dictionary of results
        for key, value in results_dict.items():
            
            # extract value of time series
            y = np.array(value)
    
            # intepolate
            f = interpolate.interp1d( np.array(time) , y , kind = kind )

            # evaluate the interpolator on the new grid
            results_intepolation[key] = extractElements( f(tnew).tolist() )
        # normalize time
        # results_intepolation['time_norm'] = [ [ t ]for t in tnew] 
        results_intepolation['time_norm'] = tnew.tolist()
        return results_intepolation
    
def cleanDymosResults(results_dict):
    # the results corresponding to the nodes of the pseudospectral transcription
    # with LGL 3 with uncompressed transcription have repeated results
    # [a,b,c,c,d,e,e,f,g]
    for key, value in results_dict.items():
        results_dict[key] = [item for index, item in enumerate(value) if (index + 1) % 3 != 0]
        results_dict[key].append( value[-1] )
    return results_dict
    
    
def load_result_and_interpolate(file_name):
    # this function returns dictionaries for the interpolated results from
    # Dymos and from the simulate method. it also extracts the opt_info dictionary

    # read resutls from .txt file
    obj_text = codecs.open(file_name , 'r', encoding='utf-8').read()
    results_dict    = json.loads(obj_text)['resultsDict']
    results_dict    = cleanDymosResults(results_dict)
    resultsSim_dict = json.loads(obj_text)['results_simDict']
    opt_info        = json.loads(obj_text)['opt_info']
    
    if opt_info['status'] !=0:
        print('Attention!! Optimization results did not converged')
        return {}
    else:
        
        tnew    = np.linspace(0, results_dict['time'][-1], 201)
        # interpolate using cubic spline to smooth out pseudospectral deffects
        interpolated_results = interpolateResultsToGrid(tnew, results_dict, kind = 'quadratic')
        
        # tnewSim = np.linspace(0, resultsSim_dict['time'][-1], 201)
        tnewSim = np.arange(0, 430, 2)
        interpolated_resultsSim = interpolateResultsToGrid(tnewSim, resultsSim_dict, kind = 'linear')
        
        results_dict = {'interpolated_results'   :interpolated_results,
                        'interpolated_resultsSim':interpolated_resultsSim,
                        'opt_info'               :opt_info}
    
        return results_dict
    
if __name__=='__main__':
    # iteratively generate dictionaries with interpolated results
    number_of_realizations = 1200
    exDict = {}

    list_of_dictionaries = []
    for Id in range (number_of_realizations):
        # group the 3 output dictionaries
        # exDict[str(Id) + '_interp_results'] =  load_result_and_interpolate(str(Id) + '_trajectory_state_history')
        list_of_dictionaries.append(load_result_and_interpolate('results/' + str(Id) + '_trajectory_state_history.txt'))
    exDict['interp_results']  = list_of_dictionaries
    
    # save to text files
    with open ('results/intepolated_results/intepolated_results.txt', 'w') as file:
        file.write(json.dumps(exDict))
        
    