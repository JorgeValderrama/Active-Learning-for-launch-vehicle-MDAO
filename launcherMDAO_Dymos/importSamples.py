# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 15:18:05 2021

@author: jorge
"""
import json
# import codecs
import openturns as ot

# from Y import calcul_comportement
# from launcherMDAO_Dymos.main_opt_traj import opt_traj
# from launcherMDAO_Dymos.interpolate_results_normalized import load_result_and_interpolate
import numpy as np

# from sample_generator import saveOutputSampleSet, saveSampleSet

class ALValidationSamples():
    """
    Validation samples for active learning
    
    container for input and output samples and mesh.
        
        Types
        ----------
        input : open turns sample
            input sample
        ouput : open turns processSample
            output sample
        mesh  : open turns mesh
            mesh for output sample
    """
    def __init__(self, pathInput, pathOutput, sampleSize):
        """
        

        Parameters
        ----------
        pathInput : string
            path to input sample
        pathOutput : string
            path to output sample
        sampleSize : integer
            number of samples to be used from the data sets

        Returns
        -------
        None.

        """
        self.input  = self.importInputSample(pathInput, sampleSize)
        self.mesh   = self.importMesh(pathOutput)
        self.output = self.importOuputSample(pathOutput, sampleSize, self.mesh)
        
    def importMesh(self, path):
        """
        import mesh for output sample

        Parameters
        ----------
        path : string
            path to sample file

        Returns
        -------
        open turns mesh
            mesh of output sample

        """
        file = open(path, 'r', encoding='utf-8')
        obj_text = file.read()
        sim = json.loads(obj_text)['mesh']['simplices']
        ver = json.loads(obj_text)['mesh']['vertices']
        file.close()
        return ot.Mesh(ver,sim)
        
    def importInputSample(self, path,sampleSize):
        """
        

        Parameters
        ----------
        path : string
            path to sample file
        sampleSize : integer
            number of samples to be used from the data sets

        Returns
        -------
        sample : open turns sample
            input sample

        """
        file = open(path, 'r', encoding='utf-8')
        obj_text = file.read()
        sample = ot.Sample(json.loads(obj_text)['resultsDict'])
        file.close()
        sample.split(sampleSize)
        
        return sample
    
    def importOuputSample(self, path, sampleSize, mesh):
        """
        

        Parameters
        ----------
        path : string
            path to sample file
        sampleSize : integer
            number of samples to be used from the data sets
        mesh : open turns mesh
            mesh for output sample

        Returns
        -------
        sample : open turns processSample
            output sample

        """
        file = open(path, 'r', encoding='utf-8')
        obj_text = file.read()
        sample_ = json.loads(obj_text)['resultsDict']
        file.close()
        # create empty process sample and loop through sample while filling it
        sample = ot.ProcessSample(mesh,0,1)
        for traj in sample_[0:sampleSize]:
            sample.add(ot.Sample(traj))

        return sample
    
    def centerOutput(self, mean, scale):
        for i in range(self.output.getSize() ):
            self.output[i] = (self.output[i] - mean ) * scale
            
    def normalizeInput(self, mean, std):
        for i in range(self.input.getSize() ):
            self.input[i] = ot.Point( np.array((self.input[i] - mean)) / np.array(std) )


