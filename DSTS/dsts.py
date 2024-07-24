import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from DSTS.calibration import *
from DSTS.synthesize import *

class dsts:
    def __init__(self, data):
        try:
            self.data = np.array(data)
        except:
            raise ValueError("Data cannot be converted to numpy ndarray")
        
        self.test(data)


    def test(self, data):
        # Check if data contains any negative or zero values
        if np.any(data <= 0):
            raise ValueError("Your data must not contain any negative or zero values.")
        
        # Check if data contains any NaN values
        if np.isnan(data).any():
            raise ValueError("Your data must not contain any NaN values.")
        
        # If no issues, pass the check
        pass         


    def generate(self, iter=3, tot_iter=4, aug=5, n_comp=2, sort = True, condGMM = False, LR = False) -> np.ndarray:
        """
        Synthesizes a new time series using DS2 algorithms.

        Parameters:
        data (np.ndarray): Input data array of shape (size, length).
        iter (int, optional): The number of calibration iterations for each timestamp. Defaults to 3. 
        tot_iter (int, optional): The number of calibration loops for whole time series. Default to 4.
        n_comp (int, optional): The number of mixture components in GMM. Default is 2.
        aug (int): The multiplier for the size of the synthesized data relative to the original data. Defaults to 5.
        sort (bool, optional): Set to True to use the sorting method, and False to bypass it. Defaults to True.
        condGMM (bool, optional): Use conditional Gaussian mixture model to generate y1. Note: LR and condGMM cannot both be True simultaneously. Defaults to False.
        LR (bool, optional): Use linear regression to generate y1. Note: LR and condGMM cannot both be True simultaneously. Defaults to False.
        
        Returns:
        np.ndarray: The synthesized data array of shape (size * aug, length).

        """
        if not self.databool:
            raise ValueError("Data is invalid")

        size = self.data.shape[0]
        length = self.data.shape[1]
        rstar = make_rstar(self.data, aug, sort)    

        assert not (condGMM and LR), "Both condGMM and LR cannot be True at the same time."

        if condGMM:
            y1 = draw_y1_cond(self.data, rstar, n_comp, sort)
        elif LR:
            y1 = lr_draw_y1(self.data, rstar, sort)
        else:
            y1 = draw_y1(self.data, n_comp, aug, sort)

        synth = np.ones((size*aug,length))
        synth[:,0] = y1
        synth[:,1:] = (y1*rstar.T).T

        calib_data = calibration(self.data, synth, iter, tot_iter)

        return calib_data
