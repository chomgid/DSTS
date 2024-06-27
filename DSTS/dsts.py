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
        
        self.databool = self.test(data)
        if not self.databool: 
            raise ValueError("Invalid data provided for DSTS model")

    def test(self, data):

        return True

    def generate(self, iter=3, tot_iter=1, aug=5, n_comp=2) -> np.ndarray:
        """
        Synthesizes a new time series using DS2 algorithms.

        Parameters:
        data (np.ndarray): Input data array of shape (size, length).
        n_comp (int): The number of mixture components in GMM. Default is 2.
        aug (int): How many times the size of the synthesized data should be relative to the original data. Default is 5.

        Returns:
        np.ndarray: The synthesized data array of shape (size * aug, length).

        """
        if not self.databool:
            raise ValueError("Data is invalid")

        size = self.data.shape[0]
        length = self.data.shape[1]
        y1 = draw_y1(self.data[:,:1], n_comp, aug)
        rs = make_rs_matrix(self.data, aug)
        synth = np.ones((size*aug,length))
        synth[:,0] = y1
        synth[:,1:] = (y1*rs.T).T

        calib_data = calibration(self.data, synth, iter, tot_iter)

        return synth, calib_data
