import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import random as rn
import eif as iso
import seaborn as sb
sb.set_style(style="whitegrid")
sb.set_color_codes()
import scipy.ndimage
from scipy.interpolate import griddata
import numpy.ma as ma
from numpy.random import uniform, seed

import sys

sys.path.insert(0, './Simulation')
sys.path.insert(0, './Fault_prediction')

import pandas as pd
from Parameters import SET_PARAMS
from Fault_utils import Dataset_order
import os

def getDepth(x, root, d):
    n = root.n
    p = root.p
    if root.ntype == 'exNode':
        return d
    else:
        if (x-p).dot(n) < 0:
            return getDepth(x,root.left,d+1)
        else:
            return getDepth(x,root.right,d+1)
        
def getVals(forest,x,sorted=True):
    theta = np.linspace(0,2*np.pi, forest.ntrees)
    r = []
    for i in range(forest.ntrees):
        temp = forest.compute_paths_single_tree(np.array([x]),i)
        r.append(temp[0])
    if sorted:
        r = np.sort(np.array(r))
    return r, theta

if __name__ == "__main__":
    confusion_matrices = []
    All_orbits = []
    X_buffer = []
    Y_buffer = []
    buffer = False
    binary_set = True
    use_previously_saved_models = False
    categorical_num = True
    
    for index in range(SET_PARAMS.Number_of_multiple_orbits):
        Y, Y_buffer, X, X_buffer, Orbit = Dataset_order(index, binary_set, buffer, categorical_num, use_previously_saved_models)
        All_orbits.append(Orbit)

        F1 = iso.iForest(X, ntrees = 500, sample_size = 1000, ExtensionLevel=1)

        xxx = np.array([[0,0.]])
        SL0 = F1.compute_paths_single_tree(xxx, 0)

        S1 = F1.compute_paths(X_in=X)

        ss1=np.argsort(S1)

        number_of_errors = np.sum(Y % 2 == 1)
        print(np.sum(Y[ss1[:number_of_errors]])/number_of_errors, index)
