import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydmd import DMD
from Simulation.Parameters import SET_PARAMS
from Fault_prediction.Fault_utils import Dataset_order
from pydmd import DMD

SET_PARAMS.load_as == ".csv"
# Firstly the data must be extracted from the csv file. 
# Afterwards the DMD operations must be executed.

buffer = False
binary_set = True
use_previously_saved_models = False
categorical_num = False

Y, Y_buffer, X, X_buffer, Orbit = Dataset_order("None", binary_set, buffer, categorical_num, use_previously_saved_models)

t = range(len(Y))

dmd = DMD(svd_rank=2)
dmd.fit(X.T)


dmd.plot_eigs(show_axes=True, show_unit_circle=True)


for dynamic in dmd.dynamics:
    plt.plot(t, dynamic.real)
    plt.title('Dynamics')
plt.show()