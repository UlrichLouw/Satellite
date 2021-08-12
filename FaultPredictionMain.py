import pandas as pd
from Simulation.Parameters import SET_PARAMS
import numpy as np
from Fault_prediction.Fault_utils import Dataset_order
from Fault_prediction.Supervised_Learning.Fault_prediction import prediction_NN, prediction_NN_determine_other_NN

class Fault_Detection:
    def __init__(self):
        pass

    def Predict(self):
        pass


if __name__ == "__main__":
    confusion_matrices = []
    All_orbits = []
    X_buffer = []
    Y_buffer = []
    buffer = False
    binary_set = True
    use_previously_saved_models = False
    categorical_num = True
    
    for index in range(SET_PARAMS.Number_of_multiple_orbits - 1):
        name = SET_PARAMS.Fault_names_values[index+1]
        Y, Y_buffer, X, X_buffer, Orbit = Dataset_order(name, binary_set, buffer, categorical_num, use_previously_saved_models)
        All_orbits.append(Orbit)

        if use_previously_saved_models == False:
            cm = prediction_NN(X, Y, index, None)
            print(cm, str(index))      
    
    if buffer == False:
        All_orbits = pd.concat(All_orbits)
        X = All_orbits.iloc[:,1:-1].values
        Y = All_orbits.iloc[:,-1].values
    else:
        X = np.asarray(X_buffer)
        Y = np.asarray(Y_buffer).reshape(X.shape[0], Y.shape[1])

    if use_previously_saved_models == False:
        index = "all samples"
        cm = prediction_NN(X, Y, index, None)
        print(cm, index)

    else:
        cm = prediction_NN_determine_other_NN(X, Y, SET_PARAMS)
        print(cm)