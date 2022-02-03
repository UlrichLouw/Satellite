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
    SET_PARAMS.Kalman_filter_use = "EKF"
    SET_PARAMS.Mode = "EARTH_SUN"
    SET_PARAMS.SensorPredictor = "None"
    SET_PARAMS.SensorRecoveror = "None" 
    SET_PARAMS.SensorIsolator = "None"
    SET_PARAMS.number_of_faults = 2
    SET_PARAMS.Number_of_satellites = 100
    SET_PARAMS.Model_or_Measured = "ORC"
    constellation = False
    multi_class = False
    lowPredictionAccuracy = False

    GenericPath = "Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/"+ SET_PARAMS.Model_or_Measured +"/" +"General CubeSat Model/"
    
    if constellation:
        GenericPath = "Constellation/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/"+ SET_PARAMS.Model_or_Measured +"/" +"General CubeSat Model/"
    
    SET_PARAMS.path = SET_PARAMS.path + GenericPath

    SET_PARAMS.Number_of_multiple_orbits = 2
    
    for index in range(1, SET_PARAMS.Number_of_multiple_orbits):
        name = SET_PARAMS.Fault_names_values[index+1]
        print(name)
        Y, Y_buffer, X, X_buffer, Orbit, ColumnNames, ClassNames = Dataset_order(name, binary_set, buffer, categorical_num, use_previously_saved_models)
        All_orbits.append(Orbit)

        # if use_previously_saved_models == False:
        #     print(X.shape, Y.shape)
        #     cm = prediction_NN(X, Y, index, None)
        #     print(cm, str(index))      
    
    # if buffer == False:
    #     All_orbits = pd.concat(All_orbits)
    #     X = All_orbits.iloc[:,1:-1].values
    #     Y = All_orbits.iloc[:,-1].values
    # else:
    #     X = np.asarray(X_buffer)
    #     Y = np.asarray(Y_buffer).reshape(X.shape[0], Y.shape[1])

    if use_previously_saved_models == False:
        index = "all samples"
        cm = prediction_NN(X, Y, index, None)
        print(cm, index)

    else:
        cm = prediction_NN_determine_other_NN(X, Y, SET_PARAMS)
        print(cm)