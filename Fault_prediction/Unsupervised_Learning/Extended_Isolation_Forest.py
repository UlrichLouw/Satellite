import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator
import numpy as np
# import eif as iso
import pandas as pd
from Simulation.Parameters import SET_PARAMS
from Fault_prediction.Fault_utils import Dataset_order

h2o.init()

def IsoForest(path, depth, multi_class = False, constellation = False, lowPredictionAccuracy = False, MovingAverage = True, includeAngularMomemntumSensors = False):
    X_list = []
    Y_list = []

    pathFiles = SET_PARAMS.path

    buffer = True
    SET_PARAMS.buffer_size = 2

    if constellation:
        for satNum in range(SET_PARAMS.Number_of_satellites):
            print(satNum)
            SET_PARAMS.path = pathFiles + str(satNum) + "/"
            for index in range(SET_PARAMS.number_of_faults):
                name = SET_PARAMS.Fault_names_values[index+1]
                if multi_class:
                    Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = False, categorical_num = True, buffer = buffer, constellation = constellation, multi_class = True, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors)
                else:
                    Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = True, buffer = buffer, categorical_num = False, constellation = constellation, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors)
                X_list.append(X)    
                Y_list.append(Y)

    else:
        for index in range(SET_PARAMS.number_of_faults):
            name = SET_PARAMS.Fault_names_values[index+1]
            if multi_class:
                Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = False, categorical_num = True, buffer = buffer, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors)
            else:
                Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = True, buffer = buffer, categorical_num = False, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors)
            X_list.append(X)    
            Y_list.append(Y)

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)

    h2o_df = pd.DataFrame(X)
    # Set the predictors
    predictors = ["Failure", "No Failure"]

    # Define an Extended Isolation forest model
    eif = H2OExtendedIsolationForestEstimator(model_id = "eif.hex",
                                            ntrees = 100,
                                            sample_size = 256,
                                            extension_level = len(predictors) - 1)

    # Train Extended Isolation Forest
    eif.train(x = predictors,
            training_frame = h2o_df)

    # Calculate score
    eif_result = eif.predict(h2o_df)

    # Number in [0, 1] explicitly defined in Equation (1) from Extended Isolation Forest paper
    # or in paragraph '2 Isolation and Isolation Trees' of Isolation Forest paper
    anomaly_score = eif_result["anomaly_score"]
    print(anomaly_score)

    # Average path length  of the point in Isolation Trees from root to the leaf
    mean_length = eif_result["mean_length"]
    print(mean_length)