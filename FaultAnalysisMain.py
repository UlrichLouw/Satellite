import numpy as np
from Simulation.Parameters import SET_PARAMS

from Fault_prediction.Feature_extraction import DMD
from Fault_prediction.Supervised_Learning import DecisionForests
from Fault_prediction.Supervised_Learning import Random_Forest
from Fault_prediction.Unsupervised_Learning import Isolation_Forest
from Fault_prediction.Unsupervised_Learning import Extended_Isolation_Forest
import sys
import pickle
import matplotlib.pyplot as plt

from sklearn import tree

np.set_printoptions(threshold=500)

if __name__ == '__main__':
    SET_PARAMS.Visualize = True
    SET_PARAMS.sensor_number = 0
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
    MovingAverage = True
    includeAngularMomentumSensors = True
    includeModelled = True

    treeDepth = [5, 10, 20, 100]

    GenericPath = "Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/"+ SET_PARAMS.Model_or_Measured +"/" +"General CubeSat Model/"
    
    if constellation:
        GenericPath = "Constellation/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/"+ SET_PARAMS.Model_or_Measured +"/" +"General CubeSat Model/"
    
    SET_PARAMS.path = SET_PARAMS.path + GenericPath
    # SET_PARAMS.numberOfSensors = 3
    # # Compute the A and B matrix to estimate X
    # for i in range(SET_PARAMS.numberOfSensors):
    #     DMD.MatrixAB(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod', includeModelled = includeModelled)
    #     SET_PARAMS.sensor_number += 1
    # SET_PARAMS.sensor_number = "ALL"
    # DMD.MatrixAB(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod', includeModelled = includeModelled)
    # DecisionTree training
    
    # Isolation_Forest.IsoForest(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod', depth = 10, constellation = constellation, multi_class = multi_class, lowPredictionAccuracy = lowPredictionAccuracy, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors)
    Extended_Isolation_Forest.IsoForest(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod', depth = 10, constellation = constellation, multi_class = multi_class, lowPredictionAccuracy = lowPredictionAccuracy, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors)
    # for depth in treeDepth:
    #     SET_PARAMS.treeDepth = depth
    #     # DecisionForests.DecisionTreeAllAnomalies(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod', depth = depth, constellation = constellation, multi_class = multi_class, lowPredictionAccuracy = lowPredictionAccuracy, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled)
    #     Random_Forest.Random_Forest(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod', depth = depth, constellation = constellation, multi_class = False, lowPredictionAccuracy = lowPredictionAccuracy, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled)