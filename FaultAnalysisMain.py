import numpy as np
from Simulation.Parameters import SET_PARAMS

from Fault_prediction.Feature_extraction import DMD
from Fault_prediction.Supervised_Learning import DecisionForests
from Fault_prediction.Supervised_Learning import Random_Forest
import sys

if __name__ == '__main__':
    SET_PARAMS.Visualize = True
    SET_PARAMS.sensor_number = 0
    SET_PARAMS.Kalman_filter_use = "EKF"
    SET_PARAMS.Mode = "EARTH_SUN"
    SET_PARAMS.SensorPredictor = "None"
    SET_PARAMS.SensorRecoveror = "None" 
    SET_PARAMS.SensorIsolator = "None"
    GenericPath = "Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/"+ "General CubeSat Model/"
    SET_PARAMS.path = SET_PARAMS.path + GenericPath
    SET_PARAMS.numberOfSensors = 4
    # Compute the A and B matrix to estimate X
    for i in range(SET_PARAMS.numberOfSensors):
        DMD.MatrixAB(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod')
        SET_PARAMS.sensor_number += 1
    SET_PARAMS.sensor_number = "ALL"
    DMD.MatrixAB(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod')
    # Binary training
    # DecisionForests.DecisionTreeAllAnomalies(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod', depth = 10)
    # Multi class prediction
    # DecisionForests.DecisionTreeAllAnomalies(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod', depth = 10, multi_class = True)
    # Random_Forest.Random_Forest(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod', depth = 10, multi_class = False)