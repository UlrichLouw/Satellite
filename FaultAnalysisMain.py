import numpy as np
from Simulation.Parameters import SET_PARAMS

from Fault_prediction.Feature_extraction import DMD
from Fault_prediction.Supervised_Learning import DecisionForests
import sys
np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':
    SET_PARAMS.Visualize = True
    SET_PARAMS.sensor_number = 0
    SET_PARAMS.Kalman_filter_use = "None"
    SET_PARAMS.Mode = "EARTH_SUN"
    SET_PARAMS.path = SET_PARAMS.path + "KalmanFilter-"+SET_PARAMS.Kalman_filter_use+"/"+SET_PARAMS.Mode+"/"
    # Compute the A and B matrix to estimate X
    for i in range(SET_PARAMS.numberOfSensors):
        DMD.MatrixAB(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod')
        SET_PARAMS.sensor_number += 1
    SET_PARAMS.sensor_number = "ALL"
    DMD.MatrixAB(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod')
    #DecisionForests.DecisionTreeAllAnomalies(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod', depth = 10)