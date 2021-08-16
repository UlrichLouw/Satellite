import numpy as np
from Simulation.Parameters import SET_PARAMS

from Fault_prediction.Feature_extraction import DMD
from Fault_prediction.Supervised_Learning import DecisionForests
import sys
np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':
    SET_PARAMS.Visualize = True
    SET_PARAMS.sensor_number = 1
    # Compute the A and B matrix to estimate X
    DMD.MatrixAB(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod')
    DecisionForests.DecisionTreeAllAnomalies(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod', depth = 10)