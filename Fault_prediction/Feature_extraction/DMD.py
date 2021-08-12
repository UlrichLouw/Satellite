from Fault_prediction.Fault_utils import Dataset_order
import numpy as np
from Simulation.Parameters import SET_PARAMS
from pathlib import Path

def MatrixAB(path, fromTimeStep = 0, ToTimeStep = 1000):
    SET_PARAMS.load_as == ".csv"
    # Firstly the data must be extracted from the csv file. 
    # Afterwards the DMD operations must be executed.
    Y, _, X, _, _ = Dataset_order("None", binary_set = True, buffer = False, categorical_num = False)

    sensor_number = 4

    t = len(Y) - ToTimeStep - fromTimeStep
    # Select the data for the sensor of interest
    x1 = X[fromTimeStep:ToTimeStep-1,3*sensor_number:3*sensor_number + 3].T

    # Select the data for the sensors that will be used to predict the next step of sensor x
    x2 = X[fromTimeStep + 1:ToTimeStep,3*sensor_number:3*sensor_number + 3].T

    # Select y, which impacts the vector x (Control DMD)
    y1 = np.roll(X, 3*sensor_number)[fromTimeStep:ToTimeStep-1,3:].T

    #! Without control matrix B
    #! A = x2 @ np.linalg.pinv(x1)
    #! x2_approximate = A @ x1

    G = x2 @ np.linalg.pinv(np.concatenate((x1, y1)))

    A = G[:,:3]
    B = G[:,3:]

    path_to_folder = Path(path)
    path_to_folder.mkdir(exist_ok=True)

    np.save(SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod/A_matrixs.npy', A)
    np.save(SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod/B_matrix.npy', B)