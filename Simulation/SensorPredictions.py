import collections
from Simulation.Parameters import SET_PARAMS
import numpy as np

class SensorPredictionsDMD:
    def __init__(self, sensors_X):
        self.Buffer_est = collections.deque(maxlen = SET_PARAMS.MovingAverageSizeOfBuffer)
        self.Buffer_act = collections.deque(maxlen = SET_PARAMS.MovingAverageSizeOfBuffer)
        self.DMD_Prediction_A = np.load(SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod/A_matrixs.npy')
        self.DMD_Prediction_B = np.load(SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod/B_matrixs.npy')
        self.DMD_Prediction_k = 0.001
        self.x_est = sensors_X
        self.x = sensors_X

    def MovingAverage(self, sensors_X, sensors_Y):
        if SET_PARAMS.sensor_number == "ALL":
            x_est = self.DMD_Prediction_A @ self.x_est + self.DMD_Prediction_k*(self.x - self.x_est)
        else:
            x_est = self.DMD_Prediction_A @ self.x_est + self.DMD_Prediction_B @ sensors_Y + self.DMD_Prediction_k*(self.x - self.x_est)

        print(x_est - sensors_X)

        self.x = sensors_X
        self.Buffer_est.append(self.x_est)
        self.Buffer_act.append(self.x)

        self.x_est = x_est

        Actual_Sensor = np.array(self.Buffer_act)

        Estimated_Sensor = np.array(self.Buffer_est)

        V = 1/SET_PARAMS.MovingAverageSizeOfBuffer * np.sum((Actual_Sensor - Estimated_Sensor) @ (Actual_Sensor - Estimated_Sensor).T)

        return V