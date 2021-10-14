import collections
from Simulation.Parameters import SET_PARAMS
import numpy as np

class SensorPredictionsDMD:
    def __init__(self, sensors_X, sensor_number):
        self.Buffer_est = collections.deque(maxlen = SET_PARAMS.MovingAverageSizeOfBuffer)
        self.Buffer_act = collections.deque(maxlen = SET_PARAMS.MovingAverageSizeOfBuffer)
        self.DMD_Prediction_A = np.load(SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod/' + str(sensor_number) + 'A_matrixs.npy')
        self.DMD_Prediction_B = np.load(SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod/' + str(sensor_number) + 'B_matrixs.npy')
        self.DMD_Prediction_k = 0.001
        self.x_est = sensors_X 
        self.x = self.x_est
        self.sensor_number = sensor_number

    def MovingAverage(self, sensors_X, sensors_Y):
        x_est = self.DMD_Prediction_A @ self.x_est + self.DMD_Prediction_B @ sensors_Y + self.DMD_Prediction_k*(self.x - self.x_est)

        self.x = sensors_X
        self.x_est = x_est
        self.Buffer_est.append(x_est)
        self.Buffer_act.append(self.x)

        summation = np.zeros((len(x_est), len(x_est)))

        for index in range(len(self.Buffer_act)):
            Actual_Sensor = self.Buffer_act[index]
            Estimated_Sensor = self.Buffer_est[index]
            dif = (Actual_Sensor - Estimated_Sensor)
            dif = np.clip(dif, -1e6, 1e6)
            summation += (dif) @ ((dif).T)

        #! This is not the same as the original article
        V = 1/SET_PARAMS.MovingAverageSizeOfBuffer * summation # Alternatively summation (without np.sum or np.abs)

        return V