import numpy as np 
import math
import Simulation.Quaternion_functions as Quaternion_functions
from Simulation.Parameters import SET_PARAMS

pi = math.pi

class Control:
    def __init__(self):
        self.Kp = SET_PARAMS.Kp
        self.Kd = SET_PARAMS.Kd
        self.w_ref = SET_PARAMS.w_ref
        self.q_ref = SET_PARAMS.q_ref
        self.SolarPanelPosition = SET_PARAMS.SolarPanelPosition
        self.N_max = SET_PARAMS.N_ws_max
        self.first = True

    def control(self, w_bi_est, w_est, q, Inertia, B, angular_momentum, earthVector, sunVector, sun_in_view):             
        if SET_PARAMS.Mode == "Nominal":   # Normal operation
            self.q_ref = SET_PARAMS.q_ref
            
        elif SET_PARAMS.Mode == "EARTH_SUN":
            if sun_in_view:
                self.SunCommandQuaternion(sunVector)
            else:
                self.q_ref = SET_PARAMS.q_ref

        normQ = np.linalg.norm(self.q_ref)

        if normQ != 0:
            self.q_ref = self.q_ref/normQ

        if SET_PARAMS.Mode == "Safe":    # Detumbling mode
            N_magnet = self.magnetic_torquers(B, w_est)
            N_wheel = np.zeros((3,1))
        else:
            N_magnet = np.zeros((3,1))
            N_wheel = self.control_wheel(w_bi_est, w_est, q, Inertia, angular_momentum)

        #N_magnet, N_wheel = np.zeros(N_magnet.shape), np.zeros(N_wheel.shape)
        return N_magnet, N_wheel


    ######################################################
    # DETERMINE THE COMMAND QUATERNION FOR SUN FOLLOWING #
    ######################################################
    def SunCommandQuaternion(self, sunVector):
        u1 = self.SolarPanelPosition * sunVector
        uc = u1/np.linalg.norm(u1)

        delta = np.clip(np.dot(self.SolarPanelPosition, sunVector),-1,1)

        q13 = uc * np.sin(delta/2)
        q4 = np.cos(delta/2)
        self.q_ref = np.array(([[q13[0]],[q13[1]],[q13[2]],[q4]]))


    def control_wheel(self, w_bi_est, w_est, q, Inertia, angular_momentum):
        q_error = Quaternion_functions.quaternion_error(q, self.q_ref)
        w_error = w_est - self.w_ref
        print("Proportional", self.Kp * Inertia @ np.reshape(q_error[0:3],(3,1)))
        print("Derivative", self.Kd * Inertia @ w_error)
        N = self.Kp * Inertia @ np.reshape(q_error[0:3],(3,1)) + self.Kd * Inertia @ w_error - w_bi_est * (Inertia @ w_bi_est + angular_momentum)
        N = np.clip(N, -self.N_max,self.N_max)
        return N
    
    def magnetic_torquers(self, B, w):
        if self.first == True:
            self.first = False
            My = 0.0
            Mx = 0.0
            Mz = 0.0
            Beta = 0.0
        else:
            Beta = np.arccos(B[1]/np.linalg.norm(B))
            My = SET_PARAMS.Kd_magnet * (Beta - self.Beta)/SET_PARAMS.Ts
            if B[2] > B[0]:
                Mx = SET_PARAMS.Ks_magnet * (w[1][0] - self.w_ref[1][0])*np.sign(B[2])
                Mz = 0
            else:
                Mz = SET_PARAMS.Ks_magnet * (self.w_ref[1][0] - w[1][0])*np.sign(B[0])
                Mx = 0

        M = np.array(([[Mx],[My],[Mz]]))
        self.Beta = Beta
        N = np.reshape(np.matmul(M,np.reshape(B,(1,3)))[1,:],(3,1))
        N = np.clip(N, -SET_PARAMS.M_magnetic_max, SET_PARAMS.M_magnetic_max)
        return N
    
    def reinitialize(self):
        self.Kp = SET_PARAMS.Kp
        self.Kd = SET_PARAMS.Kd