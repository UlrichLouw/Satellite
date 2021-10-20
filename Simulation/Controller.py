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
        self.Earth_sensor_position = SET_PARAMS.Earth_sensor_position
        self.N_max = SET_PARAMS.N_ws_max
        self.first = True
        self.angular_momentum_ref = np.zeros(3)
        self.delay = 0
        self.t = SET_PARAMS.time

    def control(self, w_bi_est, w_est, q, Inertia, B, angular_momentum, earthVector, sunVector, sun_in_view):             
        if SET_PARAMS.Mode == "Nominal":   # Normal operation
            self.q_ref = SET_PARAMS.q_ref
            N_magnet = self.Momentum_dumping(B, angular_momentum)
            
        elif SET_PARAMS.Mode == "EARTH_SUN":
            if sun_in_view:
                self.nadir_pointing = False
                self.q_ref = self.SunCommandQuaternion(sunVector)
                # self.q_ref = np.array(([0,1,0,0]))
                N_magnet = np.zeros(3)
            else:
                if not self.nadir_pointing:
                    self.beginning_of_nadir_pointing = self.t
                    self.delay = 0
                else:
                    self.delay = self.t - self.beginning_of_nadir_pointing
                
                self.nadir_pointing = True
                
                self.q_ref = SET_PARAMS.q_ref
                if self.delay >= 200:
                    N_magnet = self.Momentum_dumping(B, angular_momentum)
                else:
                    N_magnet = np.zeros(3)

        if SET_PARAMS.Mode == "Safe":    # Detumbling mode
            N_magnet = self.B_dot_control(B, w_est)
            N_wheel = np.zeros(3)
        else:
            N_wheel = self.Full_State_Quaternion(w_bi_est, w_est, q, Inertia, angular_momentum)


        self.t += SET_PARAMS.Ts

        return N_magnet, N_wheel
    ########################################################
    # DETERMINE THE COMMAND QUATERNION FOR EARTH FOLLOWING #
    ########################################################
    def EarthCommandQuaternion(self, earthVector):
        u1 = np.cross(self.Earth_sensor_position, earthVector)
        normu1 = np.linalg.norm(u1)
        if normu1 != 0:
            uc = u1/np.linalg.norm(u1)
        else:
            uc = u1

        delta = np.clip(np.dot(self.Earth_sensor_position, earthVector),-1,1)

        q13 = uc * np.sin(delta/2)
        q4 = np.cos(delta/2)
        q_ref = np.array(([q13[0],q13[1],q13[2],q4]))
        return q_ref

    ######################################################
    # DETERMINE THE COMMAND QUATERNION FOR SUN FOLLOWING #
    ######################################################
    def SunCommandQuaternion(self, sunVector):
        u1 = np.cross(self.SolarPanelPosition, sunVector)
        normu1 = np.linalg.norm(u1)
        if normu1 != 0:
            uc = u1/np.linalg.norm(u1)
        else:
            uc = u1

        delta = np.clip(np.dot(self.SolarPanelPosition, sunVector),-1,1)

        q13 = uc * np.sin(delta/2)
        q4 = np.cos(delta/2)
        q_ref = np.array(([q13[0],q13[1],q13[2],q4]))
        return q_ref

    def Full_State_Quaternion(self, w_bi_est, w_est, q, Inertia, angular_momentum):
        #! self.q_error = Quaternion_functions.quaternion_error(q, self.q_ref)
        #! self.q_e = self.q_error[0:3]
        q_error = Quaternion_functions.quaternion_error(q, self.q_ref)
        self.q_e = q_error[0:3]
        w_error = w_est - self.w_ref
        self.w_e = w_error
        N = self.Kp * Inertia @ self.q_e + self.Kd * Inertia @ w_error - np.cross(w_bi_est,(Inertia @ w_bi_est + angular_momentum))
        N = np.clip(N, -self.N_max,self.N_max)
        return N
    
    def B_dot_control(self, B, w):
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
        N = np.matmul(M,B)[1,:]
        N = np.clip(N, -SET_PARAMS.M_magnetic_max, SET_PARAMS.M_magnetic_max)
        return N
    
    def Momentum_dumping(self, B, angular_momentum):
        error = -SET_PARAMS.Kw * (angular_momentum - self.angular_momentum_ref)
        M = np.cross(error,B)/(np.linalg.norm(B)**2)
        return M

    def reinitialize(self):
        self.Kp = SET_PARAMS.Kp
        self.Kd = SET_PARAMS.Kd