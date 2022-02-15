import numpy as np
from Simulation.Parameters import SET_PARAMS
import time
from Simulation.Disturbances import Disturbances
from Simulation.utilities import crossProduct

Ts = SET_PARAMS.Ts

def Transformation_matrix(q):
    q1, q2, q3, q4 = q
    A = np.zeros((3,3))
    A[0,0] = q1**2-q2**2-q3**2+q4**2
    A[0,1] = 2*(q1*q2 + q3*q4)
    A[0,2] = 2*(q1*q3 - q2*q4)
    A[1,0] = 2*(q1*q2 - q3*q4)
    A[1,1] = -q1**2+q2**2-q3**2+q4**2
    A[1,2] = 2*(q2*q3 + q1*q4)
    A[2,0] = 2*(q1*q3 + q2*q4)
    A[2,1] = 2*(q2*q3 - q1*q4)
    A[2,2] = -q1**2-q2**2+q3**2+q4**2
    return A

# ! Update Q_k, R_k
# ! Qk - System Noise Covariance Matrix page 120. Look at B.27
# ! Rk - Measurement Noise Covariance Matrix

class EKF():
    def __init__(self):
        self.angular_noise = SET_PARAMS.RW_sigma/3600

        self.measurement_noise =  0.001

        self.process_noise = self.angular_noise

        self.P_k = SET_PARAMS.P_k

        self.sigma_k = np.eye(7)

        self.w_bi = SET_PARAMS.wbi

        self.w_bo = SET_PARAMS.wbo # Angular velocity in SBC

        self.q = SET_PARAMS.quaternion_initial

        self.x_k = np.concatenate((self.w_bi, self.q), axis = 0).T

        self.Ix = SET_PARAMS.Ix                     # Ixx inertia
        self.Iy = SET_PARAMS.Iy                     # Iyy inertia
        self.Iz = SET_PARAMS.Iz                     # Izz inertia
        self.Inertia = np.identity(3)*np.array(([self.Ix, self.Iy, self.Iz]))
        self.Ixyz = np.array([self.Ix, self.Iy, self.Iz], dtype = "float64")

        self.kgx, self.kgy, self.kgz = SET_PARAMS.kgx, SET_PARAMS.kgy, SET_PARAMS.kgz
        self.k = np.array([self.kgx, self.kgy, self.kgz], dtype = "float64")

        self.R_k, self.m_k = measurement_noise_covariance_matrix(self.measurement_noise)       # standard deviation

        self.Q_wt = system_noise_covariance_matrix(self.angular_noise)
        self.Q_k = SET_PARAMS.Q_k
        self.R_k = SET_PARAMS.R_k

        self.wo = SET_PARAMS.wo
        self.angular_momentum_wheels = SET_PARAMS.initial_angular_wheels
        self.t = SET_PARAMS.time
        self.dt = Ts                  # Time step
        self.dh = self.dt/SET_PARAMS.NumberOfIntegrationSteps                        # Size of increments for Runga-kutta method
        self.dist = Disturbances(None)

        self.Inertia_Inverse = np.linalg.inv(self.Inertia)


    def Kalman_update(self, vmeas_k, vmodel_k, Nm, Nw, t):
        if self.t != t or self.t == SET_PARAMS.time:
            #* Model update
            self.x_k_est, self.w_bi_est, self.q_est, self.angular_momentum_wheels, self.A_ORC_to_SBC_est = self.Model_update(self.q, self.w_bi, self.w_bo, self.angular_momentum_wheels, Nw, Nm)
            
            #* Peripherials after model update
            self.P_k_est, self.K_k = self.Peripherals_update(self.P_k, vmodel_k, self.w_bi_est, self.q_est, self.angular_momentum_wheels, self.A_ORC_to_SBC_est)
            
            self.t = t
        else:
            #* Update H_est with specific model
            H_k_est = Jacobian_H(self.q_est, vmodel_k)

            #* Update K_k with specific H_k_est
            self.K_k = Jacobian_K(self.P_k_est, H_k_est, self.R_k)
        
        #* Measurement update
        x_k_updated, w_bo, P_k, q_updated, w_bi_updated = self.Measurement_update(self.P_k_est, self.x_k_est, self.K_k, self.A_ORC_to_SBC_est, vmeas_k, vmodel_k)

        self.P_k = P_k

        # Updated self.q and self.w_bi with the measurement update
        self.q, self.w_bi, self.w_bo = q_updated, w_bi_updated, w_bo

        return self.A_ORC_to_SBC_est, x_k_updated, w_bo, self.P_k, self.angular_momentum_wheels, self.K_k

    def Model_update(self, q, w_bi, w_bo, angular_momentum_wheels, Nw, Nm):
        ########################################################################
        # THE UPDATED ESTIMATION OF THE QUATERNION MATRIX (ALREADY NORMALIZED) #
        ########################################################################
        q_est = rungeKutta_q(w_bo, 0, q, self.dt, self.dh)

        ###################################################
        # CALCULATES THE ORC TO SBC TRANSFORMATION_MATRIX #
        ###################################################
        A_ORC_to_SBC_est = Transformation_matrix(q_est)

        Ngg = self.dist.Gravity_gradient_func(A_ORC_to_SBC_est) 

        w_bi_est, angular_momentum_wheels_est = rungeKutta_w(self.Inertia, self.Inertia_Inverse, 0, w_bi, self.dt, self.dh, angular_momentum_wheels, Nw, Nm, Ngg, SET_PARAMS.h_ws_max , SET_PARAMS.wheel_angular_d_max)

        #################################
        # PRINTS ERROR IF NAN IN SELF.Q #
        #################################  
        error_message(q_est)

        #######################################################
        # AFTER BOTH THE QUATERNIONS AND THE ANGULAR VELOCITY #
        #  IS CALCULATED, THE STATE VECTOR CAN BE CALCULATED  #
        #######################################################
        x_k_est = np.concatenate((w_bi_est, q_est), axis = 0).T

        return x_k_est, w_bi_est, q_est, angular_momentum_wheels_est, A_ORC_to_SBC_est



    def Peripherals_update(self, P_k, vmodel_k, w_bi_est, q_est, angular_momentum_wheels_est, A_ORC_to_SBC_est):
        #################################################################
        # CALCULATES THE ANGULAR VELOCITY FOR THE SATELITE IN ORC FRAME #
        #################################################################
        w_bo = w_bi_est - A_ORC_to_SBC_est @ np.array(([0,-self.wo,0]))

        #############################################################################
        # PROVIDES THE MATRIX THAT IS USED FOR MULTIPLE CALCULATIONS FROM SELF.W_BO #
        #############################################################################
        omega_k = omega_k_function(w_bo)

        ############################################################
        # THE CONTINUOUS SYSTEM PERTURBATION (JACOBIAN MATRIX F_T) #
        ############################################################
        F_t, TL, TR, BL, BR = F_t_function(angular_momentum_wheels_est, w_bi_est, q_est, omega_k, A_ORC_to_SBC_est, self.Ixyz, self.k, self.wo)
        T11, T12, T21, T22 = TL, TR, BL, BR

        ####################################
        # THE DISCRETE SYSTEM PERTURBATION #
        ####################################
        sigma_k = sigma_k_function(F_t)

        ################################################################################
        #         CALCULATING THE SYSTEM NOISE COVARIANCE MATRIX. THIS MATRIX          #
        # CAN EITHER BE FIXED AT INITIATION OR CALCULATED BASED ON THE CURRENT F_T AND #
        #                THE NOISE OF THE ANGULAR VELOCITY (SELF.Q_WT)                 #
        ################################################################################
        # if self.t == SET_PARAMS.time:
        #     self.Q_k = system_noise_covariance_matrix_discrete(T11, T12, T21, T22, self.Q_wt, SET_PARAMS.RW_sigma)
        #     print(self.Q_k)

        ##########################################################
        # CALCULATE THE MEASUREMENT PERTURBATION MATRIX FROM THE #
        #   ESTIMATED STATE VECTOR (JACOBIAN MATRIX SELF.H_K)    #
        ##########################################################
        H_k_est = Jacobian_H(q_est, vmodel_k)

        ###################################################
        # CALCULATE THE ESTIMATED STATE COVARIANCE MATRIX #
        ###################################################
        P_k_est = state_covariance_matrix(self.Q_k, P_k, sigma_k)

        #################################
        # CALCULATE THE GAIN MATRIX K_K #
        #################################
        K_k = Jacobian_K(P_k_est, H_k_est, self.R_k)

        error_message(K_k)

        return P_k_est, K_k

    def Measurement_update(self, P_k_est, x_k_est, K_k, A_ORC_to_SBC_est, vmeas_k, vmodel_k):
        #####################################################################
        # CALCULATE THE DIFFERENCE BETWEEN THE MODELLED AND MEASURED VECTOR #
        #####################################################################
        e_k = e_k_function(vmeas_k, A_ORC_to_SBC_est, vmodel_k)

        x_k_updated = state_measurement_update(x_k_est, K_k, e_k)

        ########################################################
        # IF ANY VALUE WITHIN THE STATE VECTOR IS EQUAL TO NAN #
        ########################################################
        error_message(x_k_updated)
        
        ##################################################
        # CALCULATE THE MEASUREMENT PERTURBATE ESTIMATED #
        ##################################################
        q_updated = x_k_updated[3:]

        H_k_updated = Jacobian_H(q_updated, vmodel_k)

        P_k = update_state_covariance_matrix(K_k, H_k_updated, P_k_est, self.R_k)

        ###################################
        # NORMALIZE THE QUATERNION MATRIX #
        ###################################
        q_updated = q_updated/np.linalg.norm(q_updated)
        x_k_updated[3:] = q_updated
        w_bi_updated = x_k_updated[:3]
        A_ORC_to_SBC_updated = Transformation_matrix(q_updated)
        w_bo = w_bi_updated - A_ORC_to_SBC_updated @ np.array(([0,-self.wo,0]))

        return x_k_updated, w_bo, P_k, q_updated, w_bi_updated


def error_message(variable):
    if np.isnan(variable).any() and SET_PARAMS.printBreak:
        print("Break")    

#@njit
def system_noise_covariance_matrix_discrete(T11, T12, T21, T22, Q_wt, RW_sigma):
    TL = Ts*Q_wt
    TR = 0.5 * (Ts**2) * (Q_wt @ T21.T)
    BL = 0.5 * (Ts**2) * (T21 @ Q_wt)
    BR = (1/3) * (Ts**3) * (T21 @ Q_wt @ T21.T)
    T = np.concatenate((TL, TR), axis = 1)

    B = np.concatenate((BL, BR), axis = 1)

    Q_k = np.concatenate((T, B))
    # S1 = np.diag([RW_sigma**2, RW_sigma**2, RW_sigma**2, 0, 0, 0, 0])

    # TL = Q_wt @ T11.T + T11 @ Q_wt 
    # TR = Q_wt @ T21.T
    # BL = T21 @ Q_wt
    # BR = np.zeros((4,4), dtype = 'float64')
    
    # T = np.concatenate((TL, TR), axis = 1)

    # B = np.concatenate((BL, BR), axis = 1)

    # S2 = np.concatenate((T, B))

    # TL = T11 @ Q_wt @ T11.T
    # TR = T11 @ Q_wt @ T21.T
    # BL = T21 @ Q_wt @ T11.T
    # BR = T21 @ Q_wt @ T21.T
    
    # T = np.concatenate((TL, TR), axis = 1)

    # B = np.concatenate((BL, BR), axis = 1)

    # S3 = np.concatenate((T, B))

    # Q_k = Ts*S1 + 0.5 * Ts**2 * S2 + (1/3) * Ts**3 * S3

    return Q_k

#@njit
def omega_k_function(w_bo):
    wx, wy, wz = w_bo

    W = np.array(([0, wz, -wy, wx], 
                  [-wz, 0, wx, wy], 
                  [wy, -wx, 0, wz], 
                  [-wx, -wy, -wz, 0]))
    return W

#@njit
def kq_function(w_bo):
    wx, wy, wz = w_bo
    w_bo_norm = np.sqrt(wx**2 + wy**2 + wz**2)
    kq = Ts/2 * w_bo_norm
    return kq, w_bo_norm


def measurement_noise_covariance_matrix(measurement_noise):
    m_k = np.array(([[measurement_noise], [measurement_noise], [measurement_noise]]), dtype = "float64")
    R_k = np.diag([measurement_noise, measurement_noise, measurement_noise]) #** 2
    return R_k, m_k


def system_noise_covariance_matrix(angular_noise):
    Q_t = np.diag([angular_noise,angular_noise,angular_noise]) ** 2
    return Q_t

#@njit
def Jacobian_H(q, vmodel_k):
    q1, q2, q3, q4 = q

    vmodel_k = np.reshape(vmodel_k, (3,1))

    zero3 = np.zeros((3,3))
    h1 = 2 * np.array(([[q1, q2, q3], [q2, -q1, q4], [q3, -q4, -q1]])) @ vmodel_k
    h2 = 2 * np.array(([[-q2, q1, -q4], [q1, q2, q3], [q4, q3, -q2]])) @ vmodel_k
    h3 = 2 * np.array(([[-q3, q4, q1],[-q4, -q3, q2],[q1, q2, q3]])) @ vmodel_k
    h4 = 2 * np.array(([[q4, q3, -q2],[-q3, q4, q1],[q2, -q1, q4]])) @ vmodel_k
    H_k = np.concatenate((zero3, h1, h2, h3, h4), axis = 1)

    return H_k

#@njit
def delta_angular(Inertia, Nm, Nw, Ngyro, Ngg):
    return Inertia @ (Nm - Nw - Ngyro - Ngg)

#@njit
def state_model_update(delta_angular, x_prev):
    return x_prev + (Ts/2) * (3 * delta_angular - delta_angular)

#@njit
def state_model_update_quaternion(q, kq, omega_k, w_ob):
    return ((np.cos(kq) * np.eye(4) + (1/w_ob)*np.sin(kq)*omega_k) @ q).flatten()

#@njit
def Jacobian_K(P_k, H_k, R_k):
    K_k = P_k @ (H_k.T) @ np.linalg.inv(H_k @ P_k @ H_k.T + R_k) #! np.linalg.inv(H_k @ P_k @ H_k.T + R_k)
    return K_k

#@njit
def state_covariance_matrix(Q_k, P_k, sigma_k):
    P_k = sigma_k @ P_k @ sigma_k.T + Q_k
    return P_k

#@njit
def update_state_covariance_matrix(K_k, H_k, P_k, R_k):
    P_k = (np.eye(7) - K_k @ H_k) @ P_k @ (np.eye(7) - K_k @ H_k).T + K_k @ R_k @ K_k.T #! (np.eye(7) - K_k @ H_k) @ P_k @ np.linalg.inv(np.eye(7) - K_k @ H_k) + K_k @ R_k @ K_k.T
    return P_k

#@njit
def state_measurement_update(x_k, K_k, e_k):
    x_k = x_k + K_k @ e_k
    return x_k

#@njit
def e_k_function(vmeas_k, A, vmodel_k):
    vmodel_k_SBC = A @ vmodel_k
    e_k = vmeas_k - vmodel_k_SBC
    # print(e_k)
    return e_k

#@njit
def sigma_k_function(F_t):
    sigma_k = np.eye(7) + Ts*F_t + (0.5 * Ts**2 * np.linalg.matrix_power(F_t, 2)) + (1/3 * Ts**3 * np.linalg.matrix_power(F_t,3))
    return sigma_k

#@njit
def F_t_function(h, wi, q, omega_k, A, Inertia, kg, wo):
    wx, wy, wz = wi
    hx, hy, hz = h

    Ix, Iy, Iz = Inertia

    a11 = 0
    a12 = (wz*(Iy - Iz) - hz)/Ix
    a13 = (wy*(Iy - Iz) + hy)/Ix

    a21 = (wz*(Iz - Ix) + hz)/Iy
    a22 = 0
    a23 = (wx*(Iz - Ix) - hx)/Iy

    a31 = (wy*(Ix - Iy) - hy)/Iz
    a32 = (wx*(Ix - Iy) + hx)/Iz
    a33 = 0
    
    TL = np.array(([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]))

    kgx, kgy, kgz = kg
    
    K = np.array(([[2*kgx, 0, 0],
                    [0 , 2*kgy, 0], 
                    [0 , 0 , 2*kgz]]))

    q1, q2, q3, q4 = q

    A13 = A[0, 2]
    A23 = A[1, 2]
    A33 = A[2, 2]

    d1 = np.array(([[(-q1*A23 + q4*A33)/Ix], [(-q1*A13 + q3*A33)/Iy], [(q3*A23 + q4*A13)/Iz]]))

    d2 = np.array(([[(-q2*A23 + q3*A33)/Ix], [(-q2*A13 - q4*A33)/Iy], [(-q4*A23 + q3*A13)/Iz]]))

    d3 = np.array(([[(q3*A23 + q2*A33)/Ix], [(q3*A13 + q1*A33)/Iy], [(q1*A23 + q2*A13)/Iz]]))

    d4 = np.array(([[(q4*A23 + q1*A33)/Ix], [(q4*A13 - q2*A33)/Iy], [(-q2*A23 + q1*A13)/Iz]]))

    D = np.concatenate((d1, d2, d3, d4), axis = 1)

    TR = K @ D
    
    BL = 0.5 * np.array(([[q4, -q3, q2],[q3, q4, -q1],[-q2, q1, q4],[-q1, -q2, -q3]]))

    BR = 0.5 * omega_k + wo * np.array(([[q1*q3, q1*q4, 1 - q1**2, -q1*q2],
                                                    [q2*q3, q2*q4, -q1*q2, 1-q2**2],
                                                    [-(1-q3**2), q3*q4, -q1*q3, -q2*q3],
                                                    [q3*q4, -(1-q4**2), -q1*q4, -q2*q4]]))

    T = np.concatenate((TL, TR), axis = 1)

    B = np.concatenate((BL, BR), axis = 1)

    Ft = np.concatenate((T, B))

    return Ft, TL, TR, BL, BR

#@njit
def rungeKutta_h(x0, angular, x, h, N_control, h_ws_max):
    angular_momentum_derived = N_control
    n = int(np.round((x - x0)/h))

    y = angular + angular_momentum_derived * (h)
    # for _ in range(n):
    #     k1 = h*(angular_momentum_derived) 
    #     k2 = h*((angular_momentum_derived) + 0.5*k1) 
    #     k3 = h*((angular_momentum_derived) + 0.5*k2) 
    #     k4 = h*((angular_momentum_derived) + k3) 

    #     y = y + (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    #     x0 = x0 + h; 
    
    y = np.clip(y, -h_ws_max, h_ws_max)

    return y

########################################################################################
# FUNCTION TO CALCULATE THE SATELLITE ANGULAR VELOCITY BASED ON THE DERIVATIVE THEREOF #
########################################################################################
#@njit
def rungeKutta_w(Inertia, InvInertia, x0, w, x, h, angular_momentum_wheels, Nw, Nm, Ngg, h_ws_max, wheel_angular_d_max):  

    ######################################################
    # CONTROL TORQUES IMPLEMENTED DUE TO THE CONTROL LAW #
    ######################################################
    n = int(np.round((x - x0)/h))
    y = w

    ######################################################
    # ALL THE DISTURBANCE TORQUES ADDED TO THE SATELLITE #
    ######################################################
    x01 = x0

    for _ in range(n):    
        N_gyro = crossProduct(y,(Inertia @ y + angular_momentum_wheels))
        N = - Nw - N_gyro + Nm + Ngg
        k1 = h*((InvInertia @ N)) 
        k2 = h*((InvInertia @ N) + 0.5*k1) 
        k3 = h*((InvInertia @ N) + 0.5*k2) 
        k4 = h*((InvInertia @ N) + k3) 
        y = y + (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        
        x0 = x0 + h; 

        angular_momentum_wheels = rungeKutta_h(x01, angular_momentum_wheels, x, h, Nw, h_ws_max)

    y = np.clip(y, -SET_PARAMS.angularSatelliteMax, SET_PARAMS.angularSatelliteMax)

    return y, angular_momentum_wheels

###########################################################################################
# FUNCTION TO CALCULATE THE SATELLITE QUATERNION POSITION BASED ON THE DERIVATIVE THEREOF #
###########################################################################################
#@njit
def rungeKutta_q(w_bo, x0, y0, x, h):      
    wx, wy, wz = w_bo
    n = int(np.round((x - x0)/h))

    y = y0

    W = np.array(([[0, wz, -wy, wx], [-wz, 0, wx, wy], [wy, -wx, 0, wz], [-wx, -wy, -wz, 0]]), dtype = 'float64')
    for _ in range(n):
        k1 = h*(0.5 * W @ y)
        k2 = h*(0.5 * W @ (y + 0.5*k1))
        k3 = h*(0.5 * W @ (y + 0.5*k2))
        k4 = h*(0.5 * W @ (y + k3))

        y = y + (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        x0 = x0 + h; 
    
    norm_y = np.linalg.norm(y)
    y = y/norm_y
    
    if (np.isnan(y).any() or (y == 0).all()) and SET_PARAMS.printBreak:
        print("Break")

    return y

if __name__ == "__main__":
    ekf = EKF()
    v_k = np.zeros((3,))
    Nm = np.array(([0, 1, 0])).T
    Nw = np.array(([0.1, 0.1, -0.1])).T
    Ngyro = np.array(([-0.3, 0.15, 0])).T
    Ngg = np.array(([-0.3, 0.15, 0])).T
    A = np.array(([[-1.0, 0.0, 0.0], [0, -1., 0.0], [0., 0., 1.]]))
    vmodel_k = SET_PARAMS.star_tracker_vector
    vmodel_k = np.reshape(vmodel_k,(3,1))
    vmeas_k = A @ vmodel_k
    vmeas_k = vmeas_k/np.linalg.norm(vmeas_k)
    dt = 1
    for i in range(100):
        ekf.Kalman_update(vmeas_k, vmodel_k, Nm, Nw, Ngyro, Ngg, dt)