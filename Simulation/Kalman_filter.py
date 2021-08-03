import numpy as np
from Simulation.Parameters import SET_PARAMS
Ts = 1

class RKF():
    def __init__(self):
        self.angular_noise = SET_PARAMS.RW_sigma

        self.measurement_noise =  0.5

        self.P_k = np.eye(3)

        self.sigma_k = np.eye(3)

        self.x_k = SET_PARAMS.wbi

        self.Inertia = SET_PARAMS.Inertia

        self.R_k, self.m_k = measurement_noise_covariance_matrix(0.5)
        self.Q_k = system_noise_covariance_matrix(self.angular_noise)

        self.t = SET_PARAMS.time

    def Kalman_update(self, v_k, Nm, Nw, Ngyro, t):
        # Model update
        H_k = Jacobian_H(v_k)

        if self.t != t or self.t == SET_PARAMS.time:
            # Model update
            self.t = t     
            self.w_b = delta_angular(self.Inertia, Nm, Nw, Ngyro)
            self.x_k_update = state_model_update(self.w_b, self.x_k)
            self.P_k_update = state_covariance_matrix(self.Q_k, self.P_k, self.sigma_k)

        # Measurement update
        y_k = measurement_state_y(H_k, self.w_b, self.m_k)
        K_k = Jacobian_K(self.P_k_update, H_k, self.R_k)
        self.P_k = update_state_covariance_matrix(K_k, H_k, self.P_k_update)
        self.x_k = state_measurement_update(self.x_k_update, K_k, y_k, H_k)
        if (self.x_k == np.nan).any():
            print("break")
        return self.x_k

def measurement_noise_covariance_matrix(measurement_noise):
    m_k = np.array(([[measurement_noise], [measurement_noise], [measurement_noise]]))
    R_k = np.diag([measurement_noise, measurement_noise, measurement_noise]) #** 2
    R_k = np.array([[0.5, 0, 0],[0, 0.5, 0],[0, 0, 0.5]]) #** 2
    #R_k = np.eye(3)
    return R_k, m_k

def system_noise_covariance_matrix(angular_noise):
    Q_k = np.diag([angular_noise,angular_noise,angular_noise]) ** 2
    Q_k = np.array([[0.707, 0.2, 0.5],[0.5, 0.707, 0.2],[0.2, 0.5, 0.707]]) #** 2
    Q_k = np.eye(3)
    return Q_k

def Jacobian_H(v_k):
    vx, vy, vz = v_k[:,0]
    H_k = np.array(([[0, Ts*vz, -Ts*vy], [-Ts*vz, 0, Ts*vx], [Ts*vy, -Ts*vx, 0]]))
    return H_k

def delta_angular(Inertia, Nm, Nw, Ngyro):
    return Inertia @ (Nm - Nw - Ngyro)

def state_model_update(delta_angular, x_prev):
    y = x_prev + (Ts/2) * (3 * delta_angular - delta_angular)
    return y

def state_covariance_matrix(Q_k, P_k, sigma_k):
    P_k = sigma_k @ P_k @ sigma_k.T + Q_k
    return P_k

def Jacobian_K(P_k, H_k, R_k):
    K_k = P_k @ H_k.T @ np.linalg.inv(H_k @ P_k @ H_k.T + R_k)
    return K_k

def update_state_covariance_matrix(K_k, H_k, P_k):
    P_k = (np.eye(3) - K_k @ H_k) @ P_k
    return P_k

def state_measurement_update(x_k, K_k, y_k, H_k):
    x_k = x_k + K_k @ (y_k - H_k @ x_k)
    return x_k

def measurement_state_y(H_k, w_b, m_k):
    y_k = H_k @ w_b + m_k
    return y_k


if __name__ == "__main__":
    rkf = RKF()
    v_k = np.zeros((3,))
    Nm = np.array(([0, 1, 0])).T
    Nw = np.array(([0.1, 0.1, -0.1])).T
    Ngyro = np.array(([-0.3, 0.15, 0])).T
    for i in range(100):
        rkf.Kalman_update(v_k, Nm, Nw, Ngyro)