from Simulation.Parameters import SET_PARAMS
import numpy as np
from Simulation.Earth_model import orbit
from Simulation.Sensors import Sensors

class Disturbances:
    def __init__(self, sense):
        self.phi_s = np.zeros((3,1)) #arbitrary phase
        self.phi_d = np.zeros((3,1)) #arbitrary phase
        self.position_vector_of_wheels = np.identity(3)*SET_PARAMS.Dimensions/2 #radius from COM
        self.orbit = orbit()
        self.sense = sense

    def Gravity_gradient_func(self, A):
        zoB = A * np.array(([[0],[0],[1]]))
        kgx = SET_PARAMS.kgx
        kgy = SET_PARAMS.kgy
        kgz = SET_PARAMS.kgz
        Ngg = np.array(([kgx*zoB[1,2]*zoB[2,2]],[kgy*zoB[0][2]*zoB[2][2]],[kgz*zoB[0,2]*zoB[1,2]]))
        
        return Ngg

    def Aerodynamic(self, DCM, EIC_to_ORC, sun_in_view):
        r_sat = np.array(([self.sense.r_sat]))
        v_A_EIC = np.matmul(np.array(([[0],[0],[SET_PARAMS.w_earth]])),r_sat)
        v_ORC = np.matmul(EIC_to_ORC,v_A_EIC)
        v_ab = np.matmul(DCM,v_ORC)
        Ai = SET_PARAMS.Surface_area_i
        alpha_i = SET_PARAMS.incidence_angle
        h, h_o, H = [Sensors.current_height_above_earth, SET_PARAMS.Height_above_earth_surface, SET_PARAMS.Scale_height]
        sigma_t = SET_PARAMS.tangential_accommodation      # tangential_accommodation
        sigma_n = SET_PARAMS.normal_accommodation  # normal_accommodation
        S = SET_PARAMS.ratio_of_molecular_exit        # ratio_of_molecular_exit
        r_pi = SET_PARAMS.offset_vector
        n_i = SET_PARAMS.unit_normal_vector
        p_o = SET_PARAMS.atmospheric_reference_density
        if sun_in_view:
            p = 0.5 * (p_o * np.exp(-(h-h_o)/H))
        else:
            p = (p_o * np.exp(-(h-h_o)/H))

        va = np.matmul(np.array(([0],[0],[SET_PARAMS.w_earth])), r_sat) - SET_PARAMS.v_sat
        N_aero = []
        for i in range(3):
            N_aero.append(p * np.linalg.norm(va)**2 * Ai[i] * np.heaviside(np.cos(alpha_i))*np.cos(alpha_i)*sigma_t*(r_pi) + (sigma_n * S + (2-sigma_n - sigma_t)*np.cos(alpha_i)*(r_pi)))
        
        N_aero = np.array((N_aero))    

        return N_aero

    def static(self, rotation_rate, t):
        ###############################################################################
        # STATIC IMBALANCE WITH ONE MASS A DISTANCE R FROM THE CENTRE OF THE FLYWHEEL #
        #                         M - MASS OF IMBALANCE IN KG                         #
        #                  R - DISTANCE FROM CENTRE OF FLYWHEEL IN M                  #
        ###############################################################################

        Us = 2.08e-7 #For the RW-0.06 wheels in kg/m; Us = m*r; Assume all the wheels are equally imbalanced
        omega = rotation_rate #rad/s rotation rate of wheel   
        F_xs = Us * omega[0,0]**2 * np.array(([np.sin(omega[0,0]*t + self.phi_s[0,0]),np.cos(omega[0,0]*t + self.phi_s[0,0]), 0]))
        F_ys = Us * omega[1,0]**2 * np.array(([np.cos(omega[1,0]*t + self.phi_s[1,0]), 0, np.sin(omega[1,0]*t + self.phi_s[1,0])]))
        F_zs = Us * omega[2,0]**2 * np.array(([0, np.sin(omega[2,0]*t + self.phi_s[2,0]), np.cos(omega[2,0]*t + self.phi_s[2,0])]))
        F = np.array((F_xs, F_ys, F_zs))
        self.phi_s = omega*t + self.phi_s
        N_xs = np.matmul(self.position_vector_of_wheels[0,:], F)
        N_ys = np.matmul(self.position_vector_of_wheels[1,:], F)
        N_zs = np.matmul(self.position_vector_of_wheels[2,:], F)
        N_s = np.array(([N_xs[0] + N_ys[0] + N_zs[0], N_xs[1] + N_ys[1] + N_zs[1], N_xs[2] + N_ys[2] + N_zs[2]]))
        return N_s

    def dynamic(self, rotation_rate, t):
        Ud = 2.08e-9        #For the RW-0.06 wheels in kg/m^2; Ud = m*r*d
        omega = rotation_rate   #rad/s rotation rate of wheel  
        N_xd = Ud * omega[0,0]**2 * np.array(([np.sin(omega[0,0]*t + self.phi_d[0,0]),np.cos(omega[0,0]*t + self.phi_d[0,0]), 0]))
        N_yd = Ud * omega[1,0]**2 * np.array(([np.cos(omega[1,0]*t + self.phi_d[1,0]), 0, np.sin(omega[1,0]*t + self.phi_d[1,0])]))
        N_zd = Ud * omega[2,0]**2 * np.array(([0, np.sin(omega[2,0]*t + self.phi_d[2,0]), np.cos(omega[2,0]*t + self.phi_d[2,0])]))
        self.phi_d = omega*t + self.phi_d 
        N_d = N_xd[0] + N_yd[0] + N_zd[0]
        N_d = np.array(([N_xd[0] + N_yd[0] + N_zd[0], N_xd[1] + N_yd[1] + N_zd[1], N_xd[2] + N_yd[2] + N_zd[2]]))
        return N_d

    def Wheel_Imbalance(self, rotation_rate, t):
        ##############################################################################
        # DYNAMIC IMBALANCE WITH TWO MASSES SEPERATED BY 180 DEGREES AND DISTANCE, D #
        #                         D - WIDTH OF FLYWHEEL IN M                         #
        #                         R - DISTANCE FROM FLYWHEEL                         #
        #                               M - MASS IN KG                               #
        ##############################################################################
        return (self.static(rotation_rate, t) + self.dynamic(rotation_rate, t))
