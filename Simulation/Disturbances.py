from Simulation.Parameters import SET_PARAMS
import numpy as np
from Simulation.Earth_model import orbit
from Simulation.Sensors import Sensors

class Disturbances:
    def __init__(self, sense):
        self.phi_sx = 0 #arbitrary phase
        self.phi_sy = 0
        self.phi_sz = 0
        self.phi_dx = 0
        self.phi_dy = 0
        self.phi_dz = 0
        
        self.position_vector_of_wheelx = np.array(([SET_PARAMS.Lx/2,0,0]))
        self.position_vector_of_wheely = np.array(([0,SET_PARAMS.Ly/2,0]))
        self.position_vector_of_wheelz = np.array(([0,0,SET_PARAMS.Lz/2]))

        self.wo = SET_PARAMS.wo

        self.surfaceI = SET_PARAMS.surfaceI

        # {
        #         'z_positive': {'Area': Lx * Ly, 'CoM-CoP': np.array([0,0,1]), 'NormalVector': np.array([0,0,1])},
        #         'z-negative': {'Area': Lx * Ly, 'CoM-CoP': np.array([0,0,-1]), 'NormalVector': np.array([0,0,-1])},
        #         'y-positive': {'Area': Lz * Lx, 'CoM-CoP': np.array([0,1,0]), 'NormalVector': np.array([0,1,0])},
        #         'y-negative': {'Area': Lz * Lx, 'CoM-CoP': np.array([0,-1,0]), 'NormalVector': np.array([0,-1,0])},
        #         'x-positive': {'Area': Lz * Ly, 'CoM-CoP': np.array([1,0,0]), 'NormalVector': np.array([1,0,0])},
        #         'x-negative': {'Area': Lz * Ly, 'CoM-CoP': np.array([-1,0,0]), 'NormalVector': np.array([-1,0,0])},
        #         'SolarPanelxpyp': {'Area': SP_Length * SP_width, 'CoM-CoP': np.array([Lx/2 + SP_Length/2, Ly/2 + SP_width/2, Lz/2]), 'NormalVector': np.array([0,0,1])},
        #         'SolarPanelxpyn': {'Area': SP_Length * SP_width, 'CoM-CoP': np.array([Lx/2 + SP_Length/2, -(Ly/2 + SP_width/2), Lz/2]), 'NormalVector': np.array([0,0,1])},
        #         'SolarPanelxnyp': {'Area': SP_Length * SP_width, 'CoM-CoP': np.array([-(Lx/2 + SP_Length/2), Ly/2 + SP_width/2, Lz/2]), 'NormalVector': np.array([0,0,1])},
        #         'SolarPanelxnyn': {'Area': SP_Length * SP_width, 'CoM-CoP': np.array([-(Lx/2 + SP_Length/2), -(Ly/2 + SP_width/2), Lz/2]), 'NormalVector': np.array([0,0,1])}
        #     }

        self.orbit = orbit()
        self.sense = sense

    def Gravity_gradient_func(self, A):
        zoB = A @ np.array(([0,0,1])).T
        Ngg = 3 * SET_PARAMS.wo**2 * (np.cross(zoB, SET_PARAMS.Inertia@zoB))

        return Ngg

    def Aerodynamic2(self, A_ORC_to_SBC, A_EIC_to_ORC, sun_in_view):
        
        v_AB = A_ORC_to_SBC @ A_EIC_to_ORC @ (np.cross(np.array([0, 0, -self.wo]), self.sense.position) - self.sense.velocity)

        normv_AB = np.linalg.norm(v_AB)

        unit_v_AB = v_AB/normv_AB

        h, h_o, H = self.sense.altitude[0,0], SET_PARAMS.Height_above_earth_surface, SET_PARAMS.Scale_height

        p_o = SET_PARAMS.atmospheric_reference_density

        hft = float(h) * 3.28084

        #! if sun_in_view:
        #!     p = 0.5 * (p_o * np.exp(-(h-h_o)/H))
        #! else:
        #!     p = p_o * np.exp(-(h-h_o)/H)

        #* According to https://www.grc.nasa.gov/www/k-12/rocket/atmos.html the model for the density of the atmosphere
        #* This is calculate with ft and then the pressure is converted back to kg/m3
        if sun_in_view:
            T = -205.05 + 0.00164 * hft
            p = 0.5 * (51.97 * ((T + 459.7)/389.98)**(-11.388))/(0.00194032)
        else:
            T = -205.05 + 0.00164 * hft
            p = (51.97 * ((T + 459.7)/389.98)**(-11.388))/(0.00194032)

        sigma_t = 0.4 #! Changed from 0.2 to 0.4
        sigma_n = 0.4

        S = 0.05

        N_aero = np.zeros(3)

        for areaI in self.surfaceI:
            Ai = self.surfaceI[areaI]["Area"]
            ri = self.surfaceI[areaI]["CoM-CoP"]
            ni = self.surfaceI[areaI]["NormalVector"] 

            ni = ni/np.linalg.norm(ni)

            cosa = np.dot(unit_v_AB, ni)

            if cosa < 0:
                heaviside = 0
            else:
                heaviside = 1

            N_aero += p * normv_AB**2 * Ai * heaviside * cosa * (sigma_t * (np.cross(ri, unit_v_AB)) + (sigma_n * S + (2 - sigma_n - sigma_t)*cosa)*(np.cross(ri, ni)))
            
        return N_aero
        


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
        
        wx, wy, wz = omega

        F_xs = Us * wx**2 * np.array(([np.sin(wx*t+self.phi_sx),np.cos(wx*t + self.phi_sx),0]))
        F_ys = Us * wy**2 * np.array(([np.sin(wy*t+self.phi_sy),np.cos(wy*t + self.phi_sy),0]))
        F_zs = Us * wz**2 * np.array(([np.sin(wz*t+self.phi_sz),np.cos(wz*t + self.phi_sz),0]))
        
        self.phi_sx = wx*t + self.phi_sx
        self.phi_sy = wy*t + self.phi_sy
        self.phi_sz = wz*t + self.phi_sz

        N_xs = np.cross(self.position_vector_of_wheelx,F_xs)
        N_ys = np.cross(self.position_vector_of_wheely,F_ys)
        N_zs = np.cross(self.position_vector_of_wheelz,F_zs)
        N_s = N_xs + N_ys + N_zs

        return N_s

    def dynamic(self, rotation_rate, t):
        ##############################################################################
        # DYNAMIC IMBALANCE WITH TWO MASSES SEPERATED BY 180 DEGREES AND DISTANCE, D #
        #                         D - WIDTH OF FLYWHEEL IN M                         #
        #                         R - DISTANCE FROM FLYWHEEL                         #
        #                               M - MASS IN KG                               #
        ##############################################################################

        Ud = 2.08e-9        #For the RW-0.06 wheels in kg/m^2; Ud = m*r*d
        omega = rotation_rate   #rad/s rotation rate of wheel  
        
        wx, wy, wz = omega

        N_xd = Ud * wx**2 * np.array(([np.sin(wx*t+self.phi_dx),np.cos(wx*t + self.phi_dx),0]))
        N_yd = Ud * wy**2 * np.array(([np.sin(wy*t+self.phi_dy),np.cos(wy*t + self.phi_dy),0]))
        N_zd = Ud * wz**2 * np.array(([np.sin(wz*t+self.phi_dz),np.cos(wz*t + self.phi_dz),0]))
        
        self.phi_dx = wx*t + self.phi_dx
        self.phi_dy = wy*t + self.phi_dy
        self.phi_dz = wz*t + self.phi_dz

        N_d = N_xd + N_yd + N_zd

        return N_d

    def Wheel_Imbalance(self, rotation_rate, t):
        return self.static(rotation_rate, t) + self.dynamic(rotation_rate, t)
