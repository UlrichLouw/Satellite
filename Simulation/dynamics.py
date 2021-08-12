import numpy as np
import Simulation.Controller as Controller
from Simulation.Disturbances import Disturbances
import Simulation.Parameters as Parameters
SET_PARAMS = Parameters.SET_PARAMS 
from Simulation.Sensors import Sensors
import Simulation.Quaternion_functions as Quaternion_functions
from Simulation.Kalman_filter import RKF
from Simulation.Extended_KF import EKF

Fault_names_to_num = SET_PARAMS.Fault_names

# The DCM must be calculated depending on the current quaternions
def Transformation_matrix(q):
    q1, q2, q3, q4 = q[:]
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

##############################################################################
# FUNCTION TO CALCULATE THE ANGULAR MOMENTUM BASED ON THE DERIVATIVE THEREOF #
##############################################################################
def rungeKutta_h(x0, angular, x, h, N_control):
    angular_momentum_derived = N_control
    n = int(np.round((x - x0)/h))

    y = angular
    for _ in range(n):
        k1 = h*(angular_momentum_derived) 
        k2 = h*((angular_momentum_derived) + 0.5*k1) 
        k3 = h*((angular_momentum_derived) + 0.5*k2) 
        k4 = h*((angular_momentum_derived) + k3) 

        y = y + (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        x0 = x0 + h; 
    
    return y


class Dynamics:
    def determine_earth_vision(self):
        #################################################################
        #      FOR THIS SPECIFIC SATELLITE MODEL, THE EARTH SENSOR      #
        #                    IS FIXED TO THE -Z FACE                    #
        # THIS IS ACCORDING TO THE ORBIT AS DEFINED BY JANSE VAN VUUREN #
        #             THIS IS DETERMINED WITH THE SBC FRAME             #
        #################################################################

        angle_difference = Quaternion_functions.rad2deg(np.arccos(np.clip(np.dot(self.r_sat_sbc, SET_PARAMS.Earth_sensor_position),-1,1)))
        if angle_difference < SET_PARAMS.Earth_sensor_angle:
            self.r_sat_sbc = self.Earth_sensor_fault.normal_noise(self.r_sat_sbc, SET_PARAMS.Earth_noise)
            self.r_sat_sbc = self.Earth_sensor_fault.General_sensor_high_noise(self.r_sat_sbc)
            self.r_sat_sbc = self.Common_data_transmission_fault.Bit_flip(self.r_sat_sbc)
            self.r_sat_sbc = self.Common_data_transmission_fault.Sign_flip(self.r_sat_sbc)
            self.r_sat_sbc = self.Common_data_transmission_fault.Insertion_of_zero_bit(self.r_sat_sbc) 
            norm_r = np.linalg.norm(self.r_sat_sbc)
            norm_r_ORC = np.linalg.norm(self.r_sat)
            if norm_r != 0:
                self.r_sat_sbc = self.r_sat_sbc/norm_r
                self.r_sat = self.r_sat/norm_r_ORC
        else:
            self.r_sat_sbc = np.zeros(self.r_sat_sbc.shape)
            self.r_sat = np.zeros(self.r_sat.shape)

    def determine_sun_vision(self):
        #################################################################
        #    FOR THIS SPECIFIC SATELLITE MODEL, THE FINE SUN SENSOR     #
        #       IS FIXED TO THE +X FACE AND THE COARSE SUN SENSOR       #
        #                   IS FIXED TO THE -X FACE.                    #
        # THIS IS ACCORDING TO THE ORBIT AS DEFINED BY JANSE VAN VUUREN #
        #             THIS IS DETERMINED WITH THE SBC FRAME             #
        #################################################################

        if self.sun_in_view:
            angle_difference_fine = Quaternion_functions.rad2deg(np.arccos(np.dot(self.S_b[:,0], SET_PARAMS.Fine_sun_sensor_position)))
            angle_difference_coarse = Quaternion_functions.rad2deg(np.arccos(np.dot(self.S_b[:,0], SET_PARAMS.Coarse_sun_sensor_position)))
            if angle_difference_fine < SET_PARAMS.Fine_sun_sensor_angle:
                self.S_b = self.Sun_sensor_fault.normal_noise(self.S_b, SET_PARAMS.Fine_sun_noise)

                norm_S_b = np.linalg.norm(self.S_b)
                S_ORC_norm = np.linalg.norm(self.S_ORC)
                if norm_S_b != 0:
                    self.S_b = self.S_b/norm_S_b
                    self.S_ORC = self.S_ORC/S_ORC_norm

                ######################################################
                # IMPLEMENT ERROR OR FAILURE OF SENSOR IF APPLICABLE #
                ######################################################

                self.S_b = self.Sun_sensor_fault.Catastrophic_sun(self.S_b, "Fine")
                self.S_b = self.Sun_sensor_fault.Erroneous_sun(self.S_b, "Fine")
                self.S_b = self.Common_data_transmission_fault.Bit_flip(self.S_b)
                self.S_b = self.Common_data_transmission_fault.Sign_flip(self.S_b)
                self.S_b = self.Common_data_transmission_fault.Insertion_of_zero_bit(self.S_b)  

                self.sun_noise = SET_PARAMS.Fine_sun_noise

            elif angle_difference_coarse < SET_PARAMS.Coarse_sun_sensor_angle:
                self.S_b = self.Sun_sensor_fault.normal_noise(self.S_b, SET_PARAMS.Coarse_sun_noise)

                norm_S_b =np.linalg.norm(self.S_b)
                S_ORC_norm = np.linalg.norm(self.S_ORC)
                if norm_S_b != 0:
                    self.S_b = self.S_b/norm_S_b
                    self.S_ORC = self.S_ORC/S_ORC_norm

                ######################################################
                # IMPLEMENT ERROR OR FAILURE OF SENSOR IF APPLICABLE #
                ######################################################

                self.S_b = self.Sun_sensor_fault.Catastrophic_sun(self.S_b, "Coarse")
                self.S_b = self.Sun_sensor_fault.Erroneous_sun(self.S_b, "Coarse")
                self.S_b = self.Common_data_transmission_fault.Bit_flip(self.S_b)
                self.S_b = self.Common_data_transmission_fault.Sign_flip(self.S_b)
                self.S_b = self.Common_data_transmission_fault.Insertion_of_zero_bit(self.S_b)  

                self.sun_noise = SET_PARAMS.Coarse_sun_noise
            else:
                self.S_b = np.zeros(self.S_b.shape)
                self.S_ORC = np.zeros(self.S_ORC.shape)
        

    def initiate_fault_parameters(self):
        #################################
        # ALL THE CURRENT FAULT CLASSES #
        #################################
        
        self.Reaction_wheel_fault = Parameters.Reaction_wheels(self.seed)
        self.Earth_sensor_fault = Parameters.Earth_Sensor(self.seed)    
        self.Sun_sensor_fault = Parameters.Sun_sensor(self.seed)
        self.Magnetometer_fault = Parameters.Magnetometers(self.seed)
        self.Magnetorquers_fault = Parameters.Magnetorquers(self.seed)
        self.Control_fault = Parameters.Overall_control(self.seed)
        self.Common_data_transmission_fault = Parameters.Common_data_transmission(self.seed)
        self.Star_tracker_fault = Parameters.Star_tracker(self.seed)
        self.Angular_sensor_fault = Parameters.Angular_Sensor(self.seed)
    
    def initiate_purposed_fault(self, fault):
        self.fault = fault
        self.Reaction_wheel_fault.failure = self.fault
        self.Earth_sensor_fault.failure = self.fault
        self.Magnetometer_fault.failure = self.fault
        self.Sun_sensor_fault.failure = self.fault
        self.Magnetorquers_fault.failure = self.fault
        self.Control_fault.failure = self.fault
        self.Common_data_transmission_fault.failure = self.fault
        self.Star_tracker_fault.failure = self.fault
        self.Angular_sensor_fault.failure = self.fault

    ########################################################################################
    # FUNCTION TO CALCULATE THE SATELLITE ANGULAR VELOCITY BASED ON THE DERIVATIVE THEREOF #
    ########################################################################################
    def rungeKutta_w(self, x0, w, x, h):      
        ######################################################
        # CONTROL TORQUES IMPLEMENTED DUE TO THE CONTROL LAW #
        ######################################################

        N_control_magnetic, N_control_wheel = self.control.control(self.w_bi_est, self.q_est, self.Inertia, self.B, self.angular_momentum)

        N_gyro = w * (self.Inertia @ w + self.angular_momentum)

        if "RW" in self.fault:
            N_control_wheel = self.Reaction_wheel_fault.Electronics_of_RW_failure(N_control_wheel)
            N_control_wheel = self.Reaction_wheel_fault.Overheated_RW(N_control_wheel)
            N_control_wheel = self.Reaction_wheel_fault.Catastrophic_RW(N_control_wheel)
            N_control_wheel = self.Control_fault.Increasing_angular_RW_momentum(N_control_wheel)
            N_control_wheel = self.Control_fault.Decreasing_angular_RW_momentum(N_control_wheel)
            N_control_wheel = self.Control_fault.Oscillating_angular_RW_momentum(N_control_wheel)

        N_control_wheel = np.clip(N_control_wheel, -SET_PARAMS.N_ws_max, SET_PARAMS.N_ws_max)
        N_aero = 0 # ! self.dist.Aerodynamic(self.A_ORC_to_SBC, self.A_EIC_to_ORC, self.sun_in_view)

        ###################################
        # DISTURBANCE OF GRAVITY GRADIENT #
        ###################################

        Ngg = self.dist.Gravity_gradient_func(self.A_ORC_to_SBC) 

        n = int(np.round((x - x0)/h))
        y = w

        if np.isnan(N_gyro).any() or np.isinf(N_gyro).any():
            print("Break")
        #############################################
        # DISTURBANCE OF A REACTION WHEEL IMBALANCE #
        #############################################

        N_rw = np.reshape(self.dist.Wheel_Imbalance(self.angular_momentum/self.Iw, x - x0),(3,1))   

        ######################################################
        # ALL THE DISTURBANCE TORQUES ADDED TO THE SATELLITE #
        ######################################################

        N_disturbance = Ngg + N_aero + N_rw - N_gyro                
        N_control = N_control_magnetic - N_control_wheel
        N = N_control + N_disturbance

        for _ in range(n):
            k1 = h*((np.linalg.inv(self.Inertia) @ N)) 
            k2 = h*((np.linalg.inv(self.Inertia) @ N) + 0.5*k1) 
            k3 = h*((np.linalg.inv(self.Inertia) @ N) + 0.5*k2) 
            k4 = h*((np.linalg.inv(self.Inertia) @ N) + k3) 
            y = y + (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            
            x0 = x0 + h; 
        
        self.Ngyro = N_gyro
        self.Nm = N_control_magnetic
        self.Nw = N_control_wheel
        self.Ngg = Ngg

        if np.isnan(self.Ngyro).any():
            print("Break")

        self.angular_momentum = rungeKutta_h(x0, self.angular_momentum, x, h, N_control_wheel)

        self.angular_momentum = self.Angular_sensor_fault.normal_noise(self.angular_momentum, SET_PARAMS.Angular_sensor_noise)

        self.angular_momentum = np.clip(self.angular_momentum, -SET_PARAMS.h_ws_max, SET_PARAMS.h_ws_max)

        y = np.clip(y, -SET_PARAMS.wheel_angular_d_max, SET_PARAMS.wheel_angular_d_max)

        return y

    ###########################################################################################
    # FUNCTION TO CALCULATE THE SATELLITE QUATERNION POSITION BASED ON THE DERIVATIVE THEREOF #
    ###########################################################################################
    def rungeKutta_q(self, x0, y0, x, h):      
        wx, wy, wz = self.w_bo[:,0]
        n = int(np.round((x - x0)/h))

        y = y0

        W = np.array(([0, wz, -wy, wx], [-wz, 0, wx, wy], [wy, -wx, 0, wz], [-wx, -wy, -wz, 0]))
        for _ in range(n):
            k1 = h*(0.5 * W @ y)
            k2 = h*(0.5 * W @ (y + 0.5*k1))
            k3 = h*(0.5 * W @ (y + 0.5*k2))
            k4 = h*(0.5 * W @ (y + k3))

            y = y + (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
            x0 = x0 + h; 
        
        norm_y = np.linalg.norm(y)
        y = y/norm_y
        
        if np.isnan(y).any() or (y == 0).all():
            print("Break")

        return y

    def Fault_implementation(self):
        if self.fault == "None":
            Faults = []
            True_faults = []
            Faults.append(self.Reaction_wheel_fault.Failure_Reliability_area(self.t)) 
            Faults.append(self.Earth_sensor_fault.Failure_Reliability_area(self.t))   
            Faults.append(self.Sun_sensor_fault.Failure_Reliability_area(self.t))
            Faults.append(self.Magnetometer_fault.Failure_Reliability_area(self.t))
            Faults.append(self.Magnetorquers_fault.Failure_Reliability_area(self.t))
            Faults.append(self.Control_fault.Failure_Reliability_area(self.t))
            Faults.append(self.Common_data_transmission_fault.Failure_Reliability_area(self.t))
            Faults.append(self.Star_tracker_fault.Failure_Reliability_area(self.t))
            for fault in Faults:
                if fault != "None":
                    True_faults.append(fault)
                
            if True_faults:
                self.fault = True_faults[self.np_random.randint(0,len(True_faults))]
                print(self.fault)
                

    def rotation(self):
        ##############################################################
        #    DETERMINE WHETHER A FAUKLT OCCURED WITHIN THE SYSTEM    #
        # BASED ON THE STATISTICAL PROBABILITY DEFINED IN PARAMETERS #
        ##############################################################

        self.Fault_implementation()

        self.A_ORC_to_SBC = Transformation_matrix(self.q)
        self.r_sat, self.v_sat_EIC, self.A_EIC_to_ORC, r_EIC = self.sense.satellite_vector(self.t)
        self.r_EIC = np.asarray(r_EIC).astype(np.float64)

        self.S_EIC, self.sun_in_view = self.sense.sun(self.t)
        self.S_ORC = self.A_EIC_to_ORC @ self.S_EIC

        ######################################
        # DETERMINE THE DCM OF THE SATELLITE #
        ######################################
        
        self.r_sat_sbc = self.A_ORC_to_SBC @ self.r_sat

        if (self.S_EIC == np.nan).any():
            print("break")

        self.S_b = self.A_ORC_to_SBC  @ self.S_ORC
        
        norm_S_b = np.linalg.norm(self.S_b)

        if self.sun_in_view and norm_S_b != 0:
            self.S_b = self.S_b/norm_S_b
 

        ##################################################
        # DETERMINE WHETHER THE SUN AND THE EARTH SENSOR #
        #   IS IN VIEW OF THE VECTOR ON THE SATELLITE    #
        ##################################################

        self.determine_sun_vision()
        self.determine_earth_vision()
    
        self.Beta = self.sense.magnetometer(self.t) 

        self.B = self.A_EIC_to_ORC @ self.Beta 
        self.B = self.A_ORC_to_SBC @ self.B

        ######################################################
        # IMPLEMENT ERROR OR FAILURE OF SENSOR IF APPLICABLE #
        ######################################################
        self.B = self.Magnetometer_fault.normal_noise(self.B, SET_PARAMS.Magnetometer_noise)

        norm_B = np.linalg.norm(self.B)

        if norm_B != 0:
            self.B = self.B/norm_B

        self.B = self.Magnetometer_fault.Stop_magnetometers (self.B)
        self.B = self.Magnetometer_fault.Interference_magnetic(self.B)
        self.B = self.Magnetometer_fault.General_sensor_high_noise(self.B)
        self.B = self.Common_data_transmission_fault.Bit_flip(self.B)
        self.B = self.Common_data_transmission_fault.Sign_flip(self.B)
        self.B = self.Common_data_transmission_fault.Insertion_of_zero_bit(self.B)

        # Model star tracker vector as measured
        self.star_tracker_vector_measured = self.Star_tracker_fault.normal_noise(self.A_ORC_to_SBC @ self.star_tracker_vector,SET_PARAMS.star_tracker_noise)
        star_norm = np.linalg.norm(self.star_tracker_vector_measured)
        if star_norm != 0:
            self.star_tracker_vector_measured = self.star_tracker_vector_measured/np.linalg.norm(self.star_tracker_vector_measured)
        else:
            self.star_tracker_vector_measured = np.zeros(self.star_tracker_vector_measured.shape)
        
        self.star_tracker_vector_measured = self.Star_tracker_fault.Closed_shutter(self.star_tracker_vector_measured)

        self.sensor_vectors = {
        "Sun_Sensor": {"measured": self.S_b, "modelled": self.S_ORC, "noise": self.sun_noise}, 
        "Earth_Sensor": {"measured": self.r_sat_sbc, "modelled": self.r_sat, "noise": SET_PARAMS.Earth_noise}, 
        "Star_tracker": {"measured": self.star_tracker_vector_measured, "modelled": self.star_tracker_vector, "noise": SET_PARAMS.star_tracker_noise}
        }

        ########################################################
        # THE ERROR FOR W_BI IS WITHIN THE RUNGEKUTTA FUNCTION #
        ######################################################## 
        self.w_bi = self.rungeKutta_w(self.t, self.w_bi, self.t+self.dt, self.dh)

        if np.isnan(self.w_bi).any():
            print("break")
        
        self.w_bo = self.w_bi - self.A_ORC_to_SBC @ np.array(([0],[-self.wo],[0]))

        self.q = self.rungeKutta_q(self.t, self.q, self.t+self.dt, self.dh)

        if np.isnan(self.q).any() or (self.q == 0).all():
            print("Break")

        ########################################
        # DETERMINE THE ACTUAL POSITION OF THE #
        # SATELLITE FROM THE EARTH AND THE SUN #
        ########################################


        if SET_PARAMS.Kalman_filter_use == "EKF":
            
            for sensor in self.sensors_kalman:
                # Step through both the sensor noise and the sensor measurement
                # vector is the vector of the sensor's measurement
                # This is used to compare it to the modelled measurement
                # Consequently, the vector is the ORC modelled vector before
                # the transformation Matrix is implemented on the vector
                # Since the transformation matrix takes the modelled and measured into account
                # Only noise is added to the measurement

                v = self.sensor_vectors[sensor]
                v_ORC_k = np.reshape(v["modelled"],(3,1))
                v_norm = np.linalg.norm(v_ORC_k)
                if (v_norm != 0).any():
                    v_ORC_k = v_ORC_k/np.linalg.norm(v_ORC_k)
                v_measured_k = np.reshape(v["measured"],(3,1))

                if not (v_ORC_k == 0.0).all():
                    # If the measured vector is equal to 0 then the sensor is not able to view the desired measurement
                    x = self.EKF.Kalman_update(v_measured_k, v_ORC_k, self.Nm, self.Nw, self.Ngyro, self.Ngg, self.t)
                    self.q_est = x[3:]
                    self.w_bi_est = x[:3]


        elif SET_PARAMS.Kalman_filter_use == "RKF":
            for sensor in self.sensors_kalman:
                # Step through both the sensor noise and the sensor measurement
                # vector is the vector of the sensor's measurement
                # This is used to compare it to the modelled measurement
                # Consequently, the vector is the ORC modelled vector before
                # the transformation Matrix is implemented on the vector
                # Since the transformation matrix takes the modelled and measured into account
                # Only noise is added to the measurement

                v = self.sensor_vectors[sensor]
                v_ORC_k = np.reshape(v["modelled"],(3,1))
                v_norm = np.linalg.norm(v_ORC_k)
                if (v_norm != 0).any():
                    v_ORC_k = v_ORC_k/np.linalg.norm(v_ORC_k)
                v_measured_k = np.reshape(v["measured"],(3,1))
                self.EKF.measurement_noise = v["noise"]

                if not (v_ORC_k == 0.0).all():
                    # If the measured vektor is equal to 0 then the sensor is not able to view the desired measurement
                    x = self.RKF.Kalman_update(v_measured_k, self.Nm, self.Nw, self.Ngyro, self.t)
                    self.w_bi_est = np.clip(x, -SET_PARAMS.wheel_angular_d_max, SET_PARAMS.wheel_angular_d_max)
                    self.q_est = self.q
        else:
            self.w_bi_est = self.w_bi
            self.q_est = self.q

        if np.isnan(self.w_bi).any():
            print("Break")

        self.t += self.dt

        self.update()

        return self.w_bi, self.q, self.A_ORC_to_SBC, self.r_EIC, self.sun_in_view


class Single_Satellite(Dynamics):
    def __init__(self, seed, s_list, t_list, J_t, fr):
        self.seed = seed
        self.np_random = np.random
        self.np_random.seed(seed)                   # Ensures that every fault parameters are implemented with different random seeds
        self.sense = Sensors(s_list, t_list, J_t, fr)
        self.dist = Disturbances(self.sense)                  # Disturbances of the simulation
        self.w_bi = SET_PARAMS.wbi                  # Angular velocity in ORC
        self.w_bi_est = self.w_bi
        self.wo = SET_PARAMS.wo                     # Angular velocity of satellite around the earth
        self.angular_wheels = SET_PARAMS.initial_angular_wheels 
        self.q = SET_PARAMS.quaternion_initial      # Quaternion position
        self.q_est = self.q
        self.t = SET_PARAMS.time                    # Beginning time
        self.dt = SET_PARAMS.Ts                     # Time step
        self.dh = self.dt/10                        # Size of increments for Runga-kutta method
        self.Ix = SET_PARAMS.Ix                     # Ixx inertia
        self.Iy = SET_PARAMS.Iy                     # Iyy inertia
        self.Iz = SET_PARAMS.Iz                     # Izz inertia
        self.Inertia = np.identity(3)*np.array(([self.Ix, self.Iy, self.Iz]))
        self.Iw = SET_PARAMS.Iw                     # Inertia of a reaction wheel
        self.angular_momentum = SET_PARAMS.initial_angular_wheels # Angular momentum of satellite wheels
        self.faster_than_control = SET_PARAMS.faster_than_control   # If it is required that satellite must move faster around the earth than Ts
        self.control = Controller.Control()         # Controller.py is used for control of satellite    
        self.star_tracker_vector = SET_PARAMS.star_tracker_vector
        self.sun_noise = SET_PARAMS.Fine_sun_noise
        self.RKF = RKF()                            # Rate Kalman_filter
        self.EKF = EKF()                            # Extended Kalman_filter
        self.sensors_kalman = ["Earth_Sensor", "Sun_Sensor", "Star_tracker"] #"Earth_Sensor", "Sun_Sensor", "Star_tracker"
        super().initiate_fault_parameters()

        ####################################################
        #  THE ORBIT_DATA DICTIONARY IS USED TO STORE ALL  #
        #     THE MEASUREMENTS FOR EACH TIMESTEP (TS)      #
        # EACH ORBIT HAS AN INDUCED FAULT WITHIN THE ADCS. #
        ####################################################

        self.Orbit_Data = {
            "Sun": [],            #S_o measurement (vector of sun in ORC)
            "Magnetometer": [],    #B vector in SBC
            "Earth": [],           #Satellite position vector in ORC
            "Angular momentum of wheels": [],    #Wheel angular velocity of each reaction wheel
            "Star": [],
            "Angular velocity of satellite": [],
            "Sun in view": [],                              #True or False values depending on whether the sun is in view of the satellite
            "Current fault": [],                            #What the fault is that the system is currently experiencing
            "Current fault numeric": [],
            "Current fault binary": []
        }

        self.zeros = np.zeros((SET_PARAMS.number_of_faults,), dtype = int)

        self.fault = "None"                      # Current fault of the system

    def update(self):
        self.Orbit_Data["Magnetometer"] = self.B
        self.Orbit_Data["Sun"] = self.S_b[:,0]
        self.Orbit_Data["Earth"] = self.r_sat_sbc
        self.Orbit_Data["Star"] = self.star_tracker_vector_measured
        self.Orbit_Data["Angular momentum of wheels"] = self.angular_momentum[:,0]
        self.Orbit_Data["Angular velocity of satellite"] = self.w_bi[:,0]
        self.Orbit_Data["Sun in view"] = self.sun_in_view
        if self.sun_in_view == False and (self.fault == "Catastrophic_sun" or self.fault == "Erroneous"):
            self.Orbit_Data["Current fault"] = "None"
            temp = list(self.zeros)
            temp[Fault_names_to_num["None"] - 1] = 1
            self.Orbit_Data["Current fault numeric"] = temp
            self.Orbit_Data["Current fault binary"] = 0
        else:
            self.Orbit_Data["Current fault"] = self.fault
            temp = list(self.zeros)
            temp[Fault_names_to_num[self.fault] - 1] = 1
            self.Orbit_Data["Current fault numeric"] = temp
            self.Orbit_Data["Current fault binary"] = 0 if self.fault == "None" else 1
"""
    # Initiate initial parameters for the beginning of each orbit set (fault)
    def __init__(self, seed, s_list, t_list, J_t, fr):
        self.seed = seed
        self.np_random = np.random
        self.np_random.seed(seed)                   # Ensures that every fault parameters are implemented with different random seeds
        self.sense = Sensors(s_list, t_list, J_t, fr)
        self.dist = Disturbances(self.sense)                  # Disturbances of the simulation
        self.w_bi = SET_PARAMS.wbi                  # Angular velocity in ORC
        self.w_bi_est = self.w_bi
        self.wo = SET_PARAMS.wo                     # Angular velocity of satellite around the earth
        self.angular_wheels = SET_PARAMS.initial_angular_wheels 
        self.q = SET_PARAMS.quaternion_initial      # Quaternion position
        self.q_est = self.q
        self.t = SET_PARAMS.time                    # Beginning time
        self.dt = SET_PARAMS.Ts                     # Time step
        self.dh = self.dt/10                        # Size of increments for Runga-kutta method
        self.Ix = SET_PARAMS.Ix                     # Ixx inertia
        self.Iy = SET_PARAMS.Iy                     # Iyy inertia
        self.Iz = SET_PARAMS.Iz                     # Izz inertia
        self.Inertia = np.identity(3)*np.array(([self.Ix, self.Iy, self.Iz]))
        self.Iw = SET_PARAMS.Iw                     # Inertia of a reaction wheel
        self.angular_momentum = SET_PARAMS.initial_angular_wheels # Angular momentum of satellite wheels
        self.faster_than_control = SET_PARAMS.faster_than_control   # If it is required that satellite must move faster around the earth than Ts
        self.control = Controller.Control()         # Controller.py is used for control of satellite    
        self.star_tracker_vector = SET_PARAMS.star_tracker_vector
        self.sun_noise = SET_PARAMS.Fine_sun_noise
        self.RKF = RKF()                            # Rate Kalman_filter
        self.EKF = EKF()                            # Extended Kalman_filter
        self.sensors_kalman = ["Earth_Sensor", "Sun_Sensor", "Star_tracker"] #"Earth_Sensor", "Sun_Sensor", "Star_tracker"
        super().initiate_fault_parameters()

        ####################################################
        #  THE ORBIT_DATA DICTIONARY IS USED TO STORE ALL  #
        #     THE MEASUREMENTS FOR EACH TIMESTEP (TS)      #
        # EACH ORBIT HAS AN INDUCED FAULT WITHIN THE ADCS. #
        ####################################################

        self.Orbit_Data = {
            "Sun": [],            #S_o measurement (vector of sun in ORC)
            "Magnetometer": [],    #B vector in SBC
            "Earth": [],           #Satellite position vector in ORC
            "Angular momentum of wheels": [],    #Wheel angular velocity of each reaction wheel
            "Star": [],
            "Angular velocity of satellite": [],
            "Sun in view": [],                              #True or False values depending on whether the sun is in view of the satellite
            "Current fault": [],                            #What the fault is that the system is currently experiencing
            "Current fault numeric": [],
            "Current fault binary": []
        }

        self.zeros = np.zeros((SET_PARAMS.number_of_faults,), dtype = int)

        self.fault = "None"                      # Current fault of the system

    def update(self):
        self.Orbit_Data["Magnetometer"].append(self.B)
        self.Orbit_Data["Sun"].append(self.S_b[:,0])
        self.Orbit_Data["Earth"].append(self.r_sat_sbc)
        self.Orbit_Data["Star"].append(self.star_tracker_vector_measured)
        self.Orbit_Data["Angular momentum of wheels"].append(self.angular_momentum[:,0])
        self.Orbit_Data["Angular velocity of satellite"].append(self.w_bi[:,0])
        self.Orbit_Data["Sun in view"].append(self.sun_in_view)
        if self.sun_in_view == False and (self.fault == "Catastrophic_sun" or self.fault == "Erroneous"):
            self.Orbit_Data["Current fault"].append("None")
            temp = list(self.zeros)
            temp[Fault_names_to_num["None"] - 1] = 1
            self.Orbit_Data["Current fault numeric"].append(temp)
            self.Orbit_Data["Current fault binary"].append(0)
        else:
            self.Orbit_Data["Current fault"].append(self.fault)
            temp = list(self.zeros)
            temp[Fault_names_to_num[self.fault] - 1] = 1
            self.Orbit_Data["Current fault numeric"].append(temp)
            self.Orbit_Data["Current fault binary"].append(0 if self.fault == "None" else 1)
"""
class Constellation_Satellites(Dynamics):
    # Initiate initial parameters for the beginning of each orbit set (fault)
    def __init__(self, seed, s_list, t_list, J_t, fr):
        self.seed = seed
        self.np_random = np.random
        self.np_random.seed(seed)                   # Ensures that every fault parameters are implemented with different random seeds
        self.sense = Sensors(s_list, t_list, J_t, fr)
        self.dist = Disturbances(self.sense)                  # Disturbances of the simulation
        self.w_bi = SET_PARAMS.wbi                  # Angular velocity in ORC
        self.w_bi_est = self.w_bi
        self.wo = SET_PARAMS.wo                     # Angular velocity of satellite around the earth
        self.angular_wheels = SET_PARAMS.initial_angular_wheels 
        self.q = SET_PARAMS.quaternion_initial      # Quaternion position
        self.q_est = self.q
        self.t = SET_PARAMS.time                    # Beginning time
        self.dt = SET_PARAMS.Ts                     # Time step
        self.dh = self.dt/10                        # Size of increments for Runga-kutta method
        self.Ix = SET_PARAMS.Ix                     # Ixx inertia
        self.Iy = SET_PARAMS.Iy                     # Iyy inertia
        self.Iz = SET_PARAMS.Iz                     # Izz inertia
        self.Inertia = np.identity(3)*np.array(([self.Ix, self.Iy, self.Iz]))
        self.Iw = SET_PARAMS.Iw                     # Inertia of a reaction wheel
        self.angular_momentum = SET_PARAMS.initial_angular_wheels # Angular momentum of satellite wheels
        self.faster_than_control = SET_PARAMS.faster_than_control   # If it is required that satellite must move faster around the earth than Ts
        self.control = Controller.Control()         # Controller.py is used for control of satellite    
        self.star_tracker_vector = SET_PARAMS.star_tracker_vector
        self.sun_noise = SET_PARAMS.Fine_sun_noise
        self.RKF = RKF()                            # Rate Kalman_filter
        self.EKF = EKF()                            # Extended Kalman_filter
        self.sensors_kalman = ["Earth_Sensor", "Sun_Sensor", "Star_tracker"] #"Earth_Sensor", "Sun_Sensor", "Star_tracker"
        super().initiate_fault_parameters()

        ####################################################
        #  THE ORBIT_DATA DICTIONARY IS USED TO STORE ALL  #
        #     THE MEASUREMENTS FOR EACH TIMESTEP (TS)      #
        # EACH ORBIT HAS AN INDUCED FAULT WITHIN THE ADCS. #
        ####################################################

        self.Orbit_Data = {
            "Sun": [],            #S_o measurement (vector of sun in ORC)
            "Magnetometer": [],    #B vector in SBC
            "Earth": [],           #Satellite position vector in ORC
            "Angular momentum of wheels": [],    #Wheel angular velocity of each reaction wheel
            "Star": [],
            "Angular velocity of satellite": [],
            "Sun in view": [],                              #True or False values depending on whether the sun is in view of the satellite
            "Current fault": [],                            #What the fault is that the system is currently experiencing
            "Current fault numeric": [],
            "Current fault binary": []
        }

        self.zeros = np.zeros((SET_PARAMS.number_of_faults,), dtype = int)

        self.fault = "None"                      # Current fault of the system

    def update(self):
        self.Orbit_Data["Magnetometer"] = self.B
        self.Orbit_Data["Sun"] = self.S_b[:,0]
        self.Orbit_Data["Earth"] = self.r_sat_sbc
        self.Orbit_Data["Star"] = self.star_tracker_vector_measured
        self.Orbit_Data["Angular momentum of wheels"] = self.angular_momentum[:,0]
        self.Orbit_Data["Angular velocity of satellite"] = self.w_bi[:,0]
        self.Orbit_Data["Sun in view"] = self.sun_in_view
        if self.sun_in_view == False and (self.fault == "Catastrophic_sun" or self.fault == "Erroneous"):
            self.Orbit_Data["Current fault"] = "None"
            temp = list(self.zeros)
            temp[Fault_names_to_num["None"] - 1] = 1
            self.Orbit_Data["Current fault numeric"] = temp
            self.Orbit_Data["Current fault binary"] = 0
        else:
            self.Orbit_Data["Current fault"] = self.fault
            temp = list(self.zeros)
            temp[Fault_names_to_num[self.fault] - 1] = 1
            self.Orbit_Data["Current fault numeric"] = temp
            self.Orbit_Data["Current fault binary"] = 0 if self.fault == "None" else 1