import numpy as np
import Simulation.Controller as Controller
from Simulation.Disturbances import Disturbances
import Simulation.Parameters as Parameters
SET_PARAMS = Parameters.SET_PARAMS 
from Simulation.Sensors import Sensors
import Simulation.Quaternion_functions as Quaternion_functions
from Simulation.Kalman_filter import RKF
from Simulation.EKF import EKF
from Simulation.SensorPredictions import SensorPredictionsDMD
import collections
import math
from Simulation.utilities import Reflection, Intersection, PointWithinParallelLines, lineEquation, line2Equation
import Fault_prediction.Fault_detection as FaultDetection

pi = math.pi

Fault_names_to_num = SET_PARAMS.Fault_names

# The DCM must be calculated depending on the current quaternions
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
    
    y = np.clip(y, -SET_PARAMS.h_ws_max, SET_PARAMS.h_ws_max)

    return y


class Dynamics:

    def determine_magnetometer(self):
        #* Normalize self.B_ORC
        norm_B_ORC = np.linalg.norm(self.B_ORC)

        if norm_B_ORC != 0:
            self.B_ORC = self.B_ORC/norm_B_ORC

        self.B_sbc = self.A_ORC_to_SBC @ self.B_ORC
        ######################################################
        # IMPLEMENT ERROR OR FAILURE OF SENSOR IF APPLICABLE #
        ######################################################
        self.B_sbc = self.Magnetometer_fault.normal_noise(self.B_sbc, SET_PARAMS.Magnetometer_noise)
        self.B_sbc = self.Magnetometer_fault.Stop_magnetometers (self.B_sbc)
        self.B_sbc = self.Magnetometer_fault.Interference_magnetic(self.B_sbc)
        self.B_sbc = self.Magnetometer_fault.General_sensor_high_noise(self.B_sbc)

        self.B_sbc = self.Common_data_transmission_fault.Bit_flip(self.B_sbc)
        self.B_sbc = self.Common_data_transmission_fault.Sign_flip(self.B_sbc)
        self.B_sbc = self.Common_data_transmission_fault.Insertion_of_zero_bit(self.B_sbc)

    def determine_star_tracker(self):
        self.star_tracker_ORC = self.Star_tracker_fault.normal_noise(self.star_tracker_vector,SET_PARAMS.star_tracker_noise)
        self.star_tracker_ORC = self.Star_tracker_fault.Closed_shutter(self.star_tracker_ORC)

        #* Star tracker
        self.star_tracker_sbc = self.A_ORC_to_SBC @ self.star_tracker_ORC

    def determine_earth_vision(self):
        #################################################################
        #      FOR THIS SPECIFIC SATELLITE MODEL, THE EARTH SENSOR      #
        #                    IS FIXED TO THE -Z FACE                    #
        # THIS IS ACCORDING TO THE ORBIT AS DEFINED BY JANSE VAN VUUREN #
        #             THIS IS DETERMINED WITH THE SBC FRAME             #
        #################################################################
        #* self.r_sat_ORC is already normalized
        self.r_sat_sbc = self.A_ORC_to_SBC @ self.r_sat_ORC

        angle_difference = Quaternion_functions.rad2deg(np.arccos(np.clip(np.dot(self.r_sat_sbc, SET_PARAMS.Earth_sensor_position),-1,1)))
        if angle_difference < SET_PARAMS.Earth_sensor_angle:
            self.r_sat_sbc = self.Earth_sensor_fault.normal_noise(self.r_sat_sbc, SET_PARAMS.Earth_noise)
            self.r_sat_sbc = self.Earth_sensor_fault.General_sensor_high_noise(self.r_sat_sbc)
            self.r_sat_sbc = self.Common_data_transmission_fault.Bit_flip(self.r_sat_sbc)
            self.r_sat_sbc = self.Common_data_transmission_fault.Sign_flip(self.r_sat_sbc)
            self.r_sat_sbc = self.Common_data_transmission_fault.Insertion_of_zero_bit(self.r_sat_sbc) 

        else:
            self.r_sat_sbc = np.zeros(self.r_sat_sbc.shape)
            self.r_sat_ORC = np.zeros(self.r_sat_ORC.shape)

    def determine_sun_vision(self):
        #################################################################
        #    FOR THIS SPECIFIC SATELLITE MODEL, THE FINE SUN SENSOR     #
        #       IS FIXED TO THE +X FACE AND THE COARSE SUN SENSOR       #
        #                   IS FIXED TO THE -X FACE.                    #
        # THIS IS ACCORDING TO THE ORBIT AS DEFINED BY JANSE VAN VUUREN #
        #             THIS IS DETERMINED WITH THE SBC FRAME             #
        #################################################################
        #* Normalize self.S_ORC
        norm_S_ORC = np.linalg.norm(self.S_ORC)

        if norm_S_ORC != 0:
             self.S_ORC = self.S_ORC/norm_S_ORC

        self.S_sbc = self.A_ORC_to_SBC @ self.S_ORC

        if self.sun_in_view:
            
            angle_difference_fine = Quaternion_functions.rad2deg(np.arccos(np.dot(self.S_sbc, SET_PARAMS.Fine_sun_sensor_position)))
            angle_difference_coarse = Quaternion_functions.rad2deg(np.arccos(np.dot(self.S_sbc, SET_PARAMS.Coarse_sun_sensor_position)))

            if angle_difference_fine < SET_PARAMS.Fine_sun_sensor_angle: 
                if SET_PARAMS.Reflection: 
                    reflectedSunVector = Reflection(self.S_sbc, SET_PARAMS.SPF_normal_vector)

                    IntersectionPointLeft = Intersection(SET_PARAMS.SSF_Plane, reflectedSunVector, SET_PARAMS.SPF_LeftTopCorner)

                    IntersectionPointRight = Intersection(SET_PARAMS.SSF_Plane, reflectedSunVector, SET_PARAMS.SPF_RightTopCorner)

                    Line1 = lineEquation(IntersectionPointLeft, SET_PARAMS.SPF_LeftBottomCorner)

                    Line2 = lineEquation(IntersectionPointRight, SET_PARAMS.SPF_RightBottomCorner)

                    Line3 = line2Equation(IntersectionPointLeft, SET_PARAMS.SPF_LeftBottomCorner)

                    Line4 = line2Equation(IntersectionPointRight, SET_PARAMS.SPF_RightBottomCorner)

                    reflection1 = PointWithinParallelLines(Line1, Line2, SET_PARAMS.SSF_LeftCorner)

                    reflection2 = PointWithinParallelLines(Line3, Line4, SET_PARAMS.SSF_LeftCorner)

                    reflection = reflection1 and reflection2

                    if not reflection:
                        reflection1 = PointWithinParallelLines(Line1, Line2, SET_PARAMS.SSF_RightCorner)

                        reflection2 = PointWithinParallelLines(Line3, Line4, SET_PARAMS.SSF_RightCorner)

                        reflection = reflection1 and reflection2

                        if reflection:
                            self.S_ORC = reflectedSunVector
                    else:
                        self.S_ORC = reflectedSunVector

                self.S_sbc = self.A_ORC_to_SBC @ self.S_ORC

                self.S_sbc = self.Sun_sensor_fault.normal_noise(self.S_sbc, SET_PARAMS.Coarse_sun_noise)

                ######################################################
                # IMPLEMENT ERROR OR FAILURE OF SENSOR IF APPLICABLE #
                ######################################################

                self.S_sbc, self.sun_in_view = self.Sun_sensor_fault.Catastrophic_sun(self.S_sbc, self.sun_in_view, "Fine")
                self.S_sbc = self.Sun_sensor_fault.Erroneous_sun(self.S_sbc, "Fine")
                self.S_sbc = self.Common_data_transmission_fault.Bit_flip(self.S_sbc)
                self.S_sbc = self.Common_data_transmission_fault.Sign_flip(self.S_sbc)
                self.S_sbc = self.Common_data_transmission_fault.Insertion_of_zero_bit(self.S_sbc)  

                self.sun_noise = SET_PARAMS.Fine_sun_noise

            elif angle_difference_coarse < SET_PARAMS.Coarse_sun_sensor_angle:
                if SET_PARAMS.Reflection: 
                    reflectedSunVector = Reflection(self.S_sbc, SET_PARAMS.SPC_normal_vector)

                    IntersectionPointLeft = Intersection(SET_PARAMS.SSC_Plane, reflectedSunVector, SET_PARAMS.SPC_LeftTopCorner)

                    IntersectionPointRight = Intersection(SET_PARAMS.SSC_Plane, reflectedSunVector, SET_PARAMS.SPC_RightTopCorner)

                    Line1 = lineEquation(IntersectionPointLeft, SET_PARAMS.SPC_LeftBottomCorner)

                    Line2 = lineEquation(IntersectionPointRight, SET_PARAMS.SPC_RightBottomCorner)

                    Line3 = line2Equation(IntersectionPointLeft, SET_PARAMS.SPC_LeftBottomCorner)

                    Line4 = line2Equation(IntersectionPointRight, SET_PARAMS.SPC_RightBottomCorner)

                    reflection1 = PointWithinParallelLines(Line1, Line2, SET_PARAMS.SSC_LeftCorner)

                    reflection2 = PointWithinParallelLines(Line3, Line4, SET_PARAMS.SSC_LeftCorner)

                    reflection = reflection1 and reflection2

                    if not reflection:
                        reflection1 = PointWithinParallelLines(Line1, Line2, SET_PARAMS.SSC_RightCorner)

                        reflection2 = PointWithinParallelLines(Line3, Line4, SET_PARAMS.SSC_RightCorner)

                        reflection = reflection1 and reflection2
                        
                        if reflection:
                            self.S_ORC = reflectedSunVector
                    else:
                        self.S_ORC = reflectedSunVector
                
                self.S_sbc = self.A_ORC_to_SBC @ self.S_ORC

                self.S_sbc = self.Sun_sensor_fault.normal_noise(self.S_sbc, SET_PARAMS.Coarse_sun_noise)

                ######################################################
                # IMPLEMENT ERROR OR FAILURE OF SENSOR IF APPLICABLE #
                ######################################################

                self.S_sbc, self.sun_in_view = self.Sun_sensor_fault.Catastrophic_sun(self.S_sbc, self.sun_in_view, "Coarse")
                self.S_sbc = self.Sun_sensor_fault.Erroneous_sun(self.S_sbc, "Coarse")
                self.S_sbc = self.Common_data_transmission_fault.Bit_flip(self.S_sbc)
                self.S_sbc = self.Common_data_transmission_fault.Sign_flip(self.S_sbc)
                self.S_sbc = self.Common_data_transmission_fault.Insertion_of_zero_bit(self.S_sbc)  

                self.sun_noise = SET_PARAMS.Coarse_sun_noise
            else:
                self.S_sbc = np.zeros(self.S_sbc.shape)
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

        N_control_magnetic, N_control_wheel = self.control.control(self.w_bi_est, self.w_bo_est, self.q_est, self.Inertia, self.B_sbc, self.angular_momentum_with_noise, self.r_sat_sbc, self.S_sbc, self.sun_in_view)

        N_gyro = np.cross(w,(self.Inertia @ w + self.angular_momentum))

        if "RW" in self.fault:
            N_control_wheel = self.Reaction_wheel_fault.Electronics_of_RW_failure(N_control_wheel)
            N_control_wheel = self.Reaction_wheel_fault.Overheated_RW(N_control_wheel)
            N_control_wheel = self.Reaction_wheel_fault.Catastrophic_RW(N_control_wheel)
            N_control_wheel = self.Control_fault.Increasing_angular_RW_momentum(N_control_wheel)
            N_control_wheel = self.Control_fault.Decreasing_angular_RW_momentum(N_control_wheel)
            N_control_wheel = self.Control_fault.Oscillating_angular_RW_momentum(N_control_wheel)

        N_aero = 0 # ! self.dist.Aerodynamic(self.A_ORC_to_SBC, self.A_EIC_to_ORC, self.sun_in_view)

        ###################################
        # DISTURBANCE OF GRAVITY GRADIENT #
        ###################################

        Ngg = self.dist.Gravity_gradient_func(self.A_ORC_to_SBC) 

        x01 = x0

        n = int(np.round((x - x0)/h))
        y = w

        for _ in range(n):
            #############################################
            # DISTURBANCE OF A REACTION WHEEL IMBALANCE #
            #############################################

            N_rw = self.dist.Wheel_Imbalance(self.angular_momentum/self.Iw, x - x0)

            ######################################################
            # ALL THE DISTURBANCE TORQUES ADDED TO THE SATELLITE #
            ######################################################

            N_disturbance = Ngg + N_aero + N_rw - N_gyro   

            N_control = N_control_magnetic - N_control_wheel
            N = N_control + N_disturbance

            k1 = h*((self.Inertia_Inverse @ N)) 
            k2 = h*((self.Inertia_Inverse @ N) + 0.5*k1) 
            k3 = h*((self.Inertia_Inverse @ N) + 0.5*k2) 
            k4 = h*((self.Inertia_Inverse @ N) + k3) 
            y = y + (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            
            x0 = x0 + h
        
        self.Ngyro = N_gyro
        self.Nm = N_control_magnetic
        self.Nw = N_control_wheel
        self.Ngg = Ngg
        self.Nrw = N_rw
        self.Naero = N_aero

        self.angular_momentum = rungeKutta_h(x01, self.angular_momentum, x, h, N_control_wheel)

        self.angular_momentum_with_noise = self.Angular_sensor_fault.normal_noise(self.angular_momentum, SET_PARAMS.Angular_sensor_noise)

        self.angular_momentum_with_noise = np.clip(self.angular_momentum_with_noise, -SET_PARAMS.h_ws_max, SET_PARAMS.h_ws_max)

        y = np.clip(y, -SET_PARAMS.wheel_angular_d_max, SET_PARAMS.wheel_angular_d_max)

        return y

    ###########################################################################################
    # FUNCTION TO CALCULATE THE SATELLITE QUATERNION POSITION BASED ON THE DERIVATIVE THEREOF #
    ###########################################################################################
    def rungeKutta_q(self, x0, y0, x, h):      
        wx, wy, wz = self.w_bo
        n = int(np.round((x - x0)/h))

        y = y0

        W = np.array([[0, wz, -wy, wx], [-wz, 0, wx, wy], [wy, -wx, 0, wz], [-wx, -wy, -wz, 0]])
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

    ##############################################################################################
    # FUNCTION TO HANDLE THE PREDICTION, ISOLATION AND RECOVERY OF SENSORS FAILURES OR ANOMALIES #
    ############################################################################################## 
    def SensorFailureHandling(self):
        if SET_PARAMS.SensorFDIR:
            Sensors_X, Sensors_Y = self.SensorFeatureExtraction()

            self.predictedFailure = self.SensorPredicting(Sensors_X)

            # # If a failure is predicted the cause of the failure must be determined
            # if predictedFailure:
            #     sensorFailed = self.SensorIsolation()

            #     # After the specific sensor that has failed is identified 
            #     # The system must recover
            #     self.SensorRecovery(sensorFailed)
    
    def SensorFeatureExtraction(self):
        if SET_PARAMS.FeatureExtraction == "DMD":
            Sensors_X = np.concatenate([self.Orbit_Data["Magnetometer"], 
                                            self.Orbit_Data["Sun"], self.Orbit_Data["Earth"], 
                                            self.Orbit_Data["Star"],  
                                            self.Orbit_Data["Angular momentum of wheels"]])
            Sensors_Y = np.concatenate([self.Orbit_Data["Wheel Control Torques"], 
                                    self.Orbit_Data["Magnetic Control Torques"]])
            

            if self.t == SET_PARAMS.time:
                # Initiating parameters for SensorPredictions
                self.SensePredDMD = SensorPredictionsDMD(Sensors_X)
                self.MovingAverage = 0
            
            self.MovingAverage = self.SensePredDMD.MovingAverage(Sensors_X, Sensors_Y)

        return Sensors_X, Sensors_Y

    #################################################
    # FUNCTION TO PREDICT IF AN ANOMALY HAS OCCURED #
    #################################################
    def SensorPredicting(self, Sensors_X):
        if SET_PARAMS.SensorPredictor == "DecisionTrees":
            Sensors_X = np.array([np.concatenate([Sensors_X, np.array([self.MovingAverage])])])
            predictedFailure = self.DecisionTreeDMD.Predict(Sensors_X)
            
        return predictedFailure

    ###############################################################
    # FUNCTION TO ISOLATE (CLASSIFY) THE ANOMALY THAT HAS OCCURED #
    ###############################################################
    def SensorIsolation(self):
        #! This should account for multiple predictions of failures
        if SET_PARAMS.SensorIsolator == "DMD":
            for sensorData in self.availableData:
                self.Sensors_X = self.Orbit_Data[sensorData]
                Y = [self.Orbit_Data[data] for data in self.availableData if data != sensorData]
                self.Sensors_Y = np.concatenate(Y)

                self.MovingAverage = self.SensePredDMD.MovingAverage(self.Sensors_X, self.Sensors_Y)                
        
        
        elif SET_PARAMS.SensorIsolator == "DecisionTrees":
            pass


    ############################################
    # FUNCTION TO RECOVER FROM SENSOR FAILURES #
    ############################################
    def SensorRecovery(self, failedSensor):
        if SET_PARAMS.sensorRecoveror == "EKF":
            self.sensors_kalman.pop(self.sensors_kalman.index(failedSensor))
            

    ###########################################################
    # FUNCTION FOR THE STEP BY STEP ROTATION OF THE SATELLITE #
    ###########################################################
    def rotation(self):
        ##############################################################
        #    DETERMINE WHETHER A FAUKLT OCCURED WITHIN THE SYSTEM    #
        # BASED ON THE STATISTICAL PROBABILITY DEFINED IN PARAMETERS #
        ##############################################################

        self.Fault_implementation()

        ######################################
        # DETERMINE THE DCM OF THE SATELLITE #
        ######################################
        self.A_ORC_to_SBC = Transformation_matrix(self.q)
        self.w_bo = self.w_bi - self.A_ORC_to_SBC @ np.array(([0,-self.wo,0]))

        ##################################################
        # USE SENSOR MODELS TO FIND NADIR AND SUN VECTOR #
        ##################################################
        
        #* Earth sensor
        self.r_sat_ORC, self.v_sat_EIC, self.A_EIC_to_ORC, self.r_EIC = self.sense.Earth(self.t)

        #* Sun sensor
        S_EIC, self.sun_in_view = self.sense.sun(self.t)
        self.S_ORC = self.A_EIC_to_ORC @ S_EIC

        #* Magnetometer
        self.Beta = self.sense.magnetometer(self.t) 

        self.B_ORC = self.A_EIC_to_ORC @ self.Beta 

        ##################################################
        # DETERMINE WHETHER THE SUN AND THE EARTH SENSOR #
        #   IS IN VIEW OF THE VECTOR ON THE SATELLITE    #
        ##################################################
        self.determine_sun_vision()
        self.determine_earth_vision()

        #############################################
        # ADD NOISE AND ANOMALIES TO SENSORS IN ORC #
        #############################################
        self.determine_magnetometer()
        self.determine_star_tracker()

        #* Create dictionary of all the sensors
        self.sensor_vectors = {
        "Magnetometer": {"SBC": self.B_sbc, "ORC": self.B_ORC, "noise": SET_PARAMS.Magnetometer_noise},
        "Sun_Sensor": {"SBC": self.S_sbc, "ORC": self.S_ORC, "noise": self.sun_noise}, 
        "Earth_Sensor": {"SBC": self.r_sat_sbc, "ORC": self.r_sat_ORC, "noise": SET_PARAMS.Earth_noise}, 
        "Star_tracker": {"SBC": self.star_tracker_sbc, "ORC": self.star_tracker_ORC, "noise": SET_PARAMS.star_tracker_noise}
        }

        self.q = self.rungeKutta_q(self.t, self.q, self.t+self.dt, self.dh)

        ########################################################
        # THE ERROR FOR W_BI IS WITHIN THE RUNGEKUTTA FUNCTION #
        ######################################################## 
        self.w_bi = self.rungeKutta_w(self.t, self.w_bi, self.t+self.dt, self.dh)

        self.w_bo = self.w_bi - self.A_ORC_to_SBC @ np.array(([0,-self.wo,0]))

        ########################################
        # DETERMINE THE ACTUAL POSITION OF THE #
        # SATELLITE FROM THE EARTH AND THE SUN #
        ########################################
        mean = []
        covariance = []

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
                v_ORC_k = v["ORC"]
                v_measured_k = v["SBC"]

                if not (v_ORC_k == 0.0).all():
                    # If the measured vector is equal to 0 then the sensor is not able to view the desired measurement
                    x, self.w_bo_est, P_k = self.EKF.Kalman_update(v_measured_k, v_ORC_k, self.Nm, self.Nw, self.t)
                    self.q_est = x[3:]
                    self.w_bi_est = x[:3]
                    mean.append(np.mean(x))
                    covariance.append(np.mean(P_k))



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
                v_model_k = v["ORC"]
                v_measured_k = v["SBC"]
                self.RKF.measurement_noise = v["noise"]

                if not (v_model_k == 0.0).all():
                    # If the measured vektor is equal to 0 then the sensor is not able to view the desired measurement
                    x = self.RKF.Kalman_update(v_measured_k, self.Nm, self.Nw, self.Ngyro, self.t)
                    self.w_bi_est = np.clip(x, -SET_PARAMS.wheel_angular_d_max, SET_PARAMS.wheel_angular_d_max)
                    self.q_est = self.q
        else:
            self.w_bi_est = self.w_bi
            self.q_est = self.q
            self.w_bo_est = self.w_bo

        self.update()

        self.t += self.dt

        self.KalmanControl = {"w_est": self.w_bo_est,
        "w_act": self.w_bo,
        "quaternion_est": self.q_est,
        "quaternion_actual": self.q,
        "quaternion_ref": self.control.q_ref,
        "w_ref": self.control.w_ref,
        "quaternion_error": self.control.q_e,
        "w_error": self.control.w_e
        }

        self.MeasurementUpdateDictionary = {"Mean": mean,
                            "Covariance": covariance}

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
        self.w_bo = SET_PARAMS.wbo # Angular velocity in SBC
        self.w_bo_est = self.w_bo
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
        self.Inertia = np.diag([self.Ix, self.Iy, self.Iz])
        self.Inertia_Inverse = np.linalg.inv(self.Inertia)
        self.Iw = SET_PARAMS.Iw                     # Inertia of a reaction wheel
        self.angular_momentum = SET_PARAMS.initial_angular_wheels # Angular momentum of satellite wheels
        self.angular_momentum_with_noise = self.angular_momentum
        self.faster_than_control = SET_PARAMS.faster_than_control   # If it is required that satellite must move faster around the earth than Ts
        self.control = Controller.Control()         # Controller.py is used for control of satellite    
        self.star_tracker_vector = SET_PARAMS.star_tracker_vector
        self.sun_noise = SET_PARAMS.Fine_sun_noise
        self.RKF = RKF()                            # Rate Kalman_filter
        self.EKF = EKF()                            # Extended Kalman_filter
        self.MovingAverage = 0
        self.sensors_kalman = ["Magnetometer", "Earth_Sensor", "Sun_Sensor", "Star_tracker"] #, "Star_tracker"] #Sun_Sensor, Earth_Sensor, Magnetometer
        self.DecisionTreeDMD = FaultDetection.DecisionTreePredict(path = SET_PARAMS.pathHyperParameters + '/PhysicsEnabledDMDMethod/DecisionTreesPhysicsEnabledDMD.sav')
        super().initiate_fault_parameters()
        self.availableData = SET_PARAMS.availableData
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
            "Moving Average": [],
            "Wheel Control Torques": [],
            "Magnetic Control Torques": [], 
            "Sun in view": [],                              #True or False values depending on whether the sun is in view of the satellite
            "Current fault": [],                            #What the fault is that the system is currently experiencing
            "Current fault numeric": [],
            "Current fault binary": [],
            "Wheel disturbance torques": [],
            "Gravity Gradient toques": [],
            "Gyroscopic torques": [],
            "Predicted fault": []
        }

        self.zeros = np.zeros((SET_PARAMS.number_of_faults,), dtype = int)

        self.fault = "None"                      # Current fault of the system

        #! Just for testing kalman filter
        self.est_q_error = 0
        self.est_w_error = 0

    def update(self):
        self.Orbit_Data["Magnetometer"] = self.B_sbc
        self.Orbit_Data["Sun"] = self.S_sbc
        self.Orbit_Data["Earth"] = self.r_sat_sbc
        self.Orbit_Data["Star"] = self.star_tracker_sbc
        self.Orbit_Data["Angular momentum of wheels"] = self.angular_momentum_with_noise
        self.Orbit_Data["Angular velocity of satellite"] = self.w_bi
        self.Orbit_Data["Sun in view"] = self.sun_in_view
        self.Orbit_Data["Wheel Control Torques"] = self.Nw
        self.Orbit_Data["Wheel disturbance torques"] = self.Nrw
        self.Orbit_Data["Gravity Gradient toques"] = self.Ngg
        self.Orbit_Data["Gyroscopic torques"] = self.Ngyro
        self.Orbit_Data["Magnetic Control Torques"] = self.Nm
        self.Orbit_Data["Moving Average"] = self.MovingAverage
        # Predict the sensor parameters and add them to the Orbit_Data
        self.SensorFailureHandling()

        self.Orbit_Data["Predicted fault"] = self.predictedFailure

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
        self.Inertia_Inverse = np.linalg.inv(self.Inertia)
        self.Iw = SET_PARAMS.Iw                     # Inertia of a reaction wheel
        self.angular_momentum = SET_PARAMS.initial_angular_wheels # Angular momentum of satellite wheels
        self.faster_than_control = SET_PARAMS.faster_than_control   # If it is required that satellite must move faster around the earth than Ts
        self.control = Controller.Control()         # Controller.py is used for control of satellite    
        self.star_tracker_ORC = SET_PARAMS.star_tracker_ORC
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
            "Current fault binary": [],
            "Moving Average": []
        }

        self.zeros = np.zeros((SET_PARAMS.number_of_faults,), dtype = int)

        self.fault = "None"                      # Current fault of the system

    def update(self):
        self.Orbit_Data["Magnetometer"] = self.B_sbc
        self.Orbit_Data["Sun"] = self.S_sbc
        self.Orbit_Data["Earth"] = self.r_sat_sbc
        self.Orbit_Data["Star"] = self.star_tracker_sbc
        self.Orbit_Data["Angular momentum of wheels"] = self.angular_momentum
        self.Orbit_Data["Angular velocity of satellite"] = self.w_bi
        self.Orbit_Data["Sun in view"] = self.sun_in_view
        self.Orbit_Data["Control Torques"] = self.Nw
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