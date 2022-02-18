import numpy as np
import Simulation.Controller as Controller
from Simulation.Disturbances import Disturbances
import Simulation.Parameters as Parameters
from Simulation.Quaternion_to_euler import getEulerAngles
SET_PARAMS = Parameters.SET_PARAMS 
from Simulation.Sensors import Sensors
import Simulation.Quaternion_functions as Quaternion_functions
from Simulation.Kalman_filter import RKF
from Simulation.EKF import EKF
from Simulation.SensorPredictions import SensorPredictionsDMD
import math
import Fault_prediction.Fault_detection as FaultDetection
import collections
from Simulation.utilities import crossProduct, NormalizeVector
import random
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
sc = StandardScaler()

pi = math.pi

Fault_names_to_num = SET_PARAMS.Fault_names

lengthOfSensorsXBuffer = 1

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

    y = angular + angular_momentum_derived * (h)

    return y


class Dynamics:

    def determine_magnetometer(self):
        #* Normalize self.B_ORC
        self.B_ORC = NormalizeVector(self.B_ORC)

        self.sensor_vectors["Magnetometer"]["True ORC"] = self.B_ORC.copy()
        self.sensor_vectors["Magnetometer"]["Noise ORC"] = self.Magnetometer_fault.normal_noise(self.B_ORC, SET_PARAMS.process_noise)

        self.B_sbc = self.A_ORC_to_SBC @ self.B_ORC
        ######################################################
        # IMPLEMENT ERROR OR FAILURE OF SENSOR IF APPLICABLE #
        ######################################################
        self.B_sbc_meas = self.B_sbc.copy()
        self.B_sbc_meas = self.Magnetometer_fault.normal_noise(self.B_sbc_meas, SET_PARAMS.Magnetometer_noise)
        self.B_sbc_meas = self.Magnetometer_fault.Stop_magnetometers (self.B_sbc_meas)
        self.B_sbc_meas = self.Magnetometer_fault.Interference_magnetic(self.B_sbc_meas)
        self.B_sbc_meas = self.Magnetometer_fault.Magnetometer_sensor_high_noise(self.B_sbc_meas)

        self.B_sbc_meas = self.Common_data_transmission_fault.Bit_flip(self.B_sbc_meas)
        self.B_sbc_meas = self.Common_data_transmission_fault.Sign_flip(self.B_sbc_meas)
        self.B_sbc_meas = self.Common_data_transmission_fault.Insertion_of_zero_bit(self.B_sbc_meas)


    def determine_star_tracker(self):
        self.star_tracker_ORC = self.sense.starTracker()

        # self.star_tracker_ORC = self.Star_tracker_fault.normal_noise(self.star_tracker_ORC,SET_PARAMS.process_noise)

        self.sensor_vectors["Star_tracker"]["True ORC"] = self.star_tracker_ORC.copy()
        self.sensor_vectors["Star_tracker"]["Noise ORC"] = self.Star_tracker_fault.normal_noise(self.star_tracker_ORC, SET_PARAMS.process_noise)

        #* Star tracker
        self.star_tracker_sbc = self.A_ORC_to_SBC @ self.star_tracker_ORC

        self.star_tracker_sbc_meas = self.Star_tracker_fault.normal_noise(self.star_tracker_sbc,SET_PARAMS.star_tracker_noise)

        self.star_tracker_sbc_meas = self.Star_tracker_fault.Closed_shutter(self.star_tracker_sbc)

        # self.star_tracker_sbc = NormalizeVector(self.star_tracker_sbc)


    def determine_earth_vision(self):
        #################################################################
        #      FOR THIS SPECIFIC SATELLITE MODEL, THE EARTH SENSOR      #
        #                    IS FIXED TO THE -Z FACE                    #
        # THIS IS ACCORDING TO THE ORBIT AS DEFINED BY JANSE VAN VUUREN #
        #             THIS IS DETERMINED WITH THE SBC FRAME             #
        #################################################################
        #* self.r_sat_ORC is already normalized
        self.sensor_vectors["Earth_Sensor"]["True ORC"] = self.r_sat_ORC.copy()
        self.sensor_vectors["Earth_Sensor"]["Noise ORC"] = self.Earth_sensor_fault.normal_noise(self.r_sat_ORC, SET_PARAMS.process_noise)

        self.r_sat_sbc = self.A_ORC_to_SBC @ self.r_sat_ORC

        self.r_sat_sbc_meas = self.Earth_sensor_fault.normal_noise(self.r_sat_sbc, SET_PARAMS.Earth_noise)

        self.earthSeenBySensor = True

        angle_difference = Quaternion_functions.rad2deg(np.arccos(np.clip(np.dot(self.r_sat_sbc, SET_PARAMS.Earth_sensor_position),-1,1)))

        if angle_difference < SET_PARAMS.Earth_sensor_angle:
            self.earthSeenBySensor = True
            self.r_sat_sbc_meas = self.Earth_sensor_fault.normal_noise(self.r_sat_sbc, SET_PARAMS.Earth_noise)
            self.r_sat_sbc_meas = self.Earth_sensor_fault.Earth_sensor_high_noise(self.r_sat_sbc)
            self.r_sat_sbc_meas = self.Common_data_transmission_fault.Bit_flip(self.r_sat_sbc)
            self.r_sat_sbc_meas = self.Common_data_transmission_fault.Sign_flip(self.r_sat_sbc)
            self.r_sat_sbc_meas = self.Common_data_transmission_fault.Insertion_of_zero_bit(self.r_sat_sbc) 

        else:
            self.r_sat_sbc_meas = np.zeros(self.r_sat_sbc_meas.shape)
            self.r_sat_sbc = np.zeros(self.r_sat_sbc.shape)
            self.r_sat_ORC = np.zeros(self.r_sat_ORC.shape)
            self.sensor_vectors["Earth_Sensor"]["True ORC"] = np.zeros(self.r_sat_ORC.shape)
            self.sensor_vectors["Earth_Sensor"]["Noise ORC"] = np.zeros(self.r_sat_ORC.shape)
            #* self.r_sat_ORC is already normalized
            self.earthSeenBySensor = False

    def determine_sun_vision(self):
        #################################################################
        #    FOR THIS SPECIFIC SATELLITE MODEL, THE FINE SUN SENSOR     #
        #       IS FIXED TO THE +X FACE AND THE COARSE SUN SENSOR       #
        #                   IS FIXED TO THE -X FACE.                    #
        # THIS IS ACCORDING TO THE ORBIT AS DEFINED BY JANSE VAN VUUREN #
        #             THIS IS DETERMINED WITH THE SBC FRAME             #
        #################################################################
        #* Normalize self.S_ORC
        self.S_ORC = NormalizeVector(self.S_ORC)

        self.sensor_vectors["Sun_Sensor"]["True ORC"] = self.S_ORC.copy()
        self.sensor_vectors["Sun_Sensor"]["Noise ORC"] = self.Sun_sensor_fault.normal_noise(self.S_ORC, SET_PARAMS.process_noise)

        self.S_sbc = self.A_ORC_to_SBC @ self.S_ORC

        self.S_sbc_meas = self.Sun_sensor_fault.normal_noise(self.S_sbc, SET_PARAMS.Coarse_sun_noise)

        self.SunSeenBySensor = True

        reflection = False

        if self.sun_in_view:
            angle_difference_fine = Quaternion_functions.rad2deg(np.arccos(np.dot(self.S_sbc, SET_PARAMS.Fine_sun_sensor_position)))
            angle_difference_coarse = Quaternion_functions.rad2deg(np.arccos(np.dot(self.S_sbc, SET_PARAMS.Coarse_sun_sensor_position)))

            if angle_difference_fine < SET_PARAMS.Fine_sun_sensor_angle: 
                self.SunSeenBySensor = True
                self.S_ORC_reflection, reflection = self.Sun_sensor_fault.Reflection_sun(self.S_sbc, self.S_ORC, "Fine")

                self.S_sbc = self.A_ORC_to_SBC @ self.S_ORC_reflection

                self.S_sbc_meas = self.Sun_sensor_fault.normal_noise(self.S_sbc, SET_PARAMS.Fine_sun_noise)

                ######################################################
                # IMPLEMENT ERROR OR FAILURE OF SENSOR IF APPLICABLE #
                ######################################################

                self.S_sbc_meas = self.Sun_sensor_fault.Catastrophic_sun(self.S_sbc, "Fine")
                self.S_sbc_meas = self.Sun_sensor_fault.Erroneous_sun(self.S_sbc, "Fine")
                self.S_sbc_meas = self.Common_data_transmission_fault.Bit_flip(self.S_sbc)
                self.S_sbc_meas = self.Common_data_transmission_fault.Sign_flip(self.S_sbc)
                self.S_sbc_meas = self.Common_data_transmission_fault.Insertion_of_zero_bit(self.S_sbc)  

                self.sun_noise = SET_PARAMS.Fine_sun_noise

            elif angle_difference_coarse < SET_PARAMS.Coarse_sun_sensor_angle:
                self.SunSeenBySensor = True
                self.S_ORC_reflection, reflection = self.Sun_sensor_fault.Reflection_sun(self.S_sbc, self.S_ORC, "Coarse")

                self.S_sbc = self.A_ORC_to_SBC @ self.S_ORC_reflection

                self.S_sbc_meas = self.Sun_sensor_fault.normal_noise(self.S_sbc, SET_PARAMS.Coarse_sun_noise)

                ######################################################
                # IMPLEMENT ERROR OR FAILURE OF SENSOR IF APPLICABLE #
                ######################################################

                self.S_sbc_meas = self.Sun_sensor_fault.Catastrophic_sun(self.S_sbc, "Coarse")
                self.S_sbc_meas = self.Sun_sensor_fault.Erroneous_sun(self.S_sbc, "Coarse")
                self.S_sbc_meas = self.Common_data_transmission_fault.Bit_flip(self.S_sbc)
                self.S_sbc_meas = self.Common_data_transmission_fault.Sign_flip(self.S_sbc)
                self.S_sbc_meas = self.Common_data_transmission_fault.Insertion_of_zero_bit(self.S_sbc)  

                self.sun_noise = SET_PARAMS.Coarse_sun_noise
            else:
                self.S_sbc = np.zeros(3)
                self.S_sbc_meas = np.zeros(3)
                self.S_ORC = np.zeros(3)
                self.S_ORC_reflection = np.zeros(3)
                self.SunSeenBySensor = False
                reflection = False

        else:
            self.S_sbc = np.zeros(3)
            self.S_sbc_meas = np.zeros(3)
            self.S_ORC = np.zeros(3)
            self.S_ORC_reflection = np.zeros(3)
            self.SunSeenBySensor = False
            reflection = False

        # self.S_sbc = NormalizeVector(self.S_sbc)
        self.reflection = reflection
        self.sensor_vectors["Sun_Sensor"]["ORC Measured"] = self.S_ORC_reflection.copy()

        

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
        #! Third change to implement correct control

        if SET_PARAMS.Model_or_Measured == "Model" and self.predictedFailedSensor == "Sun":
            Sun_vector = self.sensor_vectors["Sun_Sensor"]["True ORC"]
        else:
            Sun_vector = self.sensor_vectors["Sun_Sensor"]["ORC Measured"]

        if self.predictedFailedSensor == "Magnetometer":
            Magnetometer_vector = self.sensor_vectors["Magnetometer"]["Estimated SBC"]
        else:
            Magnetometer_vector = self.B_sbc

        if self.predictedFailedSensor == "Earth":
            Earth_vector = self.sensor_vectors["Earth_Sensor"]["Estimated SBC"]
        else:    
            Earth_vector = self.r_sat_sbc
        
        N_control_magnetic, N_control_wheel = self.control.control(self.w_bi_est, self.w_bo_est, self.q_est, self.Inertia, Magnetometer_vector, self.angular_momentum_wheels_with_noise, Earth_vector, Sun_vector, self.sun_in_view)

        if "RW" in self.fault:
            N_control_wheel = self.Reaction_wheel_fault.Electronics_of_RW_failure(N_control_wheel)
            N_control_wheel = self.Reaction_wheel_fault.Overheated_RW(N_control_wheel)
            N_control_wheel = self.Reaction_wheel_fault.Catastrophic_RW(N_control_wheel)
            N_control_wheel = self.Control_fault.Increasing_angular_RW_momentum(N_control_wheel)
            N_control_wheel = self.Control_fault.Decreasing_angular_RW_momentum(N_control_wheel)
            N_control_wheel = self.Control_fault.Oscillating_angular_RW_momentum(N_control_wheel)

        if SET_PARAMS.no_aero_disturbance:
            N_aero = np.zeros(3)
        else:
            N_aero = self.dist.Aerodynamic2(self.A_ORC_to_SBC, self.A_EIC_to_ORC, self.sun_in_view)

        ###################################
        # DISTURBANCE OF GRAVITY GRADIENT #
        ###################################

        Ngg = self.dist.Gravity_gradient_func(self.A_ORC_to_SBC) 

        n = int(np.round((x - x0)/h))
        y = w

        N_control = N_control_magnetic - N_control_wheel

        ######################################################
        # ALL THE DISTURBANCE TORQUES ADDED TO THE SATELLITE #
        ######################################################

        for _ in range(n):
            #############################################
            # DISTURBANCE OF A REACTION WHEEL IMBALANCE #
            #############################################
            if SET_PARAMS.no_wheel_disturbance:
                N_rw = np.zeros(3)
            else:
                N_rw = self.dist.Wheel_Imbalance(self.Iw_Inverse @ self.angular_momentum_wheels, h) #! was x0-x

            N_gyro = crossProduct(y,(self.Inertia @ y + self.angular_momentum_wheels))

            N_disturbance = Ngg + N_aero + N_rw - N_gyro   

            N = N_control + N_disturbance

            k1 = h*((self.Inertia_Inverse @ N)) 
            k2 = h*((self.Inertia_Inverse @ N) + 0.5*k1) 
            k3 = h*((self.Inertia_Inverse @ N) + 0.5*k2) 
            k4 = h*((self.Inertia_Inverse @ N) + k3) 
            y = y + (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            
            x0 = x0 + h

            self.angular_momentum_wheels = rungeKutta_h(x0, self.angular_momentum_wheels, x, h, N_control_wheel)
        
        self.Ngyro = N_gyro
        self.Nm = N_control_magnetic
        self.Nw = N_control_wheel
        self.Ngg = Ngg
        self.Nrw = N_rw
        self.Naero = N_aero

        self.angular_momentum_wheels_with_noise = self.Angular_sensor_fault.normal_noise(self.angular_momentum_wheels, SET_PARAMS.Angular_sensor_noise)
        self.angular_momentum_wheels_with_noise = self.Angular_sensor_fault.Angular_sensor_high_noise(self.angular_momentum_wheels)

        y = np.clip(y, -SET_PARAMS.angularSatelliteMax, SET_PARAMS.angularSatelliteMax)

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
        
        if (np.isnan(y).any() or (y == 0).all()) and SET_PARAMS.printBreak:
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
        reset = False
        sensorFailed = "None"
        Sensors_X, Sensors_Y, MovingAverageDict = self.SensorFeatureExtraction()

        if SET_PARAMS.SensorFDIR:
            self.DefineIfFault()

            self.predictedFailureValue = self.SensorPredicting(Sensors_X)

            if self.predictedFailureValue > 0.85:
                self.predictedFailure = True
            else:
                self.predictedFailure = False

            # If a failure is predicted the cause of the failure must be determined
            sensorFailed = self.SensorIsolation(MovingAverageDict, Sensors_X, self.predictedFailure)
            self.predictedFailedSensor = sensorFailed
            # After the specific sensor that has failed is identified 
            # The system must recover
            reset = self.SensorRecovery(sensorFailed)


        self.prevFailedSensor = sensorFailed

        return reset

    def SensorFeatureExtraction(self):
        modelledSun = self.A_ORC_to_SBC_est @ self.sensor_vectors["Sun_Sensor"]["True ORC"]

        modelledMagnetometer = self.A_ORC_to_SBC @ self.sensor_vectors["Magnetometer"]["True ORC"]

        modelledEarth = self.A_ORC_to_SBC @ self.sensor_vectors["Earth_Sensor"]["True ORC"]

        Sensors_X = np.concatenate([self.Orbit_Data["Sun"], modelledSun,
                                        self.Orbit_Data["Magnetometer"], modelledMagnetometer, 
                                        self.Orbit_Data["Earth"], modelledEarth])
        Sensors_Y = np.concatenate([self.Orbit_Data["Wheel Control Torques"], 
                                self.Orbit_Data["Magnetic Control Torques"]])

        MovingAverageDict = {}

        if SET_PARAMS.FeatureExtraction == "DMD":         

            if self.t == SET_PARAMS.time:
                # Initiating parameters for SensorPredictions
                self.SensePredDMDALL = SensorPredictionsDMD(Sensors_X, "ALL")             


            self.MovingAverage = self.SensePredDMDALL.MovingAverage(Sensors_X, Sensors_Y)

            self.MovingAverage = self.MovingAverage.flatten()              

        return Sensors_X, Sensors_Y, MovingAverageDict

    #################################################
    # FUNCTION TO PREDICT IF AN ANOMALY HAS OCCURED #
    #################################################
    def SensorPredicting(self, Sensors_X):
        predictedFailure = False
        
        if SET_PARAMS.SensorPredictor == "Constellation-DecisionTrees":
            predictedFailure = self.DecisionTreeDMDBinary.Predict(self.constellationData)

        elif SET_PARAMS.SensorPredictor == "DecisionTrees":
            Sensors_X = np.array([np.concatenate([Sensors_X, self.MovingAverage])])
            Sensors_X = np.array([np.concatenate([Sensors_X, self.Orbit_Data["Angular momentum of wheels"]])])
            Sensors_X = Sensors_X.reshape(1, Sensors_X.shape[1])
            predictedFailure = self.DecisionTreeDMDBinary.Predict(Sensors_X)
            # self.SensorsXBuffer.append(self.Orbit_Data["Sun"])
            # if len(self.SensorsXBuffer) == lengthOfSensorsXBuffer:
            #     X = np.array(self.SensorsXBuffer).reshape((1, self.Orbit_Data["Sun"].shape[0]*lengthOfSensorsXBuffer))
            # predictedFailure = self.DecisionTreeDMDBinary.Predict(X)

        elif SET_PARAMS.SensorPredictor == "RandomForest":
            Sensors_X = np.array([np.concatenate([Sensors_X, self.Orbit_Data["Angular momentum of wheels"]])])
            Sensors_X = Sensors_X.reshape(1, Sensors_X.shape[1])
            predictedFailure = self.RandomForestDMDBinary.Predict(Sensors_X)

        elif SET_PARAMS.SensorPredictor == "PERFECT":
            if self.implementedFault != "None":
                predictedFailure = True

        elif SET_PARAMS.SensorPredictor == "ANN":
            Sensors_X = np.array([np.concatenate([Sensors_X, self.MovingAverage])])
            Sensors_X = np.asarray(Sensors_X).astype(np.float32)
            NN_Basic = load_model("models/ANN")
            predictedFailure = NN_Basic.predict(Sensors_X).round()

        elif SET_PARAMS.SensorPredictor == "RandomChoice":
            predictedFailure = True if random.uniform(0,1) < 0.5 else False

        elif isinstance(SET_PARAMS.SensorPredictor, float):
            if self.implementedFault != "None":
                randomValue = random.uniform(0,1)
                predictedFailure = True if randomValue < SET_PARAMS.SensorPredictor/100 else False
            else:
                predictedFailure = False

        elif SET_PARAMS.SensorPredictor == "Std":
            self.SensorsXBuffer.append(Sensors_X)

            std = np.std(np.array(self.SensorsXBuffer))

            self.SensorsXstd.append(std)

        elif SET_PARAMS.SensorPredictor == "Isolation_Forest":
            Sensors_X = np.array([np.concatenate([Sensors_X, self.Orbit_Data["Angular momentum of wheels"]])])
            self.SensorsXBuffer.append(self.Orbit_Data["Sun"])
            if len(self.SensorsXBuffer) == lengthOfSensorsXBuffer:
                X = np.array(self.SensorsXBuffer).reshape((1, self.Orbit_Data["Sun"].shape[0]*lengthOfSensorsXBuffer))
                predictedFailure = self.IsolationForest.Predict(X)

        elif SET_PARAMS.SensorPredictor == "SBCvsORC":
            angle = Quaternion_functions.rad2deg(np.arccos(np.clip(np.dot(self.sensor_vectors["Sun_Sensor"]["Noise SBC"], self.A_ORC_to_SBC_est@self.sensor_vectors["Sun_Sensor"]["True ORC"]),-1,1)))
            if angle > 5 and angle != 90:
                predictedFailure = True

        if SET_PARAMS.PredictionBuffer:
            if not predictedFailure:
                predictedFailure = float(self.predictedFailureValue)*0.9 + float(predictedFailure)*0.1

        return predictedFailure


    def DefineIfFault(self):
        if not self.reflection and self.fault == "Reflection":
            fault = "None"
        elif not self.SunSeenBySensor and self.fault in SET_PARAMS.SunFailures:
            fault = "None"
        elif not self.earthSeenBySensor and self.fault in SET_PARAMS.EarthFailures:
            fault = "None"
        else:
            fault = self.fault
        
        if fault == "None":
            FailedSensor = "None"

        elif fault in SET_PARAMS.SunFailures:
            FailedSensor = "Sun"
        
        elif fault in SET_PARAMS.EarthFailures:
            FailedSensor = "Earth"

        elif fault in SET_PARAMS.starTrackerFailures:
            FailedSensor = "Star"

        elif fault in SET_PARAMS.magnetometerFailures:
            FailedSensor = "Magnetometer"

        self.implementedFault = fault
        self.implementedFailedSensor = FailedSensor


    ###############################################################
    # FUNCTION TO ISOLATE (CLASSIFY) THE ANOMALY THAT HAS OCCURED #
    ###############################################################
    def SensorIsolation(self, MovingAverageDict, Sensors_X, predictedFailure):
        FailedSensor = "None"

        if predictedFailure:
            #! This should account for multiple predictions of failures
            if SET_PARAMS.SensorIsolator == "DMD":
                FailedSensor = max(zip(MovingAverageDict.values(), MovingAverageDict.keys()))[1]
            
            elif SET_PARAMS.SensorIsolator == "DecisionTrees":
                arrayFault = self.DecisionTreeDMDMulti.Predict(np.array([np.concatenate([Sensors_X, self.MovingAverage.flatten()])]))
                arrayFault = arrayFault.replace("]", "").replace("[", "").replace(" ", "")
                arrayFault = arrayFault.split(",")
                indexFault = arrayFault.index("1")
                fault = SET_PARAMS.faultnames[indexFault]
                if fault in SET_PARAMS.SunFailures:
                    FailedSensor = "Sun"
                
                elif fault in SET_PARAMS.EarthFailures:
                    FailedSensor = "Earth"

                elif fault in SET_PARAMS.starTrackerFailures:
                    FailedSensor = "Star"

                elif fault in SET_PARAMS.magnetometerFailures:
                    FailedSensor = "Magnetometer"

            elif SET_PARAMS.SensorIsolator == "RandomForest":
                arrayFault = self.RandomForestDMDMulti.Predict(np.array([np.concatenate([Sensors_X, self.MovingAverage.flatten()])]))
                arrayFault = arrayFault.replace("]", "").replace("[", "").replace(" ", "")
                arrayFault = arrayFault.split(",")
                indexFault = arrayFault.index("1")
                fault = SET_PARAMS.faultnames[indexFault]
                if fault in SET_PARAMS.SunFailures:
                    FailedSensor = "Sun"
                
                elif fault in SET_PARAMS.EarthFailures:
                    FailedSensor = "Earth"

                elif fault in SET_PARAMS.starTrackerFailures:
                    FailedSensor = "Star"

                elif fault in SET_PARAMS.magnetometerFailures:
                    FailedSensor = "Magnetometer"

            elif SET_PARAMS.SensorIsolator == "PERFECT":
                FailedSensor = self.implementedFailedSensor

            elif SET_PARAMS.SensorIsolator == "OnlySun":
                FailedSensor = "Sun"

            elif isinstance(SET_PARAMS.SensorIsolator, int):
                FailedSensor = "Sun"
    
        else:
            FailedSensor = "None"


        return FailedSensor

    ############################################
    # FUNCTION TO RECOVER FROM SENSOR FAILURES #
    ############################################
    def SensorRecovery(self, failedSensor):
        reset = False
        sensors_kalman = SET_PARAMS.kalmanSensors.copy()
        # The EKF method of recovery resets the kalman filter 
        # if the predictedFailed sensor changes

        # Always use the top3 sensors (the sensors with the smalles error between the previous estimated SBC and the measured SBC)
        if SET_PARAMS.SensorRecoveror == "EKF-top3":
            Error = 0

            for _ in range(len(sensors_kalman)-3):
                Error = 0
                for sensor in sensors_kalman:
                    v = self.sensor_vectors[sensor]
                    v_est_k = v["Estimated SBC"]
                    v_measured_k = v["Noise SBC"]
                    e = np.square((v_est_k - v_measured_k)**2).mean()

                    if e > Error:
                        Error = e
                        failedSensor = sensor

                sensors_kalman.pop(sensors_kalman.index(failedSensor))
            
            self.sensors_kalman = sensors_kalman
        
        elif SET_PARAMS.SensorRecoveror == "EKF-top2":
            
            for _ in range(len(sensors_kalman)-2):
                Error = 0
                for sensor in sensors_kalman:
                    v = self.sensor_vectors[sensor]
                    v_est_k = v["Estimated SBC"]
                    v_measured_k = v["Noise SBC"]
                    e = np.square((v_est_k - v_measured_k)**2).mean()

                    if e > Error:
                        Error = e
                        failedSensor = sensor

                sensors_kalman.pop(sensors_kalman.index(failedSensor))
            
            self.sensors_kalman = sensors_kalman

        elif SET_PARAMS.SensorRecoveror == "EKF-combination":
            if failedSensor != "None":
                if SET_PARAMS.availableSensors[failedSensor] in sensors_kalman:
                    sensors_kalman.pop(sensors_kalman.index(SET_PARAMS.availableSensors[failedSensor]))

            if failedSensor != self.prevFailedSensor:
                self.failedNumber += 1
                if self.failedNumber % SET_PARAMS.NumberOfFailuresReset == 0:
                    reset = True
            
            self.sensors_kalman = sensors_kalman

        elif SET_PARAMS.SensorRecoveror == "EKF-ignore":
            if failedSensor != "None":
                if SET_PARAMS.availableSensors[failedSensor] in sensors_kalman:
                    sensors_kalman.pop(sensors_kalman.index(SET_PARAMS.availableSensors[failedSensor]))
            
            self.sensors_kalman = sensors_kalman

        elif SET_PARAMS.SensorRecoveror == "EKF-reset":

            if failedSensor != "None":
                if SET_PARAMS.availableSensors[failedSensor] in sensors_kalman:
                    sensors_kalman.pop(sensors_kalman.index(SET_PARAMS.availableSensors[failedSensor]))
                
                if SET_PARAMS.availableSensors[failedSensor] == "Sun_Sensor":
                    self.R_k = np.eye(3)*1e-2 # Changed from 1e-3 which was good

            if failedSensor != self.prevFailedSensor:
                reset = True
            
            self.sensors_kalman = sensors_kalman

        if SET_PARAMS.SensorRecoveror == "EKF-replacement":
            if failedSensor != "None":
                if SET_PARAMS.availableSensors[failedSensor] != "Sun_Sensor":
                    self.sensor_vectors[SET_PARAMS.availableSensors[failedSensor]]["SBC"] = self.Sun_sensor_fault.normal_noise(self.A_ORC_to_SBC_est @ self.sensor_vectors[SET_PARAMS.availableSensors[failedSensor]]["True ORC"], SET_PARAMS.process_noise)
                elif self.sun_in_view:
                    self.sensor_vectors[SET_PARAMS.availableSensors[failedSensor]]["SBC"] = self.Sun_sensor_fault.normal_noise(self.A_ORC_to_SBC_est @ self.sensor_vectors[SET_PARAMS.availableSensors[failedSensor]]["True ORC"], SET_PARAMS.process_noise)

            self.sensors_kalman = sensors_kalman
        
        if failedSensor == "None":
            self.R_k = SET_PARAMS.R_k
        
        return reset
            

    ###########################################################
    # FUNCTION FOR THE STEP BY STEP ROTATION OF THE SATELLITE #
    ###########################################################
    def rotation(self):
        ##############################################################
        #     DETERMINE WHETHER A FAULT OCCURED WITHIN THE SYSTEM    #
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

        ###########################################
        # DETERMINE THE ESTIMATED POSITION OF THE #
        #   SATELLITE FROM THE EARTH AND THE SUN  #
        ###########################################
        # The SBC value can be changed by some of the recovery methods, but should not change the actual value
        # Consequently, two values are created which should usually be the same unless
        # the recovery method changes the value of SBC (that is the only use thereof)
        self.sensor_vectors["Magnetometer"]["True SBC"] = self.B_sbc
        self.sensor_vectors["Magnetometer"]["Noise SBC"] = self.B_sbc_meas

        self.sensor_vectors["Sun_Sensor"]["True SBC"] = self.S_sbc
        self.sensor_vectors["Sun_Sensor"]["Noise SBC"] = self.S_sbc_meas

        self.sensor_vectors["Earth_Sensor"]["True SBC"] = self.r_sat_sbc
        self.sensor_vectors["Earth_Sensor"]["Noise SBC"] = self.r_sat_sbc_meas

        self.sensor_vectors["Star_tracker"]["True SBC"] = self.star_tracker_sbc
        self.sensor_vectors["Star_tracker"]["Noise SBC"] = self.star_tracker_sbc_meas

        # Predict whether sensors have failed
        reset = self.SensorFailureHandling()

        #Create a state of sensors that has all the data except the data from the excluded sensor
        currentState = {key: self.sensor_vectors[key] for key in self.sensor_vectors if key != self.predictedFailedSensor}

        currentState["Nw"] = self.Nw 
        currentState["Nm"] = self.Nm
        currentState["time"] = self.t
        currentState["q_est"] = self.q_est
        currentState["w_bo_est"] = self.w_bo_est
        currentState["w_bi_est"] = self.w_bi_est
        currentState["angular_momentum_est"] = self.angular_momentum_est
        currentState["P_k_est"] = self.P_k_est

        self.stateBuffer.append(currentState.copy())

        mean = []
        covariance = []

        if (reset and SET_PARAMS.Kalman_filter_use == "EKF"):          
            self.EKF = EKF()

            # Change the EKF to a new measurement noise covariance matrix when reset
            self.EKF.R_k = self.R_k
            self.A_ORC_to_SBC_est, x, self.w_bo_est, P_k, self.angular_momentum_est, K_k = resetEKF(self.EKF, self.stateBuffer, self.sensors_kalman)
            self.P_k_est = P_k
            self.K_k_est = K_k
            self.q_est = x[3:]
            self.w_bi_est = x[:3]
            

        elif SET_PARAMS.Kalman_filter_use == "EKF":
            # print(self.sensors_kalman)
            for sensor in self.sensors_kalman:
                # Step through both the sensor noise and the sensor measurement
                # vector is the vector of the sensor's measurement
                # This is used to compare it to the modelled measurement
                # Consequently, the vector is the ORC modelled vector before
                # the transformation Matrix is implemented on the vector
                # Since the transformation matrix takes the modelled and measured into account
                # Only noise is added to the measurement
                v = self.sensor_vectors[sensor]
                v_ORC_k = v["True ORC"]
                v_measured_k = v["Noise SBC"]

                if not (v_measured_k == 0.0).all():
                    # If the measured vector is equal to 0 then the sensor is not able to view the desired measurement
                    self.A_ORC_to_SBC_est, x, self.w_bo_est, P_k, self.angular_momentum_est, K_k = self.EKF.Kalman_update(v_measured_k, v_ORC_k, self.Nm, self.Nw, self.t)
            
            self.P_k_est = P_k
            self.K_k_est = K_k
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
                v_model_k = v["True ORC"]
                v_measured_k = v["True SBC"]
                self.RKF.measurement_noise = v["noise"]

                if not (v_model_k == 0.0).all():
                    # If the measured vektor is equal to 0 then the sensor is not able to view the desired measurement
                    x = self.RKF.Kalman_update(v_measured_k, self.Nm, self.Nw, self.Ngyro, self.t)
                    self.w_bi_est = x
                    self.q_est = self.q
        else:
            self.K_k_est = np.eye(3)
            self.P_k_est = np.eye(7)
            self.A_ORC_to_SBC_est = self.A_ORC_to_SBC
            self.w_bi_est = self.w_bi
            self.q_est = self.q
            self.w_bo_est = self.w_bo

        # Update the estimated vector in SBC
        for sensor in ["Magnetometer", "Earth_Sensor", "Sun_Sensor", "Star_tracker"]:
            v_ORC_k = self.sensor_vectors[sensor]["True ORC"].copy()
            self.sensor_vectors[sensor]["Estimated SBC"] = self.A_ORC_to_SBC_est @ v_ORC_k

        self.q = self.rungeKutta_q(self.t, self.q, self.t+self.dt, self.dh)

        ########################################################
        # THE ERROR FOR W_BI IS WITHIN THE RUNGEKUTTA FUNCTION #
        ######################################################## 
        self.w_bi = self.rungeKutta_w(self.t, self.w_bi, self.t+self.dt, self.dh)

        self.w_bo = self.w_bi - self.A_ORC_to_SBC @ np.array(([0,-self.wo,0]))

        self.q_ref = self.control.q_ref
        self.w_bo_ref = self.control.w_ref

        self.MeasurementUpdateDictionary = {"Mean": mean,
                            "Covariance": covariance}

        self.update()

        self.t += self.dt

        return self.w_bi, self.q, self.A_ORC_to_SBC, self.r_EIC, self.sun_in_view


#@njit
def resetEKF(EKF, stateBuffer, sensorsKalman):
    #! Change R_k and Q_k depending on sensor that failed
    #! if self.predictedFailedSensor

    lastState = stateBuffer[-1]
    EKF.q = lastState["q_est"]
    EKF.w_bi = lastState["w_bi_est"]
    EKF.w_bo = lastState["w_bo_est"]
    EKF.angular_momentum = lastState["angular_momentum_est"]
    EKF.t = lastState["time"] - 1
    # EKF.P_k = lastState["P_k_est"]
    for state in stateBuffer:
        t = state["time"]
        Nw = state["Nm"]
        Nm = state["Nm"]

        for sensor in sensorsKalman:
            if sensor in state:
                v = state[sensor]
                v_ORC_k = v["Noise ORC"]
                v_measured_k = v["Noise SBC"]

                if not (v_measured_k == 0.0).all():
                    # If the measured vector is equal to 0 then the sensor is not able to view the desired measurement
                    A_ORC_to_SBC_est, x, w_bo_est, P_k, angular_momentum_est, K_k = EKF.Kalman_update(v_measured_k, v_ORC_k, Nm, Nw, t)

    return A_ORC_to_SBC_est, x, w_bo_est, P_k, angular_momentum_est, K_k

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
        self.angular_momentum_wheels = SET_PARAMS.initial_angular_wheels 
        self.angular_momentum_wheels_with_noise = SET_PARAMS.initial_angular_wheels 
        self.q = SET_PARAMS.quaternion_initial      # Quaternion position
        self.q_est = self.q
        self.t = SET_PARAMS.time                    # Beginning time
        self.dt = SET_PARAMS.Ts                     # Time step
        self.dh = self.dt/SET_PARAMS.NumberOfIntegrationSteps                        # Size of increments for Runga-kutta method
        self.Ix = SET_PARAMS.Ix                     # Ixx inertia
        self.Iy = SET_PARAMS.Iy                     # Iyy inertia
        self.Iz = SET_PARAMS.Iz                     # Izz inertia
        self.R_k = SET_PARAMS.R_k 
        self.failedNumber = 0                       # Number of time normal behaviour change to anomalous behaviour
        self.Inertia = np.diag([self.Ix, self.Iy, self.Iz])
        self.Inertia_Inverse = np.linalg.inv(self.Inertia)
        self.Iw = SET_PARAMS.Iw                     # Inertia of a reaction wheel
        self.Iw_Inverse = np.linalg.inv(self.Iw)    # Inverse Inertia of a reaction wheel
        self.angular_momentum = SET_PARAMS.initial_angular_wheels # Angular momentum of satellite
        self.angular_momentum_est = self.angular_momentum
        self.angular_wheel_momentum_with_noise = self.angular_momentum
        self.faster_than_control = SET_PARAMS.faster_than_control   # If it is required that satellite must move faster around the earth than Ts
        self.control = Controller.Control()         # Controller.py is used for control of satellite    
        self.star_tracker_vector = SET_PARAMS.star_tracker_vector
        self.sun_noise = SET_PARAMS.Fine_sun_noise
        self.RKF = RKF()                            # Rate Kalman_filter
        self.EKF = EKF()                            # Extended Kalman_filter
        self.MovingAverage = 0
        self.predictedFailureValue = 0
        self.sensors_kalman = SET_PARAMS.kalmanSensors #Sun_Sensor, Earth_Sensor, Magnetometer
        self.DecisionTreeDMDBinary = FaultDetection.DecisionTreePredict(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod/DecisionTreesPhysicsEnabledDMDBinaryClass.sav')
        self.DecisionTreeDMDMulti = FaultDetection.DecisionTreePredict(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod/DecisionTreesPhysicsEnabledDMDMultiClass.sav')
        self.RandomForestDMDBinary = FaultDetection.DecisionTreePredict(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod/RandomForestPhysicsEnabledDMDBinaryClass.sav')
        self.RandomForestDMDMulti = FaultDetection.DecisionTreePredict(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod/RandomForestPhysicsEnabledDMDMultiClass.sav')
        self.IsolationForest = FaultDetection.IsolationForest(path = 'models/IsolationForest.sav')
        # self.NN_Basic = load_model("models/ANN")
        self.stateBuffer = collections.deque(maxlen = SET_PARAMS.stateBufferLength)
        super().initiate_fault_parameters()

        self.SensorsXBuffer = collections.deque(maxlen = lengthOfSensorsXBuffer)
        self.SensorsXstd = collections.deque(maxlen = lengthOfSensorsXBuffer)
        self.star_tracker_ORC = self.star_tracker_vector
        self.availableData = SET_PARAMS.availableData
        self.predictedFailure = 0
        self.SensePredDMDDict = {}
        self.prevFailedSensor = "None"
        self.A_ORC_to_SBC_est = np.zeros((3,3))
        self.P_k_est = SET_PARAMS.P_k
        self.constellationData = []
        self.Nm, self.Nw = np.zeros(3), np.zeros(3)
        self.globalArray = ["Sun_x",
            "Sun_y",
            "Sun_z",
            "Sun_modelled_x",
            "Sun_modelled_y",
            "Sun_modelled_z",
            "Magnetometer_x",    #B vector in SBC
            "Magnetometer_y", 
            "Magnetometer_z", 
            "Magnetometer_modelled_x",
            "Magnetometer_modelled_y",
            "Magnetometer_modelled_z",
            "Earth_x",           #Satellite position vector in ORC
            "Earth_y",
            "Earth_z",
            "Earth_modelled_x",
            "Earth_modelled_y",
            "Earth_modelled_z",
            "Angular momentum of wheels_x",    #Wheel angular velocity of each reaction wheel
            "Angular momentum of wheels_y", 
            "Angular momentum of wheels_z", 
            "Star_x",
            "Star_y",
            "Star_z",
            "Star_modelled_x",
            "Star_modelled_y",
            "Star_modelled_z",
            "Angular velocity of satellite actual_x",
            "Angular velocity of satellite actual_y",
            "Angular velocity of satellite actual_z",
            "Angular velocity of satellite estimated_x",
            "Angular velocity of satellite estimated_y",
            "Angular velocity of satellite estimated_z",
            "Angular velocity of satellite reference_x",
            "Angular velocity of satellite reference_y",
            "Angular velocity of satellite reference_z",
            "Moving Average",
            "Wheel Control Torques_x",
            "Wheel Control Torques_y",
            "Wheel Control Torques_z",
            "Magnetic Control Torques_x",
            "Magnetic Control Torques_y",
            "Magnetic Control Torques_z",
            "Sun in view",                              #True or False values depending on whether the sun is in view of the satellite
            "Current fault",                            #What the fault is that the system is currently experiencing
            "Current fault numeric",
            "Current fault binary",
            "Wheel disturbance Torques_x",
            "Wheel disturbance Torques_y",
            "Wheel disturbance Torques_z",
            "Gravity Gradient Torques_x",
            "Gravity Gradient Torques_y",
            "Gravity Gradient Torques_z",
            "Gyroscopic Torques_x",
            "Gyroscopic Torques_y",
            "Gyroscopic Torques_z",
            "Aerodynamic Torques_x",
            "Aerodynamic Torques_y",
            "Aerodynamic Torques_z",
            "Predicted fault",
            "Isolation Accuracy",
            "Prediction Accuracy",
            "Quaternions Actual_x",
            "Quaternions Actual_y",
            "Quaternions Actual_z",
            "Quaternions Estimated_x",
            "Quaternions Estimated_y",
            "Quaternions Estimated_z",
            "Quaternions Reference_x",
            "Quaternions Reference_y",
            "Quaternions Reference_z",
            "Euler Angles Actual_x",
            "Euler Angles Actual_y",
            "Euler Angles Actual_z",
            "Euler Angles Estimated_x",
            "Euler Angles Estimated_y",
            "Euler Angles Estimated_z",
            "Euler Angles Reference_x",
            "Euler Angles Reference_y",
            "Euler Angles Reference_z",
            "Pointing Metric",
            "Pointing Metric To Estimate",
            "Estimation Metric",
            "P_k1",
            "P_k2",
            "P_k3",
            "P_k4",
            "P_k5",
            "P_k6",
            "P_k7",
            "K_k_max",
            "Earth_Error_x",
            "Earth_Error_y",
            "Earth_Error_z",
            "Sun_Error_x",
            "Sun_Error_y",
            "Sun_Error_z",
            "Magnetometer_Error_x",
            "Magnetometer_Error_y",
            "Magnetometer_Error_z"
            ]
        ####################################################
        #  THE ORBIT_DATA DICTIONARY IS USED TO STORE ALL  #
        #     THE MEASUREMENTS FOR EACH TIMESTEP (TS)      #
        # EACH ORBIT HAS AN INDUCED FAULT WITHIN THE ADCS. #
        ####################################################

        self.Orbit_Data = {
            "Sun": np.zeros(3),            #S_o measurement (vector of sun in ORC)
            "Magnetometer": np.zeros(3),    #B vector in SBC
            "Earth": np.zeros(3),           #Satellite position vector in ORC
            "Angular momentum of wheels": np.zeros(3),    #Wheel angular velocity of each reaction wheel
            "Star": np.zeros(3),
            "Angular velocity of satellite actual": np.zeros(3),
            "Angular velocity of satellite estimated": np.zeros(3),
            "Angular velocity of satellite reference": np.zeros(3),
            "Moving Average": [],
            "Wheel Control Torques": np.zeros(3),
            "Magnetic Control Torques": np.zeros(3), 
            "Sun in view": [],                              #True or False values depending on whether the sun is in view of the satellite
            "Current fault": [],                            #What the fault is that the system is currently experiencing
            "Current fault numeric": [],
            "Current fault binary": [],
            "Wheel disturbance Torques": np.zeros(3),
            "Gravity Gradient Torques": np.zeros(3),
            "Gyroscopic Torques": np.zeros(3),
            "Aerodynamic Torques": np.zeros(3),
            "Predicted fault": [],
            "Isolation Accuracy": [],
            "Prediction Accuracy": [],
            "Quaternions Actual": np.zeros(3),
            "Quaternions Estimated": np.zeros(3),
            "Quaternions Reference": np.zeros(3),
            "Euler Angles Actual": np.zeros(3),
            "Euler Angles Estimated": np.zeros(3),
            "Euler Angles Reference": np.zeros(3),
            "Pointing Metric": [],
            "Estimation Metric": []
        }

        #! Fourth change, ignore Quaternion error
        # ,
        #             "Quaternion magnetitude error": []

        zero3 = np.zeros(3)
        #* Create dictionary of all the sensors

        #! First change
        self.sensor_vectors = {
        "Magnetometer": {"Noise SBC": zero3, "True SBC": zero3, "True ORC": zero3, "Noise ORC": zero3, "Estimated SBC": zero3, "noise": SET_PARAMS.Magnetometer_noise}, 
        "Sun_Sensor": {"Noise SBC": zero3,"True SBC": zero3, "True ORC": zero3, "Noise ORC": zero3, "Estimated SBC": zero3, "noise": SET_PARAMS.Fine_sun_noise},
        "Earth_Sensor": {"Noise SBC": zero3,"True SBC": zero3, "True ORC": zero3, "Noise ORC": zero3, "Estimated SBC": zero3, "noise": SET_PARAMS.Earth_noise}, 
        "Star_tracker": {"Noise SBC": zero3,"True SBC": zero3, "True ORC": zero3, "Noise ORC": zero3, "Estimated SBC": zero3, "noise": SET_PARAMS.star_tracker_noise}
        }

        self.zeros = np.zeros((SET_PARAMS.number_of_faults,), dtype = int)

        self.fault = "None"                      # Current fault of the system
        self.predictedFailedSensor = "None"

        #! Just for testing kalman filter
        self.est_q_error = 0
        self.est_w_error = 0

    def update(self):
        self.Orbit_Data["Magnetometer"] = self.B_sbc_meas
        self.Orbit_Data["Sun"] = self.S_sbc
        self.Orbit_Data["Earth"] = self.r_sat_sbc
        self.Orbit_Data["Star"] = self.star_tracker_sbc
        self.Orbit_Data["Angular momentum of wheels"] = self.angular_wheel_momentum_with_noise
        self.Orbit_Data["Angular velocity of satellite actual"] = self.w_bo
        self.Orbit_Data["Angular velocity of satellite estimated"] = self.w_bo_est
        self.Orbit_Data["Angular velocity of satellite reference"] = self.w_bo_ref
        self.Orbit_Data["Sun in view"] = self.sun_in_view
        self.Orbit_Data["Wheel Control Torques"] = self.Nw
        self.Orbit_Data["Wheel disturbance Torques"] = self.Nrw
        self.Orbit_Data["Gravity Gradient Torques"] = self.Ngg
        self.Orbit_Data["Gyroscopic Torques"] = self.Ngyro
        self.Orbit_Data["Magnetic Control Torques"] = self.Nm
        self.Orbit_Data["Aerodynamic Torques"] = self.Naero
        #! self.Orbit_Data["Quaternion magnetitude error"] = np.sum(np.abs(self.control.q_error))

        # Get the measurement difference between the actual quaternions, the reference and the estimated
        A_ORC_to_SBC_ref = Transformation_matrix(self.q_ref)

        RandomVector = np.array([0,0.5,0.5])/(np.sqrt(0.5**2 + 0.5**2))

        VectorRef = A_ORC_to_SBC_ref @ RandomVector

        VectorActual = self.A_ORC_to_SBC @ RandomVector

        VectorEstimated = self.A_ORC_to_SBC_est @ RandomVector

        referenceDifferenceAngle = Quaternion_functions.rad2deg(np.arccos(np.clip(np.dot(VectorActual, VectorRef),-1,1)))

        controlToEstimateDifferenceAngle = Quaternion_functions.rad2deg(np.arccos(np.clip(np.dot(VectorEstimated, VectorRef),-1,1)))

        estimatedDifferenceAngle = Quaternion_functions.rad2deg(np.arccos(np.clip(np.dot(VectorActual, VectorEstimated),-1,1)))

        eulerAngleActual = np.array(getEulerAngles(self.q))
        eulerAngleReference = np.array(getEulerAngles(self.q_ref))
        eulerAngleEstimated = np.array(getEulerAngles(self.q_est))

        self.Orbit_Data["Pointing Metric"] = referenceDifferenceAngle
        self.Orbit_Data["Estimation Metric"] = estimatedDifferenceAngle
        self.Orbit_Data["Euler Angles Actual"] = eulerAngleActual
        self.Orbit_Data["Euler Angles Estimated"] = eulerAngleEstimated
        self.Orbit_Data["Euler Angles Reference"] = eulerAngleReference
        self.Orbit_Data["Quaternions Actual"] = self.q[:3]
        self.Orbit_Data["Quaternions Estimated"] = self.q_est[:3]
        self.Orbit_Data["Quaternions Reference"] = self.q_ref[:3]
        # Predict the sensor parameters and add them to the Orbit_Data

        self.Orbit_Data["Moving Average"] = self.MovingAverage

        self.Orbit_Data["Predicted fault"] = self.predictedFailure

        #* Test this for new fault identification
        #* This is because the anomaly is not just when reflection occurs
        #* But also when the satellite estimation is incorrect

        #! if estimatedDifferenceAngle > 6:
        #!    fault = self.fault

        if not self.reflection and self.fault == "Reflection":
            fault = "None"
        elif not self.SunSeenBySensor and self.fault in SET_PARAMS.SunFailures:
            fault = "None"
        elif not self.earthSeenBySensor and self.fault in SET_PARAMS.EarthFailures:
            fault = "None"
        else:
            fault = self.fault
        
        self.Orbit_Data["Current fault"] = fault
        temp = list(self.zeros)
        temp[Fault_names_to_num[fault] - 1] = 1
        self.Orbit_Data["Current fault numeric"] = temp
        self.Orbit_Data["Current fault binary"] = 0 if fault == "None" else 1

        if self.predictedFailure == self.Orbit_Data["Current fault binary"]:
            self.Orbit_Data["Prediction Accuracy"] = 1
        else:
            self.Orbit_Data["Prediction Accuracy"] = 0

        if fault == "None":
            FailedSensor = "None"

        elif fault in SET_PARAMS.SunFailures:
            FailedSensor = "Sun"
        
        elif fault in SET_PARAMS.EarthFailures:
            FailedSensor = "Earth"

        elif fault in SET_PARAMS.starTrackerFailures:
            FailedSensor = "Star"

        elif fault in SET_PARAMS.magnetometerFailures:
            FailedSensor = "Magnetometer"

        self.Orbit_Data["Isolation Accuracy"] = 1 if FailedSensor == self.predictedFailedSensor else 0

        Earth_Error = self.sensor_vectors["Earth_Sensor"]["True SBC"] - self.sensor_vectors["Earth_Sensor"]["Estimated SBC"]

        Sun_Error = self.sensor_vectors["Sun_Sensor"]["True SBC"] - self.sensor_vectors["Sun_Sensor"]["Estimated SBC"]

        Magnetometer_Error = self.sensor_vectors["Magnetometer"]["True SBC"] - self.sensor_vectors["Magnetometer"]["Estimated SBC"]

        modelledSun = self.A_ORC_to_SBC_est @ self.sensor_vectors["Sun_Sensor"]["True ORC"]

        modelledMagnetometer = self.A_ORC_to_SBC @ self.sensor_vectors["Magnetometer"]["True ORC"]

        modelledEarth = self.A_ORC_to_SBC @ self.sensor_vectors["Earth_Sensor"]["True ORC"]

        modelledStar = self.A_ORC_to_SBC @ self.sensor_vectors["Star_tracker"]["True ORC"]

        self.globalArray = [self.S_sbc_meas[0],
            self.S_sbc_meas[1],
            self.S_sbc_meas[2],
            modelledSun[0],
            modelledSun[1],
            modelledSun[2],
            self.B_sbc_meas[0],
            self.B_sbc_meas[1],
            self.B_sbc_meas[2],
            modelledMagnetometer[0],
            modelledMagnetometer[1],
            modelledMagnetometer[2],
            self.r_sat_sbc_meas[0],
            self.r_sat_sbc_meas[1],
            self.r_sat_sbc_meas[2],
            modelledEarth[0],
            modelledEarth[1],
            modelledEarth[2],
            self.angular_momentum_wheels_with_noise[0],
            self.angular_momentum_wheels_with_noise[1],
            self.angular_momentum_wheels_with_noise[2],
            self.star_tracker_sbc[0],
            self.star_tracker_sbc[1],
            self.star_tracker_sbc[2],
            modelledStar[0],
            modelledStar[1],
            modelledStar[2],
            self.w_bo[0],
            self.w_bo[1],
            self.w_bo[2],
            self.w_bo_est[0],
            self.w_bo_est[1],
            self.w_bo_est[2],
            self.w_bo_ref[0],
            self.w_bo_ref[1],
            self.w_bo_ref[2],
            self.MovingAverage,
            self.Nw[0],
            self.Nw[1],
            self.Nw[2],
            self.Nm[0],
            self.Nm[1],
            self.Nm[2],
            self.sun_in_view,                              #True or False values depending on whether the sun is in view of the satellite
            fault,                            #What the fault is that the system is currently experiencing
            temp,
            self.Orbit_Data["Current fault binary"],
            self.Nrw[0],
            self.Nrw[1],
            self.Nrw[2],
            self.Ngg[0],
            self.Ngg[1],
            self.Ngg[2],
            self.Ngyro[0],
            self.Ngyro[1],
            self.Ngyro[2],
            self.Naero[0],
            self.Naero[1],
            self.Naero[2],
            self.predictedFailure,
            self.Orbit_Data["Isolation Accuracy"],
            self.Orbit_Data["Prediction Accuracy"],
            self.q[:3][0],
            self.q[:3][1],
            self.q[:3][2],
            self.q_est[:3][0],
            self.q_est[:3][1],
            self.q_est[:3][2],
            self.q_ref[:3][0],
            self.q_ref[:3][1],
            self.q_ref[:3][2],
            eulerAngleActual[0],
            eulerAngleActual[1],
            eulerAngleActual[2],
            eulerAngleEstimated[0],
            eulerAngleEstimated[1],
            eulerAngleEstimated[2],
            eulerAngleReference[0],
            eulerAngleReference[1],
            eulerAngleReference[2],
            referenceDifferenceAngle,
            controlToEstimateDifferenceAngle,
            estimatedDifferenceAngle,
            self.P_k_est[0,0],
            self.P_k_est[1,1],
            self.P_k_est[2,2],
            self.P_k_est[3,3],
            self.P_k_est[4,4],
            self.P_k_est[5,5],
            self.P_k_est[6,6],
            np.max(self.K_k_est),
            Earth_Error[0],
            Earth_Error[1],
            Earth_Error[2],
            Sun_Error[0],
            Sun_Error[1],
            Sun_Error[2],
            Magnetometer_Error[0],
            Magnetometer_Error[1],
            Magnetometer_Error[2]
            ]


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
        self.angular_momentum_wheels = SET_PARAMS.initial_angular_wheels # Angular momentum of satellite wheels
        self.angular_momentum_wheels_with_noise = SET_PARAMS.initial_angular_wheels 
        self.q = SET_PARAMS.quaternion_initial      # Quaternion position
        self.q_est = self.q
        self.t = SET_PARAMS.time                    # Beginning time
        self.dt = SET_PARAMS.Ts                     # Time step
        self.dh = self.dt/SET_PARAMS.NumberOfIntegrationSteps                        # Size of increments for Runga-kutta method
        self.Ix = SET_PARAMS.Ix                     # Ixx inertia
        self.Iy = SET_PARAMS.Iy                     # Iyy inertia
        self.Iz = SET_PARAMS.Iz                     # Izz inertia
        self.Inertia = np.identity(3)*np.array(([self.Ix, self.Iy, self.Iz]))
        self.Inertia_Inverse = np.linalg.inv(self.Inertia)
        self.Iw = SET_PARAMS.Iw                     # Inertia of a reaction wheel
        self.angular_momentum = SET_PARAMS.initial_angular_wheels # Angular momentum of satellite
        self.faster_than_control = SET_PARAMS.faster_than_control   # If it is required that satellite must move faster around the earth than Ts
        self.control = Controller.Control()         # Controller.py is used for control of satellite    
        self.star_tracker_ORC = SET_PARAMS.star_tracker_ORC
        self.sun_noise = SET_PARAMS.Fine_sun_noise
        self.RKF = RKF()                            # Rate Kalman_filter
        self.EKF = EKF()                            # Extended Kalman_filter
        self.sensors_kalman = SET_PARAMS.kalmanSensors #"Earth_Sensor", "Sun_Sensor", "Star_tracker"
        super().initiate_fault_parameters()
        self.SensePredDMDDict = {}

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