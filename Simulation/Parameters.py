import math
import numpy as np
from sgp4.api import jday
from struct import *
from scipy import special
import pathlib
from Simulation.utilities import Reflection, Intersection, PointWithinParallelLines, lineEquation, line2Equation, NormalizeVector

pi = math.pi

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

global faultNames

class SET_PARAMS:
    # All the parameters specific to the satellite and the mission
    
    ####################
    # ORBIT PARAMETERS #
    ####################
    
    eccentricity = 0.000092             # Update eccentricity list
    inclination = 97.4                  # degrees
    Semi_major_axis = 6879.55           # km The distance from the satellite to the earth + the earth radius
    Height_above_earth_surface = 500e3  # distance above earth surface
    Scale_height = 8500                 # scale height of earth atmosphere
    RAAN = 275*pi/180                   # Right ascension of the ascending node in radians
    AP = 0                              # argument of perigee
    Re = 6371.2                         # km magnetic reference radius
    Mean_motion = 15.2355000000         # rev/day
    Mean_motion_per_second = Mean_motion/(3600.0*24.0)
    Mean_anomaly = 29.3                 # degrees
    Argument_of_perigee = 57.4          # in degrees
    omega = Argument_of_perigee
    Period = 86400/Mean_motion          # seconds
    J_t,fr = jday(2020,3,16,15,30,0)    # current julian date
    epoch = J_t - 2433281.5 + fr
    Drag_term = 0.000194                # Remember to update the list term
    wo = Mean_motion_per_second*(2*pi)  # rad/s

    ############
    # TLE DATA #
    ############
        
    # s list
    satellite_number_list = '1 25544U'
    international_list = ' 98067A   '
    epoch_list = str("{:.8f}".format(epoch))
    mean_motion_derivative_first_list = '  .00001764'
    mean_motion_derivative_second_list = '  00000-0'
    Drag_term_list = '  19400-4' # B-star
    Ephereris_list = ' 0'
    element_num_checksum_list = '  7030'
    s_list = satellite_number_list + international_list + epoch_list + mean_motion_derivative_first_list + mean_motion_derivative_second_list + Drag_term_list + Ephereris_list + element_num_checksum_list
    # t list
    line_and_satellite_number_list = '2 27843  '
    inclination_list = str("{:.4f}".format(inclination))
    intermediate_list = ' '
    RAAN_list = str("{:.4f}".format(RAAN*180/pi))
    intermediate_list_2 = ' '
    eccentricity_list = '0000920  '
    perigree_list = str("{:.4f}".format(Argument_of_perigee))
    intermediate_list_3 = intermediate_list_2 + ' '
    mean_anomaly_list = str("{:.4f}".format(Mean_anomaly))
    intermediate_list_4 = intermediate_list_2
    mean_motion_list = str("{:8f}".format(Mean_motion)) + '00'
    Epoch_rev_list = '000009'
    t_list = line_and_satellite_number_list + inclination_list + intermediate_list + RAAN_list + intermediate_list_2 + eccentricity_list + perigree_list + intermediate_list_3 + mean_anomaly_list + intermediate_list_4 + mean_motion_list + Epoch_rev_list
    
    #######################################################
    # OVERWRITE JANSEN VUUREN SE WAARDES MET SGP4 EXAMPLE #
    #######################################################
    
    """
    s_list = '1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991'
    t_list = '2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482'
    """
    
    #######################
    # POSITION PARAMETERS #
    #######################
    
    a_G0 = 0        # Angle from the greenwhich
    
    ############################
    # ATMOSPHERE (AERODYNAMIC) #
    ############################
    
    normal_accommodation = 0.8
    tangential_accommodation = 0.8
    ratio_of_molecular_exit = 0.05
    offset_vector = np.array(([0.01,0.01,0.01]))
    unit_normal_vector = np.array([[0,1,0],[1,0,0],[0,0,1]])
    atmospheric_reference_density = 1.225
    
    ###############################
    # EARTH EFFECTS (GEOMAGNETIC) #
    ###############################
    
    k = 10 #order of expansion
    Radius_earth = 6371e3 # in m
    w_earth = 7.2921150e-5 #rad/s
    
    ##################
    # SUN PARAMETERS #
    ##################
    
    Radius_sun = 696340e3 # in m

    ##################
    # SATELLITE BODY #
    ##################

    Mass = 20 #kg
    Lx, Ly, Lz = 0.3, 0.3, 0.4
    Dimensions = np.array(([Lx, Ly, Lz])) # Lx, Ly, Lz
    Ix = 0.4 #kg.m^2
    Iy = 0.45 #kg.m^2
    Iz = 0.3 #kg.m^2
    Inertia = np.diag(([Ix, Iy, Iz]))
    Iw_single = 88.1e-6 #kgm^2 Inertia of the RW-06 wheel
    Iw = np.diag([Iw_single, Iw_single, Iw_single])
    Surface_area_i = np.array(([Dimensions[0] * Dimensions[1], 
                                Dimensions[1] * Dimensions[2], 
                                Dimensions[0] * Dimensions[2]]))

    
    kgx = 3 * wo**2 * (Iz - Iy)
    kgy = 3 * wo**2 * (Ix - Iz)
    kgz = 3 * wo**2 * (Iy - Ix)


    ##############################
    # SATELLITE INITIAL POSITION #
    ##############################
    
    quaternion_initial = np.array(([0, 0, 1, 0]), dtype = "float64") #Quaternion_functions.euler_to_quaternion(0,0,0) #roll, pitch, yaw
    A_ORC_to_SBC = Transformation_matrix(quaternion_initial)
    wbo = np.array(([0.0,0.0,0.0]))
    wbi = wbo + A_ORC_to_SBC @ np.array(([0,-wo,0]), dtype = "float64")
    initial_angular_wheels = np.zeros(3, dtype = "float64")
    
    ###############################
    # MAX PARAMETERS OF ACTUATERS #
    ###############################
    
    angularSatelliteMax = 2.0 #degrees per second (theta derived), angular velocity
    angularSatelliteMax = angularSatelliteMax * pi/180
    wheel_angular_d_d = 0.133 # degrees per second^2 (rotation speed derived), angular acceleration
    h_ws_max = 60e-3 # Nms
    wheel_angular_d_max = h_ws_max/Iw_single
    N_ws_max = 5e-3 # Nm
    M_magnetic_max = 25e-6 # Nm
    RW_sigma_x = 14.6
    RW_sigma_y = 8.8
    RW_sigma_z = 21.2
    RW_sigma = np.mean([RW_sigma_x, RW_sigma_y, RW_sigma_y])
    Rotation_max = 2.0 # degrees per second
    
    ######################
    # CONTROL PARAMETERS #
    ######################
    
    w_ref = np.zeros(3) # desired angular velocity of satellite
    quaternion_initial = quaternion_initial/np.linalg.norm(quaternion_initial) # initial position of satellite
    q_ref = quaternion_initial
    time = 1
    Ts = 1 # Time_step
    wn = 90

    #! Testing Kalman Filter
    best_ij = "-"
    best_error = 100

    # For no filter
    Kp = 0
    Kd = 0
    Kw = 0

    Kd_magnet = 1e-7
    Ks_magnet = 1e-7
    Kalman_filter_use = True

    ############################
    # KALMAN FILTER PARAMETERS #
    ############################
    Qw_t = np.diag([RW_sigma_x, RW_sigma_y, RW_sigma_z])
    Q_k = Ts*Qw_t
    #Q_k = np.diag([measurement_noise**2 + model_noise**2]*7)
    P_k = np.eye(7)/2

    measurement_noise = 0.001
    model_noise = 0.01

    R_k = np.array([[measurement_noise**2 + model_noise**2, 0, 0], 
                    [0, measurement_noise**2 + model_noise**2, 0], 
                    [0, 0, measurement_noise**2 + model_noise**2]])

    ######################
    # DISPLAY PARAMETERS #
    ######################
    
    faster_than_control = 1.0 # how much faster the satellite will move around the earth in simulation than the control
    Display = True # if display is desired or not
    skip = 20  # the number of iterations before display

    #######################################################################
    # NUMBER OF REPETITIONS FOR ORBITS AND HOW MANY ORBITS PER REPETITION #
    #######################################################################

    Number_of_orbits = 1 # * This value can constantly be changed as well as the number of orbits
    Number_of_multiple_orbits = 7
    numberOfSensors = 6
    availableData = ["Magnetometer", "Sun", "Earth", "Star", "Angular momentum of wheels"]
    availableSensors = {"Magnetometer": "Magnetometer", 
                        "Sun": "Sun_Sensor", 
                        "Earth": "Earth_Sensor",
                        "Star": "Star_tracker", 
                        "Angular momentum of wheels": "Angular momentum of wheels"}
    ##########################
    # VISUALIZE MEASUREMENTS #
    ##########################
    
    Visualize = True
    
    #######################
    # CSV FILE PARAMETERS #
    #######################
    
    save_as = ".xlsx"
    load_as = ".csv"
    
    ##################################
    # STORAGE OF DATA FOR PREDICTION #
    ##################################
    stateBufferLength = 20
    data_mode = "_buffer"
    buffer_mode = True
    buffer_size = 20

    ###############################
    # FAULT PREDICTION PARAMETERS #
    ###############################
    SensorFDIR = False
    Reflection = True
    Model_or_Measured = "Model"
    SensorPredictor = "DMD"
    FeatureExtraction = "DMD"
    SensorIsolator = "DMD"
    SensorRecoveror = "EKF"
    FeatureExtractionMethods = ["PCA", "DMD"]
    FaultPredictionMethods = ["PhysicsEnabledDMDDecisionTree", "ANN", "RandomForest"]
    Low_Aerodynamic_Disturbance = False
    
    # DMD fault parameters for moving average
    MovingAverageSizeOfBuffer = 20
    sensor_number = "ALL"

    # File names for the storage of the data attained during the simulation
    path = "/".join(str(pathlib.Path(__file__).parent.resolve()).split("/")[:-1]) + "/Data files/"
    filename = "Faults" + data_mode

    # Path to hyperparameters
    pathHyperParameters = "/".join(str(pathlib.Path(__file__).parent.resolve()).split("/")[:-1]) + "/Hyperparameters/"
    UsePredeterminedPositionalData = False

    #####################
    # MODE OF OPERATION #
    #####################

    Mode = "Nominal"  
    
    ####################################
    # FAULT TYPES AND FAULT PARAMETERS #
    ####################################

    global faultNames


    faultNames = ["None",
                "Reflection"]
                # ,
                # "Earth_sensor_high_noise",
                # "Magnetometer_sensor_high_noise", 
                # "Erroneous_sun",
                # "Closed_shutter",
                # "Interference_magnetic",
                # "Stop_magnetometers"]

    SunFailures = [
        "Catastrophic_sun",
        "Erroneous_sun",
        "Reflection"
    ]

    EarthFailures = ["Earth_sensor_high_noise"]

    starTrackerFailures = ["Closed_shutter"]

    magnetometerFailures = ["Magnetometer_sensor_high_noise", "Interference_magnetic", "Stop_magnetometers"]

    # faultNames = ["None", 
    # "Electronics_of_RW", 
    # "Overheated_RW", 
    # "Catastrophic_RW", 
    # "Angular_sensor_high_noise", 
    # "Earth_sensor_high_noise",
    # "Magnetometer_sensor_high_noise",
    # "Catastrophic_sun", 
    # "Erroneous_sun",
    # "Closed_shutter",
    # "Inverted_polarities_magnetorquers",
    # "Interference_magnetic",
    # "Stop_magnetometers",
    # "Increasing_angular_RW_momentum",
    # "Decreasing_angular_RW_momentum",
    # "Oscillating_angular_RW_momentum"]
    faultnames = faultNames.copy()

    Fault_names = {faultNames[i]: i+1 for i in range(len(faultNames))}

    number_of_faults = len(faultNames)

    visualizeKalman = ["w_est","w_act","q_est","q","q_ref",
                    "w_ref","q_error","w_error"]
    measurementUpdateVars = []

    likelyhood_multiplier = 1
    #Fault_simulation_mode = 1 # Continued failure, a mistake that does not go back to normal
    #Fault_simulation_mode = 0 # Failure is based on specified class failure rate. Multiple failures can occure simultaneously
    Fault_simulation_mode = 2 # A single fault occurs per orbit
    fixed_orbit_failure = 2

    
    #####################################################################################
    # FOR THE FAULT SIMULATION MODE 2, THE FAULT NAMES MUST BE THE VALUES BASED ON KEYS #
    #####################################################################################
    
    Fault_names_values = {value:key for key, value in Fault_names.items()}


    #################
    # SENSOR MODELS #
    #################
    # Star tracker
    star_tracker_vector = np.array([1.0,1.0,1.0])/np.linalg.norm(np.array([1.0,1.0,1.0]))
    star_tracker_noise = 1e-4

    # Magnetometer
    Magnetometer_noise = 7e-2         #standard deviation of magnetometer noise in Tesla

    # Earth sensor
    Earth_sensor_position = np.array(([0, 0, -Lz/2])) # x, y, en z
    Earth_sensor_FOV = 360 # Field of view in degrees #! change this back to 180
    Earth_sensor_angle = Earth_sensor_FOV/2 # The angle use to check whether the dot product angle is within the field of view
    Earth_noise = 7e-3                  #standard deviation away from where the actual earth is

    # Fine Sun sensor
    Fine_sun_sensor_position = np.array(([Lx/2, 0, 0])) # x, y, en z 
    Fine_sun_sensor_FOV = 180 # Field of view in degrees
    Fine_sun_sensor_angle = Fine_sun_sensor_FOV/2 # The angle use to check whether the dot product angle is within the field of view
    Fine_sun_noise = 1e-3                   #standard deviation away from where the actual sun is
    # Define sun sensor dimensions
    Sun_sensor_length = 0.028
    Sun_sensor_width = 0.023
    SSF_LeftCorner = np.array((Fine_sun_sensor_position[0], Fine_sun_sensor_position[1] - Sun_sensor_width/2, Fine_sun_sensor_position[2] - Sun_sensor_length/2))
    SSF_RightCorner = np.array((Fine_sun_sensor_position[0], Fine_sun_sensor_position[1] + Sun_sensor_width/2, Fine_sun_sensor_position[2] - Sun_sensor_length/2))
    
    SSF_Plane = [1/2, 0, 0, Lx/2] # x, y, z, d
    # Coarse Sun Sensor
    Coarse_sun_sensor_position = np.array(([-Lx/2, 0, 0])) # x, y, en z 
    Coarse_sun_sensor_FOV = 180 # Field of view in degrees
    Coarse_sun_sensor_angle = Coarse_sun_sensor_FOV/2 # The angle use to check whether the dot product angle is within the field of view
    Coarse_sun_noise = 3e-3 #standard deviation away from where the actual sun is

    SSC_LeftCorner = np.array(([Coarse_sun_sensor_position[0], Coarse_sun_sensor_position[1] - Sun_sensor_width/2, Coarse_sun_sensor_position[2] - Sun_sensor_length/2]))
    SSC_RightCorner = np.array(([Coarse_sun_sensor_position[0], Coarse_sun_sensor_position[1] + Sun_sensor_width/2, Coarse_sun_sensor_position[2] - Sun_sensor_length/2]))
    
    SSC_Plane = [-1/2, 0, 0, Lx/2] # x, y, z, d
    # Angular Momentum sensor
    Angular_sensor_noise = 1e-3
    
    ###################
    # HARDWARE MODELS #
    ###################
    SP_Length = Lx
    SP_width = Ly
    # Number of solar Panels = 4, only 2 accounted for with respect to reflection
    SolarPanelPosition = np.array(([0, 0, -Lz/2]))
    SPF_position = np.array(([Lx/2 + SP_Length/2, 0, -Lz/2]))    # Middle point, x, y en z
    SPC_position = np.array(([-Lx/2 - SP_Length/2, 0, -Lz/2]))
    SPC_normal_vector = np.array(([0, 0, 1]))
    SPF_normal_vector = np.array(([0, 0, 1]))
    

    # Fine solar panel
    SPF_LeftTopCorner = np.array(([SPF_position[0] + SP_Length/2, SPF_position[1] + SP_width/2, SPF_position[2]]))
    SPF_RightTopCorner = np.array(([SPF_position[0] + SP_Length/2, SPF_position[1] - SP_width/2, SPF_position[2]]))
    SPF_LeftBottomCorner = np.array(([SPF_position[0] - SP_Length/2, SPF_position[1] + SP_width/2, SPF_position[2]]))
    SPF_RightBottomCorner = np.array(([SPF_position[0] - SP_Length/2, SPF_position[1] - SP_width/2, SPF_position[2]]))


    # Coarse Solar Panel
    SPC_LeftTopCorner = np.array(([SPC_position[0] - SP_Length/2, SPC_position[1] + SP_width/2, SPC_position[2]]))
    SPC_RightTopCorner = np.array(([SPC_position[0] - SP_Length/2, SPC_position[1] - SP_width/2, SPC_position[2]]))
    SPC_LeftBottomCorner = np.array(([SPC_position[0] + SP_Length/2, SPC_position[1] + SP_width/2, SPC_position[2]]))
    SPC_RightBottomCorner = np.array(([SPC_position[0] + SP_Length/2, SPC_position[1] - SP_width/2, SPC_position[2]]))


    # Fix sun parameters (for approximation purposes)
    radiusOfSun = 696340e3
    distanceToSun = 151.21e9 + radiusOfSun
    angleOfSun = np.arctan(radiusOfSun/distanceToSun)

    ############################
    # CONSTELLATION PARAMETERS #
    ############################
    FD_strategy = "Distributed"
    Constellation = False
    Number_of_satellites = 1
    k_nearest_satellites = 5


    NumberOfRandom = 1
    NumberOfFailuresReset = 10

    #######
    surfaceI = {
        'z_positive': {'Area': Lx * Ly, 'CoM-CoP': np.array([0,0,Lz/2]), 'NormalVector': np.array([0,0,1])},
        'z-negative': {'Area': Lx * Ly, 'CoM-CoP': np.array([0,0,-Lz/2]), 'NormalVector': np.array([0,0,-1])},
        'y-positive': {'Area': Lz * Lx, 'CoM-CoP': np.array([0,Ly/2,0]), 'NormalVector': np.array([0,1,0])},
        'y-negative': {'Area': Lz * Lx, 'CoM-CoP': np.array([0,-Ly/2,0]), 'NormalVector': np.array([0,-1,0])},
        'x-positive': {'Area': Lz * Ly, 'CoM-CoP': np.array([Lx/2,0,0]), 'NormalVector': np.array([1,0,0])},
        'x-negative': {'Area': Lz * Ly, 'CoM-CoP': np.array([-Lx/2,0,0]), 'NormalVector': np.array([-1,0,0])},
        'SolarPanelxpyp': {'Area': SP_Length * SP_width, 'CoM-CoP': np.array([Lx/2 + SP_Length/2, Ly/2 + SP_width/2, Lz/2]), 'NormalVector': np.array([0,0,1])},
        'SolarPanelxpyn': {'Area': SP_Length * SP_width, 'CoM-CoP': np.array([Lx/2 + SP_Length/2, -(Ly/2 + SP_width/2), Lz/2]), 'NormalVector': np.array([0,0,1])},
        'SolarPanelxnyp': {'Area': SP_Length * SP_width, 'CoM-CoP': np.array([-(Lx/2 + SP_Length/2), Ly/2 + SP_width/2, Lz/2]), 'NormalVector': np.array([0,0,1])},
        'SolarPanelxnyn': {'Area': SP_Length * SP_width, 'CoM-CoP': np.array([-(Lx/2 + SP_Length/2), -(Ly/2 + SP_width/2), Lz/2]), 'NormalVector': np.array([0,0,1])},
        'SolarPanelxpyp2': {'Area': SP_Length * SP_width, 'CoM-CoP': np.array([Lx/2 + SP_Length/2, Ly/2 + SP_width/2, Lz/2]), 'NormalVector': np.array([0,0,-1])},
        'SolarPanelxpyn2': {'Area': SP_Length * SP_width, 'CoM-CoP': np.array([Lx/2 + SP_Length/2, -(Ly/2 + SP_width/2), Lz/2]), 'NormalVector': np.array([0,0,-1])},
        'SolarPanelxnyp2': {'Area': SP_Length * SP_width, 'CoM-CoP': np.array([-(Lx/2 + SP_Length/2), Ly/2 + SP_width/2, Lz/2]), 'NormalVector': np.array([0,0,-1])},
        'SolarPanelxnyn2': {'Area': SP_Length * SP_width, 'CoM-CoP': np.array([-(Lx/2 + SP_Length/2), -(Ly/2 + SP_width/2), Lz/2]), 'NormalVector': np.array([0,0,-1])}
    }

    #################################################################################################################

    NumberOfIntegrationSteps = 10

    process_noise = 1e-4

    kalmanSensors = ["Magnetometer", "Earth_Sensor", "Sun_Sensor", "Star_tracker"]

    treeDepth = 10


Min_high_noise = 5.0
Max_high_noise = 10.0

Min_high_speed_percentage = 0.9
Max_high_speed_percentage = 1.0

min_inteference = 3.0
max_Interference_magnetic = 5.0

Min_low_speed_percentage = 0.0
Max_low_speed_percentage = 0.1

def bitflip(x,pos):
    fs = pack('f',x)
    bval = list(unpack('BBBB',fs))
    [q,r] = divmod(pos,8)
    bval[q] ^= 1 << r
    fs = pack('BBBB', *bval)
    fnew=unpack('f',fs)
    return fnew[0]

def random_size(minimum, maximum):
    return np.clip(np.random.normal((minimum+maximum)/2,(maximum-minimum)/2),minimum,maximum)

def random_bit_flip(input_var):
    position = np.random.randint(0, 32)
    input_var = bitflip(input_var, position)
    return input_var

def Reliability(t,n,B):
    return np.exp(-(t / n)**B)

def weibull(t,n,B):
    return (B / n) * (t / n)**(B - 1) * np.exp(-(t / n)**B)

class Fault_parameters:
    def __init__(self, Fault_per_hour = 0.1, number_of_failures = 0, failures = 0, seed = 0):
        self.np_random = np.random
        self.np_random.seed(seed)
        self.Fault_rate_per_hour = Fault_per_hour
        self.failure = "None"
        self.failures = failures
        self.number_of_failures = number_of_failures
        self.Fault_rate_per_second = self.Fault_rate_per_hour/3600
        self.n = 1/(self.Fault_rate_per_second)
        self.Beta = 0.4287
        self.time = int(SET_PARAMS.Number_of_orbits*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts))
        gamma = special.gammainc(1/self.Beta, (self.time/self.n)**self.Beta)
        self.Reliability_area = self.n * ((self.time/self.n)*gamma)/(self.Beta*((self.time/self.n)**self.Beta)**(1/self.Beta))
        self.Reliability_area_per_time_step = 0
        self.first = 1

    def Failure_Reliability_area(self, t):
        self.Reliability_area_per_time_step += weibull(t, self.n, self.Beta)*SET_PARAMS.Ts
        Failed = True if self.np_random.uniform(0,1) < self.Reliability_area_per_time_step/self.Reliability_area else False
        if Failed:
            ind = self.np_random.randint(0,self.number_of_failures) 
            self.failure = self.failures[ind]

        return self.failure

    def Failure(self,t):
        mean = weibull(t, self.n, self.Beta)
        self.np_random.normal(mean, 1)

    def random_(self):
        return self.np_random.normal(self.weibull_mean, self.weibull_std) 

    def normal_noise(self, sensor, noise):
        sensor = sensor.copy()
 
        #! sensor[0] += np.random.normal(0,abs(sensor[0]*noise))
        #! sensor[1] += np.random.normal(0,abs(sensor[1]*noise))
        #! sensor[2] += np.random.normal(0,abs(sensor[2]*noise))

        noiseDistribution = np.random.normal(0, noise, 3)

        sensor = sensor + noiseDistribution

        return sensor

class Reaction_wheels(Fault_parameters):
    def __init__(self, seed):
        self.angular_wheels = SET_PARAMS.wheel_angular_d_max
        self.angular_wheels_max = SET_PARAMS.N_ws_max*random_size(minimum = Min_high_speed_percentage, maximum = Max_high_speed_percentage)
        self.angular_wheels_min = SET_PARAMS.N_ws_max*random_size(minimum = Min_low_speed_percentage, maximum = Max_low_speed_percentage)
        self.Fault_rate_per_hour = 2.5e-7 * SET_PARAMS.likelyhood_multiplier
        self.number_of_failures = 3
        self.failures = {
            0: "Electronics_of_RW",
            1: "Overheated_RW",
            2: "Catastrophic_RW"
        }
        super().__init__(self.Fault_rate_per_hour, self.number_of_failures, self.failures, seed)
        self.number = self.np_random.randint(1,4)
        self.number_of_failed_wheels = []
        tries = 0
        while tries < self.number:
            current = self.np_random.randint(0,3)
            if not current in self.number_of_failed_wheels:
                self.number_of_failed_wheels.append(current)
                tries += 1

        self.number_of_failed_wheels = sorted(self.number_of_failed_wheels)
        self.angular_failed_wheel = np.zeros(3)


    def Electronics_of_RW_failure(self, angular_wheels):
        if self.first:
            self.angular_failed_wheel = angular_wheels[self.number_of_failed_wheels]
            self.first = 0
        self.angular_failed_wheel = np.maximum((self.angular_failed_wheel - abs(self.angular_failed_wheel)/10), self.angular_wheels_min*np.ones(self.number)) if self.failure == "Electronics_of_RW" else self.angular_failed_wheel
        angular_wheels[self.number_of_failed_wheels] = self.angular_failed_wheel
        return angular_wheels

    def Overheated_RW(self, angular_wheels):
        if self.first:
            self.angular_failed_wheel = angular_wheels[self.number_of_failed_wheels]
            self.first = 0
        self.angular_failed_wheel = np.maximum((self.angular_failed_wheel - abs(self.angular_failed_wheel)/10), self.angular_wheels_min*np.ones(self.number)) if self.failure == "Overheated_RW" else self.angular_failed_wheel
        angular_wheels[self.number_of_failed_wheels] = self.angular_failed_wheel
        return angular_wheels

    def Catastrophic_RW(self, angular_wheels):
        if self.first:
            self.angular_failed_wheel = angular_wheels[self.number_of_failed_wheels]
            self.first = 0
        self.angular_failed_wheel = np.zeros(self.number) if self.failure == "Catastrophic_RW" else self.angular_failed_wheel
        angular_wheels[self.number_of_failed_wheels] = self.angular_failed_wheel
        return angular_wheels

class Sun_sensor(Fault_parameters):
    def __init__(self, seed):
        self.Fault_rate_per_hour = 8.15e-9 * SET_PARAMS.likelyhood_multiplier
        self.number_of_failures = 3
        self.failures = {
            0: "Catastrophic_sun",
            1: "Erroneous_sun",
            2: "Reflection"
        }
        super().__init__(self.Fault_rate_per_hour, self.number_of_failures, self.failures, seed)
        self.sensors = {
            0: "Fine",
            1: "Coarse"
        }
        self.Failed_sensor = self.sensors[self.np_random.randint(0,1)]

    def Catastrophic_sun(self, sun_sensor, sensor_type):
        if sensor_type == self.Failed_sensor:
            if self.failure == "Catastrophic_sun":
                return np.zeros(sun_sensor.shape)
            else:
                return sun_sensor
        else:
            return sun_sensor

    def Erroneous_sun(self, sun_sensor, sensor_type):
        # Sun_sensor must be provided as a unit vector
        if sensor_type == self.Failed_sensor:
            return self.np_random.uniform(-1,1,sun_sensor.shape) if self.failure == "Erroneous_sun" else sun_sensor
        else:
            return sun_sensor

    def Reflection_sun(self, sun_sensor, S_ORC, sensor_type):
        reflection = False
        if self.failure == "Reflection":
            S_sbc = sun_sensor
            if sensor_type == "Fine":
                reflectedSunVector = Reflection(S_sbc, SET_PARAMS.SPF_normal_vector)

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
                        S_ORC = reflectedSunVector
                else:
                    S_ORC = reflectedSunVector


                S_ORC = NormalizeVector(S_ORC)
            
            else:
                reflectedSunVector = Reflection(S_sbc, SET_PARAMS.SPC_normal_vector)

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
                        S_ORC = reflectedSunVector
                else:
                    S_ORC = reflectedSunVector

                S_ORC = NormalizeVector(S_ORC)

        return S_ORC, reflection

class Magnetorquers(Fault_parameters):
    def __init__(self, seed):
        self.Fault_rate_per_hour = 8.15e-9 * SET_PARAMS.likelyhood_multiplier
        self.number_of_failures = 2
        self.failures = {
            0: "Inverted_polarities_magnetorquers",
            1: "Interference_magnetic"
        }
        super().__init__(self.Fault_rate_per_hour, self.number_of_failures, self.failures, seed)
        self.direction_magnetorquer_failed = self.np_random.randint(0,3)
    
    def Inverted_polarities_magnetorquers(self, magnetic_torquers):
        #############################################################
        # INVERTED POLARITIES MEAND THAT THE MAGNETIC TORQUERS WILL #
        #  MOVE IN THE OPPOSITIE DIRECTION (THUS MULTIPLIED BY -1)  #
        #############################################################
        self.magnetic_torquers = magnetic_torquers
        magnetic_torquers = magnetic_torquers[self.direction_magnetorquer_failed]
        self.magnetic_torquers[self.direction_magnetorquer_failed] = -magnetic_torquers if self.failure == "Inverted_polarities_magnetorquers" else magnetic_torquers
        return self.magnetic_torquers

    def Interference_magnetic(self, Magnetorquers):
        self.magnetic_torquers = Magnetorquers
        magnetic_torquers = Magnetorquers[self.direction_magnetorquer_failed]
        self.magnetic_torquers[self.direction_magnetorquer_failed] = Magnetorquers*random_size(min_inteference, max_Interference_magnetic) if self.failure == "Interference_magnetic" else magnetic_torquers
        return self.magnetic_torquers

class Magnetometers(Fault_parameters):
    def __init__(self, seed):
        self.Fault_rate_per_hour = 8.15e-9 * SET_PARAMS.likelyhood_multiplier
        self.number_of_failures = 3
        self.failures = {
            0: "Stop_magnetometers",
            1: "Interference_magnetic",
            2: "Magnetometer_sensor_high_noise"
        }
        super().__init__(self.Fault_rate_per_hour, self.number_of_failures, self.failures, seed)

    def Stop_magnetometers(self, magnetometer):
        # All of the magnetometers are zero
        if self.failure == "Stop_magnetometers":
            magnetometer = np.zeros(3)

        return magnetometer

    def Interference_magnetic(self, magnetometers):
        self.magnetometers = magnetometers*random_size(min_inteference, max_Interference_magnetic) if self.failure == "Interference_magnetic" else magnetometers
        return self.magnetometers
    
    def Magnetometer_sensor_high_noise(self, sensor):
        return sensor*random_size(minimum = Min_high_noise, maximum = Max_high_noise) if self.failure == "Magnetometer_sensor_high_noise" else sensor

class Earth_Sensor(Fault_parameters):
    def __init__(self, seed):
        self.Fault_rate_per_hour = 8.15e-9 * SET_PARAMS.likelyhood_multiplier
        self.number_of_failures = 1
        self.failures = {
            0: "Earth_sensor_high_noise"
        }
        super().__init__(self.Fault_rate_per_hour, self.number_of_failures, self.failures, seed)

    def Earth_sensor_high_noise(self, sensor):
        return sensor*random_size(minimum = Min_high_noise, maximum = Max_high_noise) if self.failure == "Earth_sensor_high_noise" else sensor

class Angular_Sensor(Fault_parameters):
    def __init__(self, seed):
        self.Fault_rate_per_hour = 8.15e-9 * SET_PARAMS.likelyhood_multiplier
        self.number_of_failures = 1
        self.failures = {
            0: "Angular_sensor_high_noise"
        }
        super().__init__(self.Fault_rate_per_hour, self.number_of_failures, self.failures, seed)

    def Angular_sensor_high_noise(self, sensor):
        return sensor*random_size(minimum = Min_high_noise, maximum = Max_high_noise) if self.failure == "Angular_sensor_high_noise" else sensor


class Star_tracker(Fault_parameters):
    def __init__(self, seed):
        self.Fault_rate_per_hour = 8.15e-9 * SET_PARAMS.likelyhood_multiplier
        self.number_of_failures = 1
        self.failures = {
            0: "Closed_shutter"
        }
        super().__init__(self.Fault_rate_per_hour, self.number_of_failures, self.failures, seed)

    def Closed_shutter(self, Star_tracker):
        if self.failure == "Closed_shutter":
            return np.zeros(Star_tracker.shape)
        else:
            return Star_tracker

class Overall_control(Fault_parameters):
    def __init__(self, seed):
        self.Fault_rate_per_hour = 8.15e-9 * SET_PARAMS.likelyhood_multiplier
        self.number_of_failures = 3
        self.failures = {
            0: "Increasing_angular_RW_momentum",
            1: "Decreasing_angular_RW_momentum",
            2: "Oscillating_angular_RW_momentum"
        }
        self.previous_mul = -1
        self.oscillation_magnitude = 0.2
        self.angular_wheels_max = SET_PARAMS.wheel_angular_d_max*random_size(minimum = Min_high_speed_percentage*0.75, maximum = Max_high_speed_percentage)
        self.angular_wheels_min = SET_PARAMS.wheel_angular_d_max*random_size(minimum = Min_low_speed_percentage, maximum = Max_low_speed_percentage)
        self.first = True
        super().__init__(self.Fault_rate_per_hour, self.number_of_failures, self.failures, seed)

    def Increasing_angular_RW_momentum(self, angular_wheels):
        if self.first:
            self.angular_wheels = angular_wheels
            self.first = False
        if self.failure == "Increasing_angular_RW_momentum":
            self.angular_wheels = np.minimum((self.angular_wheels + abs(self.angular_wheels)/10), self.angular_wheels_max*np.ones(angular_wheels.shape)) 
        else:
            return angular_wheels
        return self.angular_wheels

    def Decreasing_angular_RW_momentum(self, angular_wheels):
        if self.first:
            self.angular_wheels = angular_wheels
            self.first = False
        if self.failure == "Decreasing":
            self.angular_wheels = np.maximum((self.angular_wheels - abs(self.angular_wheels)/10), self.angular_wheels_min*np.ones(angular_wheels.shape))  
        else:
            return angular_wheels
        return self.angular_wheels

    def Oscillating_angular_RW_momentum(self, angular_wheels):
        if self.first:
            self.angular_wheels = angular_wheels
            self.first = False
        if self.failure == "Oscillating_angular_RW_momentum":
            self.angular_wheels = (self.angular_wheels + self.angular_wheels*self.oscillation_magnitude*self.previous_mul)
        else:
            return angular_wheels
        self.previous_mul = self.previous_mul*(-1)
        return self.angular_wheels

class Common_data_transmission(Fault_parameters):
    def __init__(self, seed):
        self.Fault_rate_per_hour = 8.15e-9 * SET_PARAMS.likelyhood_multiplier
        self.number_of_failures = 3
        self.failures = {
            0: "Bit_flip",
            1: "Sign_flip",
            2: "Insertion_of_zero_bit"
        }
        super().__init__(self.Fault_rate_per_hour, self.number_of_failures, self.failures, seed)

    def Bit_flip(self, value_to_change):
        if self.failure == "Bit_flip" and self.np_random.normal(0,0.5) < 0:
            ind = self.np_random.randint(0, len(value_to_change))
            value_to_change[ind] = random_bit_flip(value_to_change[ind])

        return value_to_change

    def Sign_flip(self, value_to_change):
        return -value_to_change if self.failure == "Sign_flip"  and self.np_random.normal(0,0.5) < 0 else value_to_change

    def Insertion_of_zero_bit(self, value_to_change):
        return np.zeros(value_to_change.shape) if self.failure == "Insertion_of_zero_bit"  and self.np_random.normal(0,0.5) < 0 else value_to_change




