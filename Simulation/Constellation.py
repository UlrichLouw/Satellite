import pandas as pd
import math
import numpy as np
import Simulation.Quaternion_functions
from Simulation.Parameters import SET_PARAMS
from sgp4.api import jday
from Simulation.dynamics import Constellation_Satellites
from Fault_prediction.Fault_detection import Encompassing_detection
import multiprocessing
if SET_PARAMS.Display:
    import Simulation.Satellite_display as view

pi = math.pi

dimensions = ['x', 'y', 'z','g','h']

class Constellation:
    def __init__(self, number_of_satellites):
        self.number_of_satellites = number_of_satellites
        self.positions = [None] * number_of_satellites
        self.data = [None] * number_of_satellites
        #! Remember to adjust the initial positions to different starts of the orbit parameters
        #! The orbit parameters must be designed based on the constellation and number of satellites
        self.inclination_per_sat = 360/number_of_satellites
        self.RAAN_per_sat = 360/number_of_satellites
        self.a_G0_per_sat = 360/number_of_satellites
        if SET_PARAMS.Display:
            display = view.initializeCube(SET_PARAMS.Dimensions)
            self.pv = view.ProjectionViewer(1920, 1080, display, self.number_of_satellites)
        self.satellites = []
        self.k_nearest_satellites = SET_PARAMS.k_nearest_satellites
        self.FD_strategy = SET_PARAMS.FD_strategy
        if self.FD_strategy == "Distributed" or self.FD_strategy == "Mixed":
            self.fault_vote = [None] * self.number_of_satellites # A vote of the health of a satellite per existing satellite
            self.nearest_neighbours_all = [None] * self.number_of_satellites
        self.FD = Encompassing_detection()
    
    def initiate_satellite(self, sat_num):
        sat_init = satellite(self, sat_num)
        sat_init.initialize()
        sat = multiprocessing.Process(target = sat_init.step)
        self.satellites.append(sat)

class satellite:
    def __init__(self, constellation, sat_num):
        self.constellation = constellation
        self.sat_num = sat_num
        self.Orbit_parameters()
        self.Dynamics = Constellation_Satellites(sat_num, self.s_list, self.t_list, self.J_t, self.fr)
        self.satellite_angles = np.zeros((self.constellation.number_of_satellites,))

    ###############################################
    # PROVIDE ORBIT PARAMETERS FOR EACH SATELLITE #
    ###############################################
    def Orbit_parameters(self):
        ####################
        # ORBIT PARAMETERS #
        ####################
        
        eccentricity = 0.000092                                 # Update eccentricity list
        inclination = 60 #self.constellation.inclination_per_sat*self.sat_num     # degrees
        Semi_major_axis = 6879.55                               # km The distance from the satellite to the earth + the earth radius
        Height_above_earth_surface = 500e3                      # distance above earth surface
        Scale_height = 8500                                     # scale height of earth atmosphere
        RAAN = self.constellation.RAAN_per_sat*self.sat_num     # Right ascension of the ascending node in radians
        #RAAN = 275*pi/180                                       # Right ascension of the ascending node in radians
        AP = 0                                                  # argument of perigee
        Re = 6371.2                                             # km magnetic reference radius
        Mean_motion = 15.2355000000                             # rev/day
        Mean_motion_per_second = Mean_motion/(3600.0*24.0)
        Mean_anomaly = 29.3                                     # degrees
        Argument_of_perigee = 57.4                              # in degrees
        omega = Argument_of_perigee
        Period = 86400/Mean_motion                              # seconds
        self.J_t,self.fr = jday(2020,2,16,15,30,0)                        # current julian date
        epoch = self.J_t - 2433281.5 + self.fr
        Drag_term = 0.000194                                    # Remember to update the list term
        wo = Mean_motion_per_second*(2*pi)                      # rad/s

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
        self.s_list = satellite_number_list + international_list + epoch_list + mean_motion_derivative_first_list + mean_motion_derivative_second_list + Drag_term_list + Ephereris_list + element_num_checksum_list
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
        self.t_list = line_and_satellite_number_list + inclination_list + intermediate_list + RAAN_list + intermediate_list_2 + eccentricity_list + perigree_list + intermediate_list_3 + mean_anomaly_list + intermediate_list_4 + mean_motion_list + Epoch_rev_list
        self.a_G0 = 0 #self.constellation.a_G0_per_sat*self.sat_num


    #####################################
    # INITIATE THE SATELLITE SIMULATION #
    #####################################
    def initialize(self):
        w, q, A, r, sun_in_view = self.Dynamics.rotation()
        self.constellation.data[self.sat_num] = self.Dynamics.Orbit_Data
        self.constellation.positions[self.sat_num] = self.Dynamics.sense.position/np.linalg.norm(self.Dynamics.sense.position)
        if SET_PARAMS.Display:
            self.constellation.pv.run(w, q, A, r, sun_in_view = True, only_positions = True, sat_num = self.sat_num)


    ############################################
    # PERFORM THE SIMULATION FOR EACH TIMESTEP #
    ############################################
    def step(self):
        w, q, A, r, sun_in_view = self.Dynamics.rotation()
        data_unfiltered = self.Dynamics.Orbit_Data

        # Convert array's to individual values in the dictionary
        data = {col + "_" + dimensions[i]: data_unfiltered[col][i] for col in data_unfiltered if isinstance(data_unfiltered[col], np.ndarray) for i in range(len(data_unfiltered[col]))}

        # Add all the values to the dictionary that is not numpy arrays
        for col in data_unfiltered:
            if not isinstance(data_unfiltered[col], np.ndarray):
                data[col] = data_unfiltered[col]

        self.constellation.positions[self.sat_num] = self.Dynamics.sense.position/np.linalg.norm(self.Dynamics.sense.position)

        if SET_PARAMS.Display and self.steps%SET_PARAMS.skip == 0:
            self.constellation.pv.run(w, q, A, r, sun_in_view = True, only_positions = True, sat_num = self.sat_num)

        if self.constellation.FD_strategy == "Individual":
            predictions = self.constellation.FD.Per_Timestep(data, self.FD_strategy)
            ###############################################################################
            # USE THE VOTE OF EACH SATELLITE TO DETERMINE THE HEALTH OF ANOTHER SATELLITE #
            ###############################################################################
            self.constellation.fault_vote[self.sat_num] = predictions
        else:
            self.nearest_neighbours_func()
            data["Satellite number"] = self.sat_num
            data["Nearest Neighbours"] = self.nearest_neighbours
            self.constellation.data[self.sat_num] = data
            self.constellation.nearest_neighbours_all[self.sat_num] = self.nearest_neighbours




    ######################################
    # DETERMINE THE K NEAREST SATELLITES #
    ######################################
    def nearest_neighbours_func(self):
        for sats in range(self.constellation.number_of_satellites):
            if sats != self.sat_num:
                self.satellite_angles[sats] = abs(np.arccos(np.dot(self.constellation.positions[self.sat_num],self.constellation.positions[sats])))
            else:
                self.satellite_angles[sats] = 0

        self.nearest_neighbours = np.argpartition(self.satellite_angles, self.constellation.k_nearest_satellites + 1)[:self.constellation.k_nearest_satellites + 1]
        self.nearest_neighbours = [item for item in self.nearest_neighbours if item != self.sat_num]

        
        