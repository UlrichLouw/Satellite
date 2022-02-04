import sys
import numpy as np
from Simulation.Parameters import SET_PARAMS
import pandas as pd
import multiprocessing
from pathlib import Path
from Simulation.dynamics import Single_Satellite
from Simulation.Save_display import visualize_data, save_as_csv, save_as_pickle, save_as_excel
import Simulation.Constellation as Constellation
import math
from sgp4.api import jday
import csv
import os
from argparse import Namespace
import yaml

pi = math.pi

dimensions = ['x', 'y', 'z']

SET_PARAMS.Display = False

# ! The matplotlib cannot display plots while visual simulation runs.
# ! Consequently the Display and visualize parameters in Parameters 
# ! must be set as desired

if SET_PARAMS.Display:
    import Simulation.Satellite_display as view

#####################################
# LOOP THROUGH DYNAMICS IF MULTIPLE #
#       THREADS ARE REQUIRED        #
#####################################
def loop(index, D, SET_PARAMS):
    #! print(SET_PARAMS.Fault_names_values[index])
    if SET_PARAMS.Display:
        satellite = view.initializeCube(SET_PARAMS.Dimensions)
        pv = view.ProjectionViewer(1920, 1080, satellite)

    if SET_PARAMS.NumberOfRandom > 1:
        GenericPath = "Predictor-" + str(SET_PARAMS.SensorPredictor)+ "/Isolator-" + str(SET_PARAMS.SensorIsolator) + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/" + SET_PARAMS.Model_or_Measured +"/" + \
                    "SunSensorSize-Length:" + str(SET_PARAMS.Sun_sensor_length) + "-Width:" + str(SET_PARAMS.Sun_sensor_width) + "/" + str(SET_PARAMS.Fault_names_values[index]) 
        path = "Data files/"+ GenericPath
        path = path + "/" + "SolarPanel-Length: " + str(SET_PARAMS.SP_Length) + "SolarPanel-Width: " + str(SET_PARAMS.SP_width) + \
                    "Raan: " + str(SET_PARAMS.RAAN) + " inclinination: " +str(SET_PARAMS.inclination)
        
    else:
        GenericPath = "Predictor-" + str(SET_PARAMS.SensorPredictor)+ "/Isolator-" + str(SET_PARAMS.SensorIsolator) + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/" + SET_PARAMS.Model_or_Measured +"/" + "General CubeSat Model/"
        path = "Data files/"+ GenericPath

    if SET_PARAMS.Low_Aerodynamic_Disturbance:
        GenericPath = "Low_Disturbance/" + GenericPath
        path = "Data files/"+ GenericPath

    path_to_folder = Path(path)
    path_to_folder.mkdir(parents = True, exist_ok=True)
    
    Visualize_data = {col: [] for col in D.Orbit_Data}
    # KalmanControl = {col: [] for col in SET_PARAMS.visualizeKalman}
    MeasurementUpdates = {col: [] for col in SET_PARAMS.measurementUpdateVars}
    
    Columns = []

    for col in D.Orbit_Data:
        if isinstance(D.Orbit_Data[col], np.ndarray) and col != "Moving Average":
            for i in range(len(dimensions)):
                Columns.append(col + "_" + dimensions[i])
        else:
            Columns.append(col)

    filename = path + str(SET_PARAMS.Fault_names_values[index]) + ".csv"


    if os.path.exists(filename):
        os.remove(filename)
        print("Remove file")
    else:
        print("The file does not exist") 

    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        
        # writing the fields
        csvwriter.writerow(D.globalArray)

        # Data = pd.DataFrame(columns=Columns, index = [*range(1, int(SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)+1))])
        
        for j in range(1, int(SET_PARAMS.Number_of_orbits*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)+1)):
            w, q, A, r, sun_in_view = D.rotation()
            if SET_PARAMS.Display and j%SET_PARAMS.skip == 0:
                pv.run(w, q, A, r, sun_in_view)

            if j%(int(SET_PARAMS.Number_of_orbits*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)/100)) == 0:
                print("Number of time steps for orbit loop number", index, " = ", "%.2f" % float(j/int(SET_PARAMS.Number_of_orbits*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts))))

            # if j%int(SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)) == 0:
            #     Data = pd.DataFrame(columns=Columns, index = [*range(1, int(SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)+1))])

            if SET_PARAMS.fixed_orbit_failure == 0:
                D.initiate_purposed_fault(SET_PARAMS.Fault_names_values[index])
                if SET_PARAMS.Display:
                    pv.fault = D.fault

            elif SET_PARAMS.Fault_simulation_mode == 2 and j%(int(SET_PARAMS.Number_of_orbits*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)/SET_PARAMS.fixed_orbit_failure)) == 0:
                D.initiate_purposed_fault(SET_PARAMS.Fault_names_values[index])
                print(SET_PARAMS.Fault_names_values[index], "is initiated")
                if SET_PARAMS.Display:
                    pv.fault = D.fault

            # data_unfiltered = D.Orbit_Data
                
            # writing the fields
            csvwriter.writerow(D.globalArray)

    Data = pd.read_csv(filename)

    Datapgf= Data[int((SET_PARAMS.Number_of_orbits-1)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)):]

    DatapgfSensors = Datapgf.loc[:,Datapgf.columns.str.contains('Sun') | Datapgf.columns.str.contains('Magnetometer') |
                            Datapgf.columns.str.contains('Earth') | Datapgf.columns.str.contains('Angular momentum of wheels') |
                            Datapgf.columns.str.contains('Star')]
    
    DatapgfTorques = Datapgf.loc[:, Datapgf.columns.str.contains('Torques')]

    DatapgfKalmanFilter = Datapgf.loc[:,Datapgf.columns.str.contains('Quaternions') | Datapgf.columns.str.contains('Euler Angles') | Datapgf.columns.str.contains('Angular velocity of satellite')]

    DatapgfPrediction = Datapgf.loc[:,Datapgf.columns.str.contains('Accuracy') | Datapgf.columns.str.contains('fault')]

    DatapgfMetric = Datapgf.loc[:,Datapgf.columns.str.contains('Metric')]

    if SET_PARAMS.Visualize and SET_PARAMS.Display == False:
        pathPlots = "Plots/"+ GenericPath + str(SET_PARAMS.Fault_names_values[index]) + "/"
        path_to_folder = Path(pathPlots)
        path_to_folder.mkdir(parents = True, exist_ok=True)
        visualize_data(Data, SET_PARAMS.Fault_names_values[index], path = pathPlots)
    
    elif SET_PARAMS.Display == True:
        pv.save_plot(D.fault)

    # if SET_PARAMS.save_as == ".csv":
    #     save_as_csv(Data, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path, mode = 'a')
    # else:
    #     save_as_pickle(Data, index)

    if SET_PARAMS.NumberOfRandom <= 1:
        path = "Data files/pgfPlots/" + GenericPath

        if SET_PARAMS.save_as == ".csv":
            path_to_folder = Path(path + "/Sensors/")
            path_to_folder.mkdir(parents = True, exist_ok=True)
            save_as_csv(DatapgfSensors, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path + "/Sensors/")
        else:
            save_as_pickle(DatapgfSensors, index)

        if SET_PARAMS.save_as == ".csv":
            path_to_folder = Path(path + "/Torques/")
            path_to_folder.mkdir(parents = True, exist_ok=True)
            save_as_csv(DatapgfTorques, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path + "/Torques/")
        else:
            save_as_pickle(DatapgfTorques, index)

        if SET_PARAMS.save_as == ".csv":
            path_to_folder = Path(path + "/KalmanFilter/")
            path_to_folder.mkdir(parents = True, exist_ok=True)
            save_as_csv(DatapgfKalmanFilter, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path + "/KalmanFilter/")
        else:
            save_as_pickle(DatapgfKalmanFilter, index)

        if SET_PARAMS.save_as == ".csv":
            path_to_folder = Path(path + "/Prediction/")
            path_to_folder.mkdir(parents = True, exist_ok=True)
            save_as_csv(DatapgfPrediction, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path + "/Prediction/")
        else:
            save_as_pickle(DatapgfPrediction, index)

        if SET_PARAMS.save_as == ".csv":
            path_to_folder = Path(path + "/Metric/")
            path_to_folder.mkdir(parents = True, exist_ok=True)
            save_as_csv(DatapgfMetric, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path + "/Metric/")
        else:
            save_as_pickle(DatapgfMetric, index)

    print("Number of multiple orbits", index)  

def constellationMultiProcessing(fault, SET_PARAMS):
    Stellar = Constellation.Constellation(SET_PARAMS.Number_of_satellites, fault = fault)
    allSatData = {}
    Columns = []

    for sat_num in range(SET_PARAMS.Number_of_satellites):
        Stellar.initiate_satellite(sat_num)
        allSatData[sat_num] = []

    for col in Stellar.data[0]:
        for k in range(SET_PARAMS.k_nearest_satellites + 1):
            if isinstance(Stellar.data[0][col], np.ndarray) and col != "Moving Average":
                for i in range(len(dimensions)):
                    Columns.append(str(k) + "_" + col + "_" + dimensions[i])
            else:
                Columns.append(str(k) + "_" + col)

    Data = pd.DataFrame(columns=Columns, index = [*range(int(SET_PARAMS.Number_of_orbits*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)+1))])

    DataList = {x: Data.copy() for x in range(SET_PARAMS.Number_of_satellites)}

    for j in range(int(SET_PARAMS.Number_of_orbits*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)+1)):
        for sat_num in range(SET_PARAMS.Number_of_satellites):
            Stellar.satellites[sat_num].run()

        if Stellar.FD_strategy == "Centralised":
            data = Stellar.data
            predictions = Stellar.FD.Per_Timestep(data, Stellar.FD_strategy)
            ###############################################################################
            # USE THE VOTE OF EACH SATELLITE TO DETERMINE THE HEALTH OF ANOTHER SATELLITE #
            ###############################################################################
            Stellar.fault_vote = predictions

        

        elif Stellar.FD_strategy == "Distributed" or Stellar.FD_strategy == "Mixed":
            for sat_num in range(SET_PARAMS.Number_of_satellites):

                
                dataKNearest = [Stellar.data[item] for item in Stellar.nearest_neighbours_all[sat_num]]
                dataKNearest.insert(0, Stellar.data[sat_num])
                
                # Ensure that predictions is a dictionary
                #! predictions = Stellar.FD.Per_Timestep(dataKNearest, Stellar.FD_strategy, Stellar.nearest_neighbours_all[sat_num])
                ###############################################################################
                # USE THE VOTE OF EACH SATELLITE TO DETERMINE THE HEALTH OF ANOTHER SATELLITE #
                ###############################################################################
                #! for sat in Stellar.nearest_neighbours_all[sat_num]:
                #!    Stellar.fault_vote[sat] = predictions[sat]

                k = 0
                
                for data in dataKNearest:
                    
                    for col in data:
                        if isinstance(data[col], np.ndarray) and col != "Moving Average":
                            for i in range(len(dimensions)):
                                DataList[sat_num][str(k) + "_" + col + "_" + dimensions[i]][j] = data[col][i]
                        else:
                            DataList[sat_num][str(k) + "_" + col][j] = data[col]
                            

                    k += 1

        if j%(int(SET_PARAMS.Number_of_orbits*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)/100)) == 0:
            print("Number of time steps for orbit loop number", fault, " = ", "%.3f" % float(j/int(SET_PARAMS.Number_of_orbits*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts))))


    GenericPath = "Constellation/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/" + SET_PARAMS.Model_or_Measured +"/" + "General CubeSat Model/"


    for sat_num in range(SET_PARAMS.Number_of_satellites):
        path = "Data files/"+ GenericPath + "/" + str(sat_num) + "/"
        path_to_folder = Path(path) 
        path_to_folder.mkdir(parents = True, exist_ok=True)
        save_as_csv(DataList[sat_num], filename = SET_PARAMS.Fault_names_values[fault], index = fault, path = path)

def loadParameters(fileName):
    full_path =  "Configurations/" + fileName + "/" + fileName + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)    
    
    conf = Namespace(**conf_dict)    
    return conf
        
def load_yaml_dict(fileName):
    full_path =  "Configurations/" + fileName + "/" + fileName + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)    
    
    return conf_dict

################################################################
# FOR ALL OF THE FAULTS RUN A NUMBER OF ORBITS TO COLLECT DATA #
################################################################
#if __name__ == "__main__":
def main():
    #########################################################
    # IF THE SAVE AS IS EQUAL TO XLSX, THE THREADING CANNOT #
    #           BE USED TO SAVE SHEETS                      #     
    #########################################################
    Configuration = "EarthSun"

    params = load_yaml_dict(Configuration)

    SET_PARAMS.Display = params["Display"]
    SET_PARAMS.Visualize = params["Visualize"]
    SET_PARAMS.save_as = params["save_as"]
    SET_PARAMS.Kalman_filter_use = params["Kalman_filter_use"]
    SET_PARAMS.sensor_number = params["sensor_number"]
    SET_PARAMS.Number_of_orbits = params["Number_of_orbits"]
    SET_PARAMS.fixed_orbit_failure = params["fixed_orbit_failure"]
    SET_PARAMS.Number_of_multiple_orbits = params["Number_of_multiple_orbits"]
    SET_PARAMS.skip = params["skip"]
    SET_PARAMS.Number_of_satellites = params["Number_of_satellites"]
    SET_PARAMS.k_nearest_satellites = params["k_nearest_satellites"]
    SET_PARAMS.FD_strategy = params["FD_strategy"]
    SET_PARAMS.SensorFDIR = params["SensorFDIR"]
    SET_PARAMS.Mode = params["Mode"]
    SET_PARAMS.stateBufferLength = params["stateBufferLength"]
    numFaultStart = params["numFaultStart"]
    SET_PARAMS.NumberOfRandom = params["NumberOfRandom"]
    SET_PARAMS.NumberOfFailuresReset = params["NumberOfFailuresReset"]
    SET_PARAMS.Model_or_Measured = params["Model_or_Measured"]
    SET_PARAMS.Low_Aerodynamic_Disturbance = params["Low_Aerodynamic_Disturbance"]
    SET_PARAMS.UsePredeterminedPositionalData = params["UsePredeterminedPositionalData"]
    SET_PARAMS.no_aero_disturbance = params["no_aero_disturbance"]
    SET_PARAMS.no_wheel_disturbance = params["no_wheel_disturbance"]
    SET_PARAMS.kalmanSensors = params["kalmanSensors"]
    SET_PARAMS.printBreak = params["printBreak"]

    SET_PARAMS.NumberOfIntegrationSteps = params["NumberOfIntegrationSteps"]

    includeNone = False

    featureExtractionMethods = params["featureExtractionMethods"]
    predictionMethods = params["predictionMethods"]
    isolationMethods = params["isolationMethods"]
    recoveryMethods = params["recoveryMethods"]
    recoverMethodsWithoutPrediction = params["recoverMethodsWithoutPrediction"]

    SET_PARAMS.measurementUpdateVars = ["Mean", "Covariance"]

    settling_time = params["settling_time"]
    damping_coefficient = params["damping_coefficient"]
    wn = 1/(settling_time*damping_coefficient)

    #! If the current settings do not work, then the Kw parameter should change (the Kw parameter should decrease
    #! since the oscillations increase)

    SET_PARAMS.measurement_noise = params["measurement_noise"]
    SET_PARAMS.process_noise = params["process_noise"]

    SET_PARAMS.P_k = np.eye(7)
    SET_PARAMS.R_k = np.eye(3)*(SET_PARAMS.process_noise**2 + SET_PARAMS.measurement_noise**2) #* np.eye(3)*1e-4
    SET_PARAMS.Q_k = np.diag([8.89e-8, 8.89e-8, 8.89e-8, 7.4e-7, 7.4e-7, 7.4e-7, 7.4e-7])

    SET_PARAMS.Kp = 2 * wn**2
    SET_PARAMS.Kd = 2 * damping_coefficient * wn
    SET_PARAMS.Kw = 2e-3 #! *2e-3 #! I just changed this from e-6 to e-5 to e-4 to e-3

    #####################################
    # PARAMETERS FOR SATELLITE DYNAMICS #
    #####################################

    s_list, t_list, J_t, fr = SET_PARAMS.s_list, SET_PARAMS.t_list, SET_PARAMS.J_t, SET_PARAMS.fr

    #########################################################
    #   TO ENABLE A CONSTELLATION A CLASS IS CREATED THAT   #
    #     CONTAINS THE DATA OF THE ENTIRE CONSTELLATION     #
    #  THAT DATA IS TRANSFERED TO EACH SATELLITE DEPENDING  #
    # ON THE SATELLITES ID AND THE SATELLITES CLOSEST TO IT #
    #########################################################

    if SET_PARAMS.Number_of_satellites > 1:
        numProcess = 0
        threads = []
        for extraction in featureExtractionMethods:
            for recovery in recoveryMethods:
                for prediction in predictionMethods:
                    for isolation in isolationMethods:
                        if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None") or (prediction == isolation and prediction != "None" and recovery not in recoverMethodsWithoutPrediction):
                            if SET_PARAMS.SensorFDIR:
                                SET_PARAMS.FeatureExtraction = extraction
                                SET_PARAMS.SensorPredictor = prediction
                                SET_PARAMS.SensorIsolator = isolation
                                SET_PARAMS.SensorRecoveror = recovery
                            else:
                                SET_PARAMS.FeatureExtraction = extraction
                                SET_PARAMS.SensorPredictor = "None"
                                SET_PARAMS.SensorIsolator = "None"
                                SET_PARAMS.SensorRecoveror = "None"

                            
                            #! 2nd change to only run on faults and not "NONE"
                            for i in range(numFaultStart, SET_PARAMS.Number_of_multiple_orbits+1):
                                numProcess += 1
                                print("Beginning of", extraction, prediction, isolation, recovery, i)
                                t = multiprocessing.Process(target=constellationMultiProcessing, args=(i, SET_PARAMS))
                                threads.append(t)
                                t.start()
        
        for process in threads:     
            process.join()

        threads.clear()


    # elif SET_PARAMS.Number_of_multiple_orbits == 1:
    #     FD = Fault_detection.Basic_detection()
    #     for i in range(1, SET_PARAMS.Number_of_multiple_orbits + 1):
    #         D = Single_Satellite(i, s_list, t_list, J_t, fr)

    #         print(SET_PARAMS.Fault_names_values[i])

    #         loop(i, D, SET_PARAMS)

    ######################################################
    # IF THE SAVE AS IS NOT EQUAL TO XLSX, THE THREADING #
    #           CAN BE USED TO SAVE CSV FILES            #
    ######################################################
    else:
        inclination_per_sat = 360/SET_PARAMS.NumberOfRandom
        RAAN_per_sat = 360/SET_PARAMS.NumberOfRandom

        if SET_PARAMS.NumberOfRandom > 1:
            for randomSizes in range(SET_PARAMS.NumberOfRandom):

                # SET_PARAMS.Sun_sensor_length = SET_PARAMS.Sun_sensor_length + (np.random.rand() - 0.5) * SET_PARAMS.Ly 
                # SET_PARAMS.Sun_sensor_width = SET_PARAMS.Sun_sensor_length + (np.random.rand() - 0.5) * SET_PARAMS.Ly 
                SET_PARAMS.SP_Length = SET_PARAMS.Lx + (np.random.rand() - 0.5) * SET_PARAMS.Lx 
                SET_PARAMS.SP_width = SET_PARAMS.Ly + (np.random.rand() - 0.5) * SET_PARAMS.Ly 

                for randomOrbits in range(SET_PARAMS.NumberOfRandom):
                    ####################
                    # ORBIT PARAMETERS #
                    ####################
                    
                    eccentricity = 0.000092                                 # Update eccentricity list
                    inclination = inclination_per_sat*randomOrbits   # degrees
                    SET_PARAMS.inclination = inclination
                    Semi_major_axis = 6879.55                               # km The distance from the satellite to the earth + the earth radius
                    Height_above_earth_surface = 500e3                      # distance above earth surface
                    Scale_height = 8500                                     # scale height of earth atmosphere
                    RAAN = RAAN_per_sat*randomOrbits    # Right ascension of the ascending node in radians
                    SET_PARAMS.RAAN = RAAN
                    #RAAN = 275*pi/180                                       # Right ascension of the ascending node in radians
                    AP = 0                                                  # argument of perigee
                    Re = 6371.2                                             # km magnetic reference radius
                    Mean_motion = 15.2355000000                             # rev/day
                    Mean_motion_per_second = Mean_motion/(3600.0*24.0)
                    Mean_anomaly = 29.3                                     # degrees
                    Argument_of_perigee = 57.4                              # in degrees
                    omega = Argument_of_perigee
                    Period = 86400/Mean_motion                              # seconds
                    J_t,fr = jday(2020,2,16,15,30,0)                        # current julian date
                    epoch = J_t - 2433281.5 + fr
                    Drag_term = 0.000194                                    # Remember to update the list term
                    wo = Mean_motion_per_second*(2*pi)                      # rad/s

                    ############
                    # TLE DATA #
                    ############
                    # Create multiple random orbit parameters
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
                    
                    SET_PARAMS.t_list = t_list
                    SET_PARAMS.s_list = s_list
                    numProcess = 0
                    threads = []
                    for extraction in featureExtractionMethods:
                        for recovery in recoveryMethods:
                            for prediction in predictionMethods:
                                for isolation in isolationMethods:
                                    if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None") or (prediction == isolation and prediction != "None" and recovery not in recoverMethodsWithoutPrediction):
                                        if SET_PARAMS.SensorFDIR:
                                            SET_PARAMS.FeatureExtraction = extraction
                                            SET_PARAMS.SensorPredictor = prediction
                                            SET_PARAMS.SensorIsolator = isolation
                                            SET_PARAMS.SensorRecoveror = recovery
                                        else:
                                            SET_PARAMS.FeatureExtraction = extraction
                                            SET_PARAMS.SensorPredictor = "None"
                                            SET_PARAMS.SensorIsolator = "None"
                                            SET_PARAMS.SensorRecoveror = "None"

                                        
                                        #! 2nd change to only run on faults and not "NONE"
                                        for i in range(numFaultStart, SET_PARAMS.Number_of_multiple_orbits+1):
                                            numProcess += 1
                                            D = Single_Satellite(i, s_list, t_list, J_t, fr)

                                            t = multiprocessing.Process(target=loop, args=(i, D, SET_PARAMS))
                                            threads.append(t)
                                            t.start()
                                            print("Beginning of", extraction, prediction, isolation, recovery, i)
                    if includeNone:
                        temp = SET_PARAMS.SensorFDIR
                        SET_PARAMS.SensorFDIR = False
                        SET_PARAMS.FeatureExtraction = extraction
                        SET_PARAMS.SensorPredictor = "None"
                        SET_PARAMS.SensorIsolator = "None"
                        SET_PARAMS.SensorRecoveror = "None"

                        
                        #! 2nd change to only run on faults and not "NONE"
                        for i in range(numFaultStart, SET_PARAMS.Number_of_multiple_orbits+1):
                            numProcess += 1
                            D = Single_Satellite(i, s_list, t_list, J_t, fr)

                            t = multiprocessing.Process(target=loop, args=(i, D, SET_PARAMS))
                            threads.append(t)
                            t.start()
                            print("Beginning of", extraction, "None", "None", "None", i)
                        
                        SET_PARAMS.SensorFDIR = temp

                    for process in threads:     
                        process.join()

                    threads.clear()
        else:
            numProcess = 0
            threads = []
            for extraction in featureExtractionMethods:
                for recovery in recoveryMethods:
                    for prediction in predictionMethods:
                        for isolation in isolationMethods:
                            if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None") or (prediction == isolation and prediction != "None" and recovery not in recoverMethodsWithoutPrediction):
                                if SET_PARAMS.SensorFDIR:
                                    SET_PARAMS.FeatureExtraction = extraction
                                    SET_PARAMS.SensorPredictor = prediction
                                    SET_PARAMS.SensorIsolator = isolation
                                    SET_PARAMS.SensorRecoveror = recovery
                                else:
                                    SET_PARAMS.FeatureExtraction = extraction
                                    SET_PARAMS.SensorPredictor = "None"
                                    SET_PARAMS.SensorIsolator = "None"
                                    SET_PARAMS.SensorRecoveror = "None"

                                
                                #! 2nd change to only run on faults and not "NONE"
                                for i in range(numFaultStart, SET_PARAMS.Number_of_multiple_orbits+1):
                                    numProcess += 1
                                    D = Single_Satellite(i, s_list, t_list, J_t, fr)

                                    t = multiprocessing.Process(target=loop, args=(i, D, SET_PARAMS))
                                    threads.append(t)
                                    t.start()
                                    print("Beginning of", extraction, prediction, isolation, recovery, i)
        
        if includeNone:
            temp = SET_PARAMS.SensorFDIR
            SET_PARAMS.SensorFDIR = False
            SET_PARAMS.FeatureExtraction = extraction
            SET_PARAMS.SensorPredictor = "None"
            SET_PARAMS.SensorIsolator = "None"
            SET_PARAMS.SensorRecoveror = "None"

            
            #! 2nd change to only run on faults and not "NONE"
            for i in range(numFaultStart, SET_PARAMS.Number_of_multiple_orbits+1):
                numProcess += 1
                D = Single_Satellite(i, s_list, t_list, J_t, fr)

                t = multiprocessing.Process(target=loop, args=(i, D, SET_PARAMS))
                threads.append(t)
                t.start()
                print("Beginning of", extraction, "None", "None", "None", i)

            SET_PARAMS.SensorFDIR = temp

        for process in threads:     
            process.join()

        threads.clear()


if __name__ == "__main__": 
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()