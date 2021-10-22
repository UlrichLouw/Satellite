import sys
import numpy as np
from Simulation.Parameters import SET_PARAMS
import pandas as pd
import multiprocessing
from pathlib import Path
from Simulation.dynamics import Single_Satellite
from Simulation.Save_display import visualize_data, save_as_csv, save_as_pickle, save_as_excel
import Fault_prediction.Fault_detection as Fault_detection
import Simulation.Constellation as Constellation
from numba import njit, jit, vectorize

dimensions = ['x', 'y', 'z']

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

    Overall_data = []

    if SET_PARAMS.Display:
        satellite = view.initializeCube(SET_PARAMS.Dimensions)
        pv = view.ProjectionViewer(1920, 1080, satellite)

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

    Data = pd.DataFrame(columns=Columns, index = [0])
    
    for j in range(1, int(SET_PARAMS.Number_of_orbits*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)+1)):
        w, q, A, r, sun_in_view = D.rotation()
        if SET_PARAMS.Display and j%SET_PARAMS.skip == 0:
            pv.run(w, q, A, r, sun_in_view)

        if j%(int(SET_PARAMS.Number_of_orbits*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)/10)) == 0:
            print("Number of time steps for orbit loop number", index, " = ", "%.2f" % float(j/int(SET_PARAMS.Number_of_orbits*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts))))

        if SET_PARAMS.Fault_simulation_mode == 2 and j%(int(SET_PARAMS.Number_of_orbits*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)/SET_PARAMS.fixed_orbit_failure)) == 0:
            D.initiate_purposed_fault(SET_PARAMS.Fault_names_values[index])
            print(SET_PARAMS.Fault_names_values[index], "is initiated")
            if SET_PARAMS.Display:
                pv.fault = D.fault

        data_unfiltered = D.Orbit_Data

        # # Convert array's to individual values in the dictionary
        # data = {col + "_" + dimensions[i]: data_unfiltered[col][i] for col in data_unfiltered if isinstance(data_unfiltered[col], np.ndarray) and col != "Moving Average" for i in range(len(data_unfiltered[col]))}

        # Add all the values to the dictionary that is not numpy arrays
        for col in data_unfiltered:
            Visualize_data[col].append(data_unfiltered[col])

            # if not isinstance(data_unfiltered[col], np.ndarray):
            #     data[col] = data_unfiltered[col]
        for col in data_unfiltered:
            if isinstance(D.Orbit_Data[col], np.ndarray) and col != "Moving Average":
                for i in range(len(dimensions)):
                    Data[col + "_" + dimensions[i]][0] = data_unfiltered[col][i]
            else:
                Data[col][0] = data_unfiltered[col]

        
        Overall_data.append(Data.copy())
        # for col in D.KalmanControl:
        #     KalmanControl[col].append(D.KalmanControl[col])

        for col in D.MeasurementUpdateDictionary:
            perTimestep = D.MeasurementUpdateDictionary[col]
            for var in perTimestep:
                MeasurementUpdates[col].append(var)

    Data = pd.concat(Overall_data)

    Datapgf= Data[int((SET_PARAMS.Number_of_orbits-1)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)):]

    DatapgfSensors = Datapgf.loc[:,Datapgf.columns.str.contains('Sun') | Datapgf.columns.str.contains('Magnetometer') |
                            Datapgf.columns.str.contains('Earth') | Datapgf.columns.str.contains('Angular momentum of wheels') |
                            Datapgf.columns.str.contains('Star')]
    
    DatapgfTorques = Datapgf.loc[:, Datapgf.columns.str.contains('Torques')]

    DatapgfKalmanFilter = Datapgf.loc[:,Datapgf.columns.str.contains('Quaternions') | Datapgf.columns.str.contains('Euler Angles') | Datapgf.columns.str.contains('Angular velocity of satellite')]

    DatapgfPrediction = Datapgf.loc[:,Datapgf.columns.str.contains('Accuracy') | Datapgf.columns.str.contains('fault')]

    if SET_PARAMS.Visualize and SET_PARAMS.Display == False:
        if SET_PARAMS.Reflection:
            path = "Plots/"+ "Predictor-" + SET_PARAMS.SensorPredictor + "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror + "/KalmanFilter-"+SET_PARAMS.Kalman_filter_use+"/"+SET_PARAMS.Mode+"_with_reflection/"+ str(D.fault) + "/"
            path_to_folder = Path(path)
            path_to_folder.mkdir(parents = True, exist_ok=True)
        else:
            path = "Plots/"+ "Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/KalmanFilter-"+SET_PARAMS.Kalman_filter_use+"/"+SET_PARAMS.Mode+"/"+ str(D.fault) + "/"
            path_to_folder = Path(path)
            path_to_folder.mkdir(parents = True, exist_ok=True)
        visualize_data(Visualize_data, D.fault, path = path)
        # visualize_data(KalmanControl, D.fault, path = path)
        visualize_data(MeasurementUpdates, D.fault, path = path)
    
    elif SET_PARAMS.Display == True:
        pv.save_plot(D.fault)

    print("Number of multiple orbits", index)  

    path = "Data files/"+ "Predictor-" + SET_PARAMS.SensorPredictor + "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror + "/KalmanFilter-"+SET_PARAMS.Kalman_filter_use+"/"+SET_PARAMS.Mode+"/"
    path_to_folder = Path(path)
    path_to_folder.mkdir(parents = True, exist_ok=True)

    if SET_PARAMS.save_as == ".csv":
        save_as_csv(Data, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path)
    else:
        save_as_pickle(Data, index)

    path = "Data files/pgfPlots/"+ "Predictor-" + SET_PARAMS.SensorPredictor + "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror + "/KalmanFilter-"+SET_PARAMS.Kalman_filter_use+"/"+SET_PARAMS.Mode+"/"
    path_to_folder = Path(path)
    path_to_folder.mkdir(parents = True, exist_ok=True)

    if SET_PARAMS.save_as == ".csv":
        save_as_csv(DatapgfSensors, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path + "Sensors")
    else:
        save_as_pickle(DatapgfSensors, index)

    if SET_PARAMS.save_as == ".csv":
        save_as_csv(DatapgfTorques, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path + "Torques")
    else:
        save_as_pickle(DatapgfTorques, index)

    if SET_PARAMS.save_as == ".csv":
        save_as_csv(DatapgfKalmanFilter, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path + "KalmanFilter")
    else:
        save_as_pickle(DatapgfKalmanFilter, index)

    if SET_PARAMS.save_as == ".csv":
        save_as_csv(DatapgfPrediction, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path + "Prediction")
    else:
        save_as_pickle(DatapgfPrediction, index)

################################################################
# FOR ALL OF THE FAULTS RUN A NUMBER OF ORBITS TO COLLECT DATA #
################################################################
#if __name__ == "__main__":
def main():
    #########################################################
    # IF THE SAVE AS IS EQUAL TO XLSX, THE THREADING CANNOT #
    #           BE USED TO SAVE SHEETS                      #     
    #########################################################
    SET_PARAMS.Display = False
    SET_PARAMS.Visualize = True
    SET_PARAMS.save_as = ".csv"
    SET_PARAMS.Kalman_filter_use = "EKF"
    SET_PARAMS.sensor_number = "ALL"
    SET_PARAMS.Number_of_orbits = 40
    SET_PARAMS.fixed_orbit_failure = 10
    SET_PARAMS.Number_of_multiple_orbits = len(SET_PARAMS.Fault_names)
    SET_PARAMS.skip = 20
    SET_PARAMS.Number_of_satellites = 1
    SET_PARAMS.k_nearest_satellites = 5
    SET_PARAMS.FD_strategy = "Distributed"
    SET_PARAMS.SensorFDIR = True
    SET_PARAMS.Mode = "EARTH_SUN" # Nominal or EARTH_SUN
    #SET_PARAMS.Mode = "Nominal"

    featureExtractionMethods = ["DMD"]
    predictionMethods = ["None"]
    isolationMethods = ["None"]
    recoveryMethods = ["None"]

    if SET_PARAMS.SensorFDIR:
        featureExtractionMethods = ["DMD"]
        predictionMethods = ["DecisionTrees", "PERFECT"]
        isolationMethods = ["DecisionTrees", "PERFECT"]
        recoveryMethods = ["EKF"]
        SET_PARAMS.FeatureExtraction = "DMD"
        SET_PARAMS.SensorPredictor = "PERFECT"
        SET_PARAMS.SensorIsolator = "PERFECT"
        SET_PARAMS.SensorRecoveror = "EKF"
    else:
        SET_PARAMS.FeatureExtraction = "DMD"
        SET_PARAMS.SensorPredictor = "None"
        SET_PARAMS.SensorIsolator = "None"
        SET_PARAMS.SensorRecoveror = "None"
    
    # SET_PARAMS.visualizeKalman = ["w_est","w_act","quaternion_est","quaternion_actual",
    #                             "quaternion_ref", "w_ref","quaternion_error",
    #                             "w_error"]

    SET_PARAMS.measurementUpdateVars = ["Mean", "Covariance"]

    settling_time = 300
    damping_coefficient = 0.707
    wn = 3/(settling_time*damping_coefficient)

    SET_PARAMS.P_k = np.eye(7)
    SET_PARAMS.R_k = np.eye(3)*1e-4
    SET_PARAMS.Q_k = np.eye(7)*0.01

    SET_PARAMS.Kp = 2 * wn**2
    SET_PARAMS.Kd = 2 * damping_coefficient * wn
    SET_PARAMS.Kw = SET_PARAMS.Kp*1e-3 #! I just changed this from e-6 to e-5 to e-4

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
        Stellar = Constellation.Constellation(SET_PARAMS.Number_of_satellites)
        Overall_data = []

        for sat_num in range(SET_PARAMS.Number_of_satellites):
            Stellar.initiate_satellite(sat_num)

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
                    data = [Stellar.data[item] for item in Stellar.nearest_neighbours_all[sat_num]]
                    # Ensure that predictions is a dictionary
                    predictions = Stellar.FD.Per_Timestep(data, Stellar.FD_strategy, Stellar.nearest_neighbours_all[sat_num])
                    ###############################################################################
                    # USE THE VOTE OF EACH SATELLITE TO DETERMINE THE HEALTH OF ANOTHER SATELLITE #
                    ###############################################################################
                    for sat in Stellar.nearest_neighbours_all[sat_num]:
                        Stellar.fault_vote[sat] = predictions[sat]
            
            Overall_data.append(pd.DataFrame.from_dict(Stellar.data))

        Data = pd.concat(Overall_data)
        save_as_csv(Data, filename = "Constellation", index = "Satellite_number")

    elif SET_PARAMS.Number_of_multiple_orbits == 1:
        FD = Fault_detection.Basic_detection()
        for i in range(1, SET_PARAMS.Number_of_multiple_orbits + 1):
            D = Single_Satellite(i, s_list, t_list, J_t, fr)

            print(SET_PARAMS.Fault_names_values[i])

            loop(i, D, SET_PARAMS)

    ######################################################
    # IF THE SAVE AS IS NOT EQUAL TO XLSX, THE THREADING #
    #           CAN BE USED TO SAVE CSV FILES            #
    ######################################################
    else:
        numProcess = 0
        threads = []
        for extraction in featureExtractionMethods:
            for prediction in predictionMethods:
                for isolation in isolationMethods:
                    for recovery in recoveryMethods:

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
                        for i in range(2, SET_PARAMS.Number_of_multiple_orbits+1):
                            numProcess += 1
                            D = Single_Satellite(i, s_list, t_list, J_t, fr)

                            t = multiprocessing.Process(target=loop, args=(i, D, SET_PARAMS))
                            threads.append(t)
                            t.start()
                            print("Beginning of", extraction, prediction, isolation, recovery, i)


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