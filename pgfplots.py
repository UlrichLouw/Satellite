import pandas as pd
from Simulation.Parameters import SET_PARAMS
from pathlib import Path
from Simulation.Save_display import visualize_data, save_as_csv, save_as_pickle


def GetData(path, index, n, all = False, first = False):
    Dataframe = pd.read_csv(path, low_memory=False)

    if all:
        Datapgf = Dataframe
    elif first:
        Datapgf = Dataframe[:int((n)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts))]
    else:
        Datapgf = Dataframe[int((SET_PARAMS.Number_of_orbits-n)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)):]

    DatapgfSensors = Datapgf.loc[:,Datapgf.columns.str.contains('Sun') | Datapgf.columns.str.contains('Magnetometer') |
                            Datapgf.columns.str.contains('Earth') | Datapgf.columns.str.contains('Angular momentum of wheels') |
                            Datapgf.columns.str.contains('Star')]
    
    DatapgfTorques = Datapgf.loc[:, Datapgf.columns.str.contains('Torques')]

    DatapgfKalmanFilter = Datapgf.loc[:,Datapgf.columns.str.contains('Quaternions') | Datapgf.columns.str.contains('Euler Angles') | Datapgf.columns.str.contains('Angular velocity of satellite')]

    DatapgfPrediction = Datapgf.loc[:,Datapgf.columns.str.contains('Accuracy') | Datapgf.columns.str.contains('fault')]

    DatapgfMetric = Datapgf.loc[:,Datapgf.columns.str.contains('Metric')]

    if SET_PARAMS.NumberOfRandom > 1:
        GenericPath = "Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/"+ SET_PARAMS.Model_or_Measured +"/" +\
            "SunSensorSize-Length:" + str(SET_PARAMS.Sun_sensor_length) + "-Width:" + str(SET_PARAMS.Sun_sensor_width) + "/" + "SolarPanel-Length:" + str(SET_PARAMS.SP_Length) + "-Width:" + str(SET_PARAMS.SP_width) + \
            "T-list:" + SET_PARAMS.t_list + "S-list:" + SET_PARAMS.s_list
    else:
        GenericPath = "Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/" + SET_PARAMS.Model_or_Measured +"/" + "General CubeSat Model/"

    path = "Data files/pgfPlots/" + GenericPath

    if all:
        path = path + "/All_"
    elif n > 1:
        path = path + "/" + str(n)

    if SET_PARAMS.save_as == ".csv":
        path_to_folder = Path(path + "Sensors/")
        path_to_folder.mkdir(parents = True, exist_ok=True)
        save_as_csv(DatapgfSensors, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path + "Sensors/")
    else:
        save_as_pickle(DatapgfSensors, index)

    if SET_PARAMS.save_as == ".csv":
        path_to_folder = Path(path + "Torques/")
        path_to_folder.mkdir(parents = True, exist_ok=True)
        save_as_csv(DatapgfTorques, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path + "Torques/")
    else:
        save_as_pickle(DatapgfTorques, index)

    if SET_PARAMS.save_as == ".csv":
        path_to_folder = Path(path + "KalmanFilter/")
        path_to_folder.mkdir(parents = True, exist_ok=True)
        save_as_csv(DatapgfKalmanFilter, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path + "KalmanFilter/")
    else:
        save_as_pickle(DatapgfKalmanFilter, index)

    if SET_PARAMS.save_as == ".csv":
        path_to_folder = Path(path + "Prediction/")
        path_to_folder.mkdir(parents = True, exist_ok=True)
        save_as_csv(DatapgfPrediction, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path + "Prediction/")
    else:
        save_as_pickle(DatapgfPrediction, index)

    if SET_PARAMS.save_as == ".csv":
        path_to_folder = Path(path + "Metric/")
        path_to_folder.mkdir(parents = True, exist_ok=True)
        save_as_csv(DatapgfMetric, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path + "Metric/")
    else:
        save_as_pickle(DatapgfMetric, index)

if __name__ == "__main__":
    featureExtractionMethods = ["DMD"]
    # predictionMethods = ["RandomForest"]
    # isolationMethods = ["RandomForest"] #! "RandomForest", 
    # recoveryMethods = ["EKF-combination", "EKF-reset", "EKF-ignore"]
    # recoverMethodsWithoutPrediction = ["None", "EKF-top3"]
    predictionMethods = ["None"]
    isolationMethods = ["None"] #! "RandomForest", 
    recoveryMethods = ["None"]
    recoverMethodsWithoutPrediction = ["None"]
    SET_PARAMS.Mode = "EARTH_SUN"
    SET_PARAMS.Model_or_Measured = "Model"
    SET_PARAMS.Number_of_orbits = 25
    SET_PARAMS.save_as = ".csv"
    SET_PARAMS.Low_Aerodynamic_Disturbance = True
    index = 1
    Number = 15
    ALL = False
    first = True

    includeNone = True

    for extraction in featureExtractionMethods:
        for prediction in predictionMethods:
            for isolation in isolationMethods:
                for recovery in recoveryMethods:
                    if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None") or (prediction == isolation and prediction != "None" and recovery not in recoverMethodsWithoutPrediction):
                        SET_PARAMS.FeatureExtraction = extraction
                        SET_PARAMS.SensorPredictor = prediction
                        SET_PARAMS.SensorIsolator = isolation
                        SET_PARAMS.SensorRecoveror = recovery
                        GenericPath = "Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/" + SET_PARAMS.Model_or_Measured +"/"+ "General CubeSat Model/"
                        
                        if SET_PARAMS.Low_Aerodynamic_Disturbance:
                            GenericPath = "Low_Disturbance/" + GenericPath
                        
                        path = "Data files/"+ GenericPath + SET_PARAMS.Fault_names_values[index] + ".csv"
                        path = Path(path)
                        GetData(path, index, n = Number, all = ALL, first = first) 

    SET_PARAMS.FeatureExtraction = "EKF"
    SET_PARAMS.SensorPredictor = "None"
    SET_PARAMS.SensorIsolator = "None"
    SET_PARAMS.SensorRecoveror = "None"
    GenericPath = "Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/" + SET_PARAMS.Model_or_Measured +"/"+ "General CubeSat Model/"
    path = "Data files/"+ GenericPath + SET_PARAMS.Fault_names_values[index] + ".csv"
    path = Path(path)
    GetData(path, index, n = Number, all = ALL, first = first)