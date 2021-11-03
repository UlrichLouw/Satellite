import pandas as pd
from Simulation.Parameters import SET_PARAMS
from pathlib import Path
from Simulation.Save_display import visualize_data, save_as_csv, save_as_pickle
import os
import glob
import numpy as np

def GetData(path, nameList):
    ###################################################################################################################################################################################
    # This is the globalVariables.pathOfWorkbook to the folder for the archived circuitDB
    excel_folder = os.path.join(path)
    ####################################################################################################################################################################################
    # glob.glob returns all files matching the pattern.
    excel_files = list(glob.glob(os.path.join(excel_folder, '*.csv*')))
    ####################################################################################################################################################################################
    # Append all the csv files in the Dataframe list 
    Dataframe = []

    for f in excel_files:
        df = pd.read_csv(f)
        df = df[nameList]
        Dataframe.append(df)
    ####################################################
    #     IF THERE IS NO EXISTING CSV FILES AND NO     #
    # PREVIOUS VERSIONS IGNORE IMPORT PREVIOUS VERSION #
    ####################################################
    
    return Dataframe

def SaveSummary(path, method):
    nameList = ["Pointing Accuracy", "Estimation Accuracy"]
    DataFrames = GetData(path, nameList)

    meanList, stdList = [], []

    columns = pd.MultiIndex.from_product([range(1,SET_PARAMS.Number_of_orbits+1), ['Mean','Std']], names = ['Orbits', 'Metric'])

    df = pd.DataFrame(columns = columns, index = [method])
    
    for orbit in range(1,SET_PARAMS.Number_of_orbits+1):
        for DataFrame in DataFrames:
            DF = DataFrame[int((orbit-1)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)):int((orbit)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts))]
            
            PA = DF["Pointing Accuracy"]
            EA = DF["Estimation Accuracy"]
            meanList.append(PA.mean())
            stdList.append(PA.std())

        df.loc[method, (orbit,'Mean')] = sum(meanList)/len(meanList)
        df.loc[method, (orbit,'Std')] = sum(stdList)/len(stdList)

    return df


if __name__ == "__main__":
    featureExtractionMethods = ["DMD"]
    predictionMethods = ["DecisionTrees", "PERFECT"]
    isolationMethods = ["DecisionTrees", "PERFECT"] #! "RandomForest", 
    recoveryMethods = ["EKF"]
    SET_PARAMS.Mode = "EARTH_SUN"
    SET_PARAMS.Number_of_orbits = 20
    index = 2

    dfList = []

    for extraction in featureExtractionMethods:
        for prediction in predictionMethods:
            for isolation in isolationMethods:
                for recovery in recoveryMethods:
                    SET_PARAMS.FeatureExtraction = extraction
                    SET_PARAMS.SensorPredictor = prediction
                    SET_PARAMS.SensorIsolator = isolation
                    SET_PARAMS.SensorRecoveror = recovery
                    GenericPath = "Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/"+"SunSensorSize-Length:0.03-Width:0.02/"
                    path = "Data files/"+ GenericPath
                    method = extraction + prediction + isolation + recovery
                    print("Begin: " + method)
                    path = Path(path)
                    dataFrame = SaveSummary(path, method)
                    dfList.append(dataFrame.copy())
                    print(method)

    SET_PARAMS.FeatureExtraction = "None"
    SET_PARAMS.SensorPredictor = "None"
    SET_PARAMS.SensorIsolator = "None"
    SET_PARAMS.SensorRecoveror = "None"
    GenericPath = "Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/"+"SunSensorSize-Length:0.03-Width:0.02/"
    path = "Data files/"+ GenericPath
    method = extraction + prediction + isolation + recovery
    print("Begin: " + method)
    path = Path(path)
    dataFrame = SaveSummary(path, method)
    dfList.append(dataFrame.copy())
    print(method)

    dataFrame = pd.concat(dfList)
    path = "Data files/Summary/"

    path_to_folder = Path(path)
    path_to_folder.mkdir(parents = True, exist_ok=True)
    save_as_csv(dataFrame, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path)
