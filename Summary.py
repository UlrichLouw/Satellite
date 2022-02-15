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
        df = df[[nameList]]
        Dataframe.append(df)
    ####################################################
    #     IF THERE IS NO EXISTING CSV FILES AND NO     #
    # PREVIOUS VERSIONS IGNORE IMPORT PREVIOUS VERSION #
    ####################################################
    
    return Dataframe

def dataFrameToLatex():
    pass

def SaveSummary(path, method, recovery, prediction, col):
    
    DataFrames = GetData(path, col)

    meanList, stdList, columns = [], [], []

    columns.append(("Orbits", "Detection Strategy", "Detection Strategy"))
    columns.append(("Orbits", "Recovery Strategy", "Recovery Strategy"))

    for num in range(1,SET_PARAMS.Number_of_orbits+1):
        columns.append((num, "Metric ($\theta$)", 'Mean'))
        columns.append((num, "Metric ($\theta$)", 'Std'))

    columns = pd.MultiIndex.from_tuples(columns)

    df = pd.DataFrame(columns = columns, index = [method])

    df.loc[method, ("Orbits", "Detection Strategy", "Detection Strategy")] = prediction

    df.loc[method, ("Orbits", "Recovery Strategy", "Recovery Strategy")] = recovery
    
    for orbit in range(1,SET_PARAMS.Number_of_orbits+1):
        for DataFrame in DataFrames:
            DF = DataFrame[int((orbit-1)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)):int((orbit)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts))]
            
            Metric = DF[col]
            meanList.append(Metric.mean())
            stdList.append(Metric.std())

        df.loc[method, (orbit,"Metric ($\theta$)",'Mean')] = sum(meanList)/len(meanList)
        df.loc[method, (orbit,"Metric ($\theta$)",'Std')] = sum(stdList)/len(stdList)

    return df


if __name__ == "__main__":
    featureExtractionMethods = ["DMD"]
    predictionMethods = ["DecisionTrees","RandomForest"] #! "DecisionTrees","RandomForest", "PERFECT", "RandomChoice"
    isolationMethods = ["DecisionTrees","RandomForest"] #! "RandomForest", 
    recoveryMethods = ["EKF-ignore"] # ["
    # predictionMethods = ["RandomForest"]
    # isolationMethods = ["RandomForest"] #! "RandomForest", 
    # recoveryMethods = ["EKF-replacement"]
    SET_PARAMS.Mode = "EARTH_SUN"
    SET_PARAMS.Model_or_Measured = "ORC"
    SET_PARAMS.Number_of_orbits = 5
    index = 2

    dfList = []

    nameList = ["Pointing Metric"] #!, "Estimation Metric", "Prediction Accuracy"]

    path_of_execution = str(Path(__file__).parent.resolve()).split("/Satellite")[0] + "/Journal articles/My journal articles/Journal articles/Robust Kalman Filter/Tables"

    Path(path_of_execution).mkdir(parents = True, exist_ok=True)

    orbitsToLatex = [1, 2, 3, 4, 5, 30]

    for name in nameList:
        for extraction in featureExtractionMethods:
            for prediction in predictionMethods:
                for isolation in isolationMethods:
                    for recovery in recoveryMethods:
                        if prediction == isolation:
                            SET_PARAMS.FeatureExtraction = extraction
                            SET_PARAMS.SensorPredictor = prediction
                            SET_PARAMS.SensorIsolator = isolation
                            SET_PARAMS.SensorRecoveror = recovery
                            GenericPath = "Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/" + SET_PARAMS.Model_or_Measured +"/" + "General CubeSat Model/"
                            path = "Data files/"+ GenericPath #+ SET_PARAMS.Fault_names_values[index] 
                            method = extraction + prediction + isolation + recovery
                            print("Begin: " + method)
                            path = Path(path)
                            dataFrame = SaveSummary(path, method, recovery, prediction, name)
                            dfList.append(dataFrame.copy())
                            print(method)

        SET_PARAMS.FeatureExtraction = "None"
        SET_PARAMS.SensorPredictor = "None"
        SET_PARAMS.SensorIsolator = "None"
        SET_PARAMS.SensorRecoveror = "None"
        GenericPath = "Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/"+SET_PARAMS.Model_or_Measured +"/" + "General CubeSat Model/"
        path = "Data files/"+ GenericPath #+ SET_PARAMS.Fault_names_values[index]
        method = "DMD" + "None" + "None" + "None"
        print("Begin: " + method)
        path = Path(path)
        dataFrame = SaveSummary(path, method, recovery, prediction, name)
        dfList.append(dataFrame.copy())
        print(method)

        dataFrame = pd.concat(dfList)
        # dataFrame.set_index(["Detection Strategy", "Recovery Strategy"], inplace = True, append = True, drop = False)
        path = "Data files/Summary/" + name + "/"

        path_to_folder = Path(path)
        path_to_folder.mkdir(parents = True, exist_ok=True)
        save_as_csv(dataFrame, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path,  float_format="%.2f")

        for orbit in range(1,SET_PARAMS.Number_of_orbits+1):
            if orbit not in orbitsToLatex:
                dataFrame = dataFrame.loc[:, ~(orbit,"Metric ($\theta$)",'Mean')]
                dataFrame = dataFrame.loc[:, ~(orbit,"Metric ($\theta$)",'Std')]

        # with open(
        #     Path(path_of_execution + "/Reflection.tex"), "w"
        # ) as tf:
        #     tf.write(dataFrame.to_latex(index = False, float_format="{:0.2f}".format, escape = False))

        # dataFrame = dataFrame.stack()
        # dataFrame = dataFrame.flatten()

        dataFrame.to_latex(buf = Path(path_of_execution + "/Reflection.tex"), index = False, float_format="{:0.2f}".format, escape = False)