from glob import escape
import matplotlib
from Simulation.Parameters import SET_PARAMS
import pandas as pd
from pathlib import Path

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def GetData(path, index, n, all = False, first = False):
    Dataframe = pd.read_csv(path, low_memory=False)

    if all:
        Datapgf = Dataframe
    elif first:
        Datapgf = Dataframe[:int((n)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts))]
    else:
        Datapgf = Dataframe[int((SET_PARAMS.Number_of_orbits-n)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)):]

    return Datapgf

def MetricPlots(index, Number, Number_of_orbits = 30, ALL = True, first = False, width = 8.0, height = 6.0):
    featureExtractionMethods = ["DMD"]
    predictionMethods = ["None", "DecisionTrees", "RandomForest", "PERFECT", "Isolation_Forest"]
    isolationMethods = ["None", "OnlySun"] #! "RandomForest", "PERFECT",  
    recoveryMethods = ["None", "EKF-ignore"]
    recoverMethodsWithoutPrediction = ["None", "EKF-top3"]
    # predictionMethods = ["None"]
    # isolationMethods = ["None"] #! "RandomForest", 
    # recoveryMethods = ["None"]
    # recoverMethodsWithoutPrediction = ["None"]
    SET_PARAMS.Mode = "EARTH_SUN"
    SET_PARAMS.Model_or_Measured = "ORC"
    SET_PARAMS.Number_of_orbits = Number_of_orbits
    SET_PARAMS.save_as = ".csv"
    SET_PARAMS.Low_Aerodynamic_Disturbance = False

    if index == 1:
        predictionMethods = ["None"]
        isolationMethods = ["None"]
        recoveryMethods = ["None"]
        recoverMethodsWithoutPrediction = ["None"]

    includeNone = True

    cm = 1/2.54

    plotColumns = ["Prediction Accuracy", "Estimation Metric", "Pointing Metric"]
    # plotColumns = ["Estimation Metric"]

    path_of_execution = str(Path(__file__).parent.resolve()).split("/Satellite")[0] + "/Journal articles/My journal articles/Journal articles/Robust Kalman Filter/Figures/TexFigures"

    Path(path_of_execution).mkdir(parents = True, exist_ok=True)

    plt = matplotlib.pyplot

    for extraction in featureExtractionMethods:
        for prediction in predictionMethods:
            for isolation in isolationMethods:
                for recovery in recoveryMethods:
                    if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None") or (prediction != "None" and isolation != "None" and recovery not in recoverMethodsWithoutPrediction):
                        SET_PARAMS.FeatureExtraction = extraction
                        SET_PARAMS.SensorPredictor = prediction
                        SET_PARAMS.SensorIsolator = isolation
                        SET_PARAMS.SensorRecoveror = recovery
                        GenericPath = "Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/" + SET_PARAMS.Model_or_Measured +"/"+ "General CubeSat Model/"
                        
                        if SET_PARAMS.Low_Aerodynamic_Disturbance:
                            GenericPath = "Low_Disturbance/" + GenericPath
                        
                        path = "Data files/"+ GenericPath + SET_PARAMS.Fault_names_values[index] + ".csv"
                        path = Path(path)
                        Datapgf = GetData(path, index, n = Number, all = ALL, first = first) 
                        
                        currenctSunStatus = Datapgf.loc[0, "Sun in view"]

                        SunInView = []
                        Eclipse = []
                        
                        for ind in Datapgf.index:
                            if Datapgf.loc[ind, "Sun in view"] != currenctSunStatus:
                                if Datapgf.loc[ind, "Sun in view"]:
                                    SunInView.append(ind)
                                else:
                                    Eclipse.append(ind)
                            currenctSunStatus = Datapgf.loc[ind, "Sun in view"]

                        for col in plotColumns:
                            fig = plt.figure(figsize = (width*cm, height*cm))

                            plt.plot(range(len(Datapgf[[col]])), Datapgf[[col]])

                            if SET_PARAMS.Fault_names_values[index] == "None":
                                plt.title(col + " of Perfectly Designed Satellite", fontsize = int(width*1.2))
                            elif recovery == "None":
                                plt.title(col + " of Without Recovery", fontsize = int(width*1.2))

                            plt.grid(visible = True, which = 'both')

                            plt.xlabel("Time: (s)", fontsize = int(width))

                            if "Metric" in col:
                                plt.ylabel("$\\theta$ (deg)", fontsize = int(width))
                            else:
                                plt.ylabel("Accuracy", fontsize = int(width))

                            for sun in SunInView:
                                plt.axvline(x=sun, linestyle = '--', c = 'r', linewidth=0.4)

                            for eclipse in Eclipse:
                                plt.axvline(x=eclipse, linestyle = '--', c = 'k', linewidth=0.4)

                            plt.tight_layout()

                            path = path_of_execution + "/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"-" + SET_PARAMS.Model_or_Measured +"-General CubeSat Model/" + SET_PARAMS.Fault_names_values[index]
                            
                            Path(path).mkdir(parents = True, exist_ok=True)

                            plt.savefig(Path(path + "/" + col + '.pgf'))