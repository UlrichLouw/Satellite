from re import S
import matplotlib
from Simulation.Parameters import SET_PARAMS
import pandas as pd
from pathlib import Path
import numpy as np

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

def SummaryPlots(RecoveryBuffer, predictionBuffer, perfectNoFailurePrediction, includeNone, bbox_to_anchor, loc, plotColumns, featureExtractionMethods, predictionMethods, isolationMethods, recoveryMethods, recoverMethodsWithoutPrediction, index, Number, Number_of_orbits = 30, ALL = True, first = False, width = 8.0, height = 6.0, groupBy = "Recovery", uniqueTag = ""):
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

    cm = 1/2.54

    path_of_execution = str(Path(__file__).parent.resolve()).split("/Satellite")[0] + "/Journal articles/My journal articles/Journal articles/Robust Kalman Filter/Figures/TexFigures"

    Path(path_of_execution).mkdir(parents = True, exist_ok=True)

    plt = matplotlib.pyplot

    for name in plotColumns:
        if groupBy == "Recovery":
            for recovery in recoveryMethods:

                legendLines = []
                legendNames = []

                path = "Data files/Summary/" + name + "/" + SET_PARAMS.Fault_names_values[index] + ".csv"

                Datapgf = GetData(path, index, n = Number, all = ALL, first = first) 

                Data = Datapgf[Datapgf.columns[Datapgf.iloc[1] == "Mean"]].copy()

                Data["Unnamed: 0"] = Datapgf["Unnamed: 0"].copy()

                plt.figure(figsize = (width*cm, height*cm))
                
                plt.grid(visible = True, which = 'both')

                plt.xlabel("Number of Orbits", fontsize = int(width))

                # plt.title(name + " for various methods", fontsize = int(width*1.2))

                if "Metric" in name:
                    plt.ylabel("$\\theta$ (deg)", fontsize = int(width))
                else:
                    plt.ylabel("Accuracy", fontsize = int(width))

                for extraction in featureExtractionMethods:

                    for isolation in isolationMethods:
                        for prediction in predictionMethods:
                            if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None") or (prediction != "None" and isolation != "None" and recovery not in recoverMethodsWithoutPrediction):
                                
                                method = extraction + str(prediction) + str(isolation) + recovery + SET_PARAMS.Fault_names_values[index]
                                if predictionBuffer:
                                    method += RecoveryBuffer

                                if perfectNoFailurePrediction:
                                    method += str("perfectNoFailurePrediction")

                                print(method)

                                plotData = [float(x) for x in Data[Data["Unnamed: 0"] == method].values[0] if x != method]

                                legendLines.append(plt.plot(range(len(plotData)), plotData))
                                legendNames.append(str(prediction))

                if includeNone:
                    method = "DMDNoneNoneNone" + SET_PARAMS.Fault_names_values[index]

                    plotData = [float(x) for x in Data[Data["Unnamed: 0"] == method].values[0] if x != method]

                    legendLines.append(plt.plot(range(len(plotData)), plotData))
                    legendNames.append(str("None"))

                if len(legendNames) > 1:
                    plt.legend(legendNames, loc = loc, fontsize = int(width), bbox_to_anchor=bbox_to_anchor, handlelength = 0.2)

                plt.tight_layout()

                path = path_of_execution + "/Summary/" + str(recovery)

                Path(path).mkdir(parents = True, exist_ok=True)

                # plt.show()

                plt.savefig(Path(path + "/" + name + uniqueTag+ '.pgf'))

                plt.close()

        elif groupBy == "Prediction":
            for prediction in predictionMethods:
                legendLines = []
                legendNames = []

                path = "Data files/Summary/" + name + "/" + SET_PARAMS.Fault_names_values[index] + ".csv"

                Datapgf = GetData(path, index, n = Number, all = ALL, first = first) 

                Data = Datapgf[Datapgf.columns[Datapgf.iloc[1] == "Mean"]].copy()

                Data["Unnamed: 0"] = Datapgf["Unnamed: 0"].copy()

                plt.figure(figsize = (width*cm, height*cm))
                
                plt.grid(visible = True, which = 'both')

                plt.xlabel("Number of Orbits", fontsize = int(width))

                # plt.title(name + " for various methods", fontsize = int(width*1.2))

                if "Metric" in name:
                    plt.ylabel("$\\theta$ (deg)", fontsize = int(width))
                else:
                    plt.ylabel("Accuracy", fontsize = int(width))

                for extraction in featureExtractionMethods:

                    for isolation in isolationMethods:
                        for recovery in recoveryMethods:
                            if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None") or (prediction != "None" and isolation != "None" and recovery not in recoverMethodsWithoutPrediction):
                                
                                method = extraction + str(prediction) + str(isolation) + recovery + SET_PARAMS.Fault_names_values[index] 
                                if predictionBuffer:
                                    method += RecoveryBuffer

                                if perfectNoFailurePrediction:
                                    method += str("perfectNoFailurePrediction")

                                plotData = [float(x) for x in Data[Data["Unnamed: 0"] == method].values[0] if x != method]

                                legendLines.append(plt.plot(range(len(plotData)), plotData))
                                legendNames.append(str(recovery))

                if includeNone:
                    method = "DMDNoneNoneNone" + SET_PARAMS.Fault_names_values[index]

                    plotData = [float(x) for x in Data[Data["Unnamed: 0"] == method].values[0] if x != method]

                    legendLines.append(plt.plot(range(len(plotData)), plotData))
                    legendNames.append(str("None"))

                if len(legendNames) > 1:
                    plt.legend(legendNames, loc = loc, fontsize = int(width), bbox_to_anchor=bbox_to_anchor, handlelength = 0.2)

                plt.tight_layout()

                path = path_of_execution + "/Summary/" + str(prediction)

                Path(path).mkdir(parents = True, exist_ok=True)

                # plt.show()

                plt.savefig(Path(path + "/" + name + uniqueTag + '.pgf'))

                plt.close()

        elif groupBy == "Isolation":
            for isolation in isolationMethods:
                legendLines = []
                legendNames = []

                path = "Data files/Summary/" + name + "/" + SET_PARAMS.Fault_names_values[index] + ".csv"

                Datapgf = GetData(path, index, n = Number, all = ALL, first = first) 

                Data = Datapgf[Datapgf.columns[Datapgf.iloc[1] == "Mean"]].copy()

                Data["Unnamed: 0"] = Datapgf["Unnamed: 0"].copy()

                plt.figure(figsize = (width*cm, height*cm))
                
                plt.grid(visible = True, which = 'both')

                plt.xlabel("Number of Orbits", fontsize = int(width))

                # plt.title(name + " for various methods", fontsize = int(width*1.2))

                if "Metric" in name:
                    plt.ylabel("$\\theta$ (deg)", fontsize = int(width))
                else:
                    plt.ylabel("Accuracy", fontsize = int(width))

                for extraction in featureExtractionMethods:

                    for recovery in recoveryMethods:
                        for prediction in predictionMethods:
                            if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None") or (prediction != "None" and isolation != "None" and recovery not in recoverMethodsWithoutPrediction):
                                
                                method = extraction + str(prediction) + str(isolation) + recovery + SET_PARAMS.Fault_names_values[index] 

                                if predictionBuffer:
                                    method += RecoveryBuffer

                                if perfectNoFailurePrediction:
                                    method += str("perfectNoFailurePrediction")

                                plotData = [float(x) for x in Data[Data["Unnamed: 0"] == method].values[0] if x != method]

                                legendLines.append(plt.plot(range(len(plotData)), plotData))
                                legendNames.append(str(prediction))

                if includeNone:
                    method = "DMDNoneNoneNone" + SET_PARAMS.Fault_names_values[index]

                    plotData = [float(x) for x in Data[Data["Unnamed: 0"] == method].values[0] if x != method]

                    legendLines.append(plt.plot(range(len(plotData)), plotData))
                    legendNames.append(str("None"))


                if len(legendNames) > 1:
                    plt.legend(legendNames, loc = loc, fontsize = int(width), bbox_to_anchor=bbox_to_anchor, handlelength = 0.2)

                plt.tight_layout()

                path = path_of_execution + "/Summary/" + str(isolation)

                Path(path).mkdir(parents = True, exist_ok=True)

                # plt.show()

                plt.savefig(Path(path + "/" + name + uniqueTag + '.pgf'))

                plt.close()