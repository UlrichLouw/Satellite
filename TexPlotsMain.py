import LatexPlots.Metrics as Metrics
import LatexPlots.Vectors as Vectors
import LatexPlots.Summary as Summary


if __name__ == '__main__':
    featureExtractionMethods = ["DMD"]
    predictionMethod = ["None", "DecisionTrees", "RandomForest"]
    isolationMethods = ["None", "OnlySun"]
    recoveryMethods = ["EKF-top2"] #, "EKF-combination"] #,  "EKF-top2"
    recoverMethodsWithoutPrediction = ["EKF-top2"]
    plotColumns = ["Prediction Accuracy", "Estimation Metric", "Pointing Metric"] #"Prediction Accuracy", 
    treeDepth = [5, 10, 20, 100]

    RecoveryBuffer = "EKF-top2"
    PredictionBuffer = False

    perfectNoFailurePrediction = False

    groupBy = "Recovery"
    tag = ""
    includeNone = False
    bbox_to_anchor = (0.5, 0.5, 0.5, 0.5)

    loc = 1

    predictionMethods = []

    for prediction in predictionMethod:
        if prediction == "DecisionTrees" or prediction == "RandomForest":
            for depth in treeDepth:
                predictionMethods.append(prediction + str(depth))
        else:
            predictionMethods.append(prediction)

    # Metrics.MetricPlots(RecoveryBuffer, PredictionBuffer, perfectNoFailurePrediction, bbox_to_anchor, loc, featureExtractionMethods, predictionMethods, isolationMethods, recoveryMethods, recoverMethodsWithoutPrediction, index = 2, Number = 5, Number_of_orbits = 30, first = True, ALL = False, width = 8.0, height = 6.0)
    # Vectors.VectorPlots(bbox_to_anchor, loc, featureExtractionMethods, predictionMethods, isolationMethods, recoveryMethods, recoverMethodsWithoutPrediction, index = 2, Number = 2, Number_of_orbits = 30, first = True, ALL = False, width = 8.0, height = 6.0)
    Summary.SummaryPlots(RecoveryBuffer, PredictionBuffer, perfectNoFailurePrediction, includeNone, bbox_to_anchor, loc, plotColumns, featureExtractionMethods, predictionMethods, isolationMethods, recoveryMethods, recoverMethodsWithoutPrediction, index = 2, Number = 30, Number_of_orbits = 30, first = True, ALL = True, width = 8.0, height = 6.0, groupBy = groupBy, uniqueTag = tag)