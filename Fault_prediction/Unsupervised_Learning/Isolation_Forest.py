import numpy as np
# import eif as iso
import pandas as pd
from Simulation.Parameters import SET_PARAMS
from Fault_prediction.Fault_utils import Dataset_order
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

def getDepth(x, root, d):
    n = root.n
    p = root.p
    if root.ntype == 'exNode':
        return d
    else:
        if (x-p).dot(n) < 0:
            return getDepth(x,root.left,d+1)
        else:
            return getDepth(x,root.right,d+1)
        
def getVals(forest,x,sorted=True):
    theta = np.linspace(0,2*np.pi, forest.ntrees)
    r = []
    for i in range(forest.ntrees):
        temp = forest.compute_paths_single_tree(np.array([x]),i)
        r.append(temp[0])
    if sorted:
        r = np.sort(np.array(r))
    return r, theta

def IsoForest(path, depth, multi_class = False, constellation = False, lowPredictionAccuracy = False, MovingAverage = True, includeAngularMomemntumSensors = False):
    X_list = []
    Y_list = []

    pathFiles = SET_PARAMS.path

    buffer = True
    SET_PARAMS.buffer_size = 2

    if constellation:
        for satNum in range(SET_PARAMS.Number_of_satellites):
            print(satNum)
            SET_PARAMS.path = pathFiles + str(satNum) + "/"
            for index in range(SET_PARAMS.number_of_faults):
                name = SET_PARAMS.Fault_names_values[index+1]
                if multi_class:
                    Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = False, categorical_num = True, buffer = buffer, constellation = constellation, multi_class = True, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors)
                else:
                    Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = True, buffer = buffer, categorical_num = False, constellation = constellation, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors)
                X_list.append(X)    
                Y_list.append(Y)

    else:
        for index in range(SET_PARAMS.number_of_faults):
            name = SET_PARAMS.Fault_names_values[index+1]
            if multi_class:
                Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = False, categorical_num = True, buffer = buffer, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors)
            else:
                Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = True, buffer = buffer, categorical_num = False, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors)
            X_list.append(X)    
            Y_list.append(Y)

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)

    random_state = np.random.RandomState(42)

    model=IsolationForest(n_estimators=10,max_samples='auto',contamination=0.2,random_state=random_state, bootstrap = True)

    print(X.shape)

    model.fit(X)

    print(model.get_params())

    scores = model.decision_function(X)

    anomaly_score = model.predict(X)

    Y = Y *-2

    Y = Y + 1

    step = []

    initVal = Y[0]

    for val in Y:
        if val != initVal:
            step.append(-1)
        else:
            step.append(1)
        initVal = val

    step = np.array(step)

    accuracy = 100*list(anomaly_score).count(-1)/(np.count_nonzero(Y==-1))
    print("Accuracy of the model:", accuracy)

    cm = confusion_matrix(anomaly_score, Y)

    print(cm)

    cm = confusion_matrix(anomaly_score, step)

    print(cm)

    print(list(anomaly_score).count(-1))

    fromIndex = 250000

    plt.figure()
    plt.plot(range(len(Y[fromIndex:])), Y[fromIndex:], 'b', alpha = 0.3)
    plt.plot(range(len(anomaly_score[fromIndex:])), anomaly_score[fromIndex:], 'r', alpha = 0.3)
    plt.show()

    pickle.dump(model, open('models/IsolationForest.sav', 'wb'))
    # F1 = iso.iForest(X, ntrees = 500, sample_size = 1000, ExtensionLevel=1)

    # xxx = np.array([[0,0.]])
    # SL0 = F1.compute_paths_single_tree(xxx, 0)

    # S1 = F1.compute_paths(X_in=X)

    # ss1=np.argsort(S1)

    # number_of_errors = np.sum(Y % 2 == 1)
    # print(np.sum(Y[ss1[:number_of_errors]])/number_of_errors, index)