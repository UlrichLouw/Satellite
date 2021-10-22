import pickle
from Fault_prediction.Fault_utils import Dataset_order
from sklearn import tree
from sklearn.metrics import confusion_matrix
import numpy as np
from Simulation.Parameters import SET_PARAMS
from pathlib import Path
import matplotlib.pyplot as plt

def DecisionTreeAllAnomalies(path, depth, multi_class = False):
    X_list = []
    Y_list = []

    for index in range(SET_PARAMS.number_of_faults):
        name = SET_PARAMS.Fault_names_values[index+1]
        if multi_class:
            Y, _, X, _, _ = Dataset_order(name, binary_set = False, categorical_num = True, buffer = False)
        else:
            Y, _, X, _, _ = Dataset_order(name, binary_set = True, buffer = False, categorical_num = False)
        X_list.append(X)    
        Y_list.append(Y)

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)

    # Beform a decision tree on the X and Y matrices
    # This must however include the moving average
    clf = tree.DecisionTreeClassifier(max_depth = depth)

    # Split data into training and testing data
    mask = np.random.rand(len(X)) <= 0.6
    training_data = X[mask]
    testing_data = X[~mask]

    training_Y = Y[mask]
    testing_Y = Y[~mask]

    clf = clf.fit(training_data,training_Y)

    predict_y = clf.predict(testing_data)

    cm = confusion_matrix(testing_Y, predict_y)

    print(cm)

    path_to_folder = Path(path)
    path_to_folder.mkdir(exist_ok=True)

    if multi_class:
        pickle.dump(clf, open(path + '/DecisionTreesPhysicsEnabledDMDMultiClass.sav', 'wb'))
        if SET_PARAMS.Visualize:
            fig = plt.figure(figsize=(25,20))
            tree.plot_tree(clf,
                            filled=True, max_depth = 2)
            fig.savefig(path + '/DecisionTreeMultiClass.png')
    else:
        pickle.dump(clf, open(path + '/DecisionTreesPhysicsEnabledDMDBinaryClass.sav', 'wb'))

        if SET_PARAMS.Visualize:
            fig = plt.figure(figsize=(25,20))
            tree.plot_tree(clf,
                            filled=True, max_depth = 2)
            fig.savefig(path + '/DecisionTreeBinaryClass.png')