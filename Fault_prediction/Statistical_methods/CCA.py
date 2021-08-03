# Imports
import numpy as np
import rcca
from Parameters import SET_PARAMS
from Fault_utils import Dataset_order

# Initialize number of samples
nSamples = 1000
confusion_matrices = []
All_orbits = []
X_buffer = []
Y_buffer = []
buffer = False
binary_set = True
use_previously_saved_models = False
categorical_num = True

if __name__ == "__main__":
    # Create two datasets, with each dimension composed as a sum of 75% one of the latent variables and 25% independent component
    for index in range(SET_PARAMS.Number_of_multiple_orbits):
        Y, Y_buffer, X, X_buffer, Orbit = Dataset_order(index, direction, binary_set, buffer, categorical_num, use_previously_saved_models, 
                                                        columns_compare = ["Earth x", "Earth y", "Earth z"], 
                                                        columns_compare_to = ["Angular momentum of wheels x", "Angular momentum of wheels y", "Angular momentum of wheels z"])
        All_orbits.append(Orbit)

        # Split each dataset into two halves: training set and test set
        train1 = Y[:int(nSamples/2)]
        train2 = X[:int(nSamples/2)]
        test1 = Y[int(nSamples/2):]
        test2 = X[int(nSamples/2):]

        # Create a cca object as an instantiation of the CCA object class. 
        cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 2)

        # Use the train() method to find a CCA mapping between the two training sets.
        cca.train([train1, train2])

        # Use the validate() method to test how well the CCA mapping generalizes to the test data.
        # For each dimension in the test data, correlations between predicted and actual data are computed.
        testcorrs = cca.validate([test1, test2])
        print(testcorrs)