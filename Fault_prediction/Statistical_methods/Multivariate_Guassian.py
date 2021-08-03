import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal#calculate the covariance matrix

# Initialize number of samples
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
    x = np.linspace(0, 5, 10, endpoint=False)
    for index in range(SET_PARAMS.Number_of_multiple_orbits):
        Y, Y_buffer, X, X_buffer, Orbit = Dataset_order(index, direction, binary_set, buffer, categorical_num, use_previously_saved_models, 
                                                        columns_compare = ["Earth x"], 
                                                        columns_compare_to = ["Angular momentum of wheels x"])
        All_orbits.append(Orbit)

        if use_previously_saved_models == False:
            #define x1 and x2 
            x1 = np.arange(1,50,1) 
            x2 = np.square(x1) + np.random.randint(-200,200)
            #adding outliers
            x1 = np.append(x1,17)
            x2 = np.append(x2,1300)
            data = np.stack((x1,x2),axis=1)
            plt.scatter(x1,x2)

            data = np.stack((x1,x2),axis=0)
            covariance_matrix = np.cov(data)

            #calculating the mean
            mean_values = [np.mean(x1),np.mean(x2)]

            #multivariate normal distribution
            model = multivariate_normal(cov=covariance_matrix,mean=mean_values)
            data = np.stack((x1,x2),axis=1)

            #finding the outliers
            threshold = 1.0e-07
            outlier = model.pdf(data).reshape(-1) < threshhold

            for boolean,i in enumerate(outlier):
                if i == True:
                    print(data[boolean]," is an Outlier")