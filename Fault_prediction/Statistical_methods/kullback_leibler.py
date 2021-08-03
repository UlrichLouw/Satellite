import numpy as np
from scipy.stats import norm
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
from Parameters import SET_PARAMS
from Fault_utils import Dataset_order
sns.set

# Initialize number of samples
confusion_matrices = []
All_orbits = []
X_buffer = []
Y_buffer = []
buffer = False
binary_set = True
use_previously_saved_models = False
categorical_num = True

def kl_divergence(p,q):
    return np.sum(np.where(p != 0, p * np.log(p/q), 0))

if __name__ == "__main__":
    # Create two datasets, with each dimension composed as a sum of 75% one of the latent variables and 25% independent component
    x = np.arange(-10, 10 , 0.001)
    for index in range(SET_PARAMS.Number_of_multiple_orbits):
        Y, Y_buffer, X, X_buffer, Orbit = Dataset_order(index, direction, binary_set, buffer, categorical_num, use_previously_saved_models, 
                                                        columns_compare = ["Earth x"], 
                                                        columns_compare_to = ["Angular momentum of wheels x"])
        All_orbits.append(Orbit)

        if use_previously_saved_models == False:
            p = norm.pdf(x, np.linalg.norm(X), np.std(X))
            q = norm.pdf(x, np.linalg.norm(Y), np.std(Y))

            plt.plot('KL(P||Q) = %1.3f' %kl_divergence(p,q))
            plt.plot(x,p)
            plt.plot(x,q, c = 'red')
            plt.show()