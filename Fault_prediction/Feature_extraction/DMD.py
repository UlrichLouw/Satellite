import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydmd import DMD

# Firstly the data must be extracted from the csv file. 
# Afterwards the DMD operations must be executed.

def f1(x,t): 
    return 1./np.cosh(x+3)*np.exp(2.3j*t)

def f2(x,t):
    return 2./np.cosh(x)*np.tanh(x)*np.exp(2.8j*t)

x = np.linspace(-5, 5, 128)

Data = pd.read_csv("Data files/None.csv")
t = range(len(Data.index))

xgrid, tgrid = np.meshgrid(x, t)

X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2



titles = ['$f_1(x,t)$', '$f_2(x,t)$', '$f$']
data = [X1, X2, X]

fig = plt.figure(figsize=(17,6))
for n, title, d in zip(range(131,134), titles, data):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, d.real)
    plt.title(title)
plt.colorbar()
plt.show()


"""
The dmd object contains the principal information about the decomposition:

    the attribute modes is a 2D numpy array where the columns are the low-rank structures individuated;
    the attribute dynamics is a 2D numpy array where the rows refer to the time evolution of each mode;
    the attribute eigs refers to the eigenvalues of the low dimensional operator;
    the attribute reconstructed_data refers to the approximated system evolution.

Moreover, some helpful methods for the graphical representation are provided.

Thanks to the eigenvalues, we can check if the modes are stable or not: if an eigenvalue is on the unit circle, the corresponding mode will be stable; while if an eigenvalue is inside or outside the unit circle, the mode will converge or diverge, respectively. From the following plot, we can note that the two modes are stable.
"""

dmd = DMD(svd_rank=2)
dmd.fit(X.T)

#####################################################
# FINALLY, WE CAN RECONSTRUCT THE ORIGINAL DATASET  #
# AS THE PRODUCT OF MODES AND DYNAMICS. WE PLOT     #
# THE EVOLUTION OF EACH MODE TO EMPHASIZE THEIR     #
# SIMILARITY WITH THE INPUT FUNCTIONS AND WE PLOT   #
# THE RECONSTRUCTED DATA.                           #
#####################################################

for mode in dmd.modes.T:
    plt.plot(x, mode.real)
    plt.title('Modes')
plt.show()

for dynamic in dmd.dynamics:
    plt.plot(t, dynamic.real)
    plt.title('Dynamics')
plt.show()



###########################################################################################
# WE CAN ALSO PLOT THE ABSOLUTE ERROR BETWEEN THE APPROXIMATED DATA AND THE ORIGINAL ONE. #
###########################################################################################

fig = plt.figure(figsize=(17,6))

for n, mode, dynamic in zip(range(131, 133), dmd.modes.T, dmd.dynamics):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, (mode.reshape(-1, 1).dot(dynamic.reshape(1, -1))).real.T)
    
plt.subplot(133)
plt.pcolor(xgrid, tgrid, dmd.reconstructed_data.T.real)
plt.colorbar()

plt.show()



####################################################################
# THE RECONSTRUCTED SYSTEM LOOKS ALMOST EQUAL TO THE ORIGINAL ONE: #
# THE DYNAMIC MODE DECOMPOSITION MADE POSSIBLE THE IDENTIFICATION  #
# OF THE MEANINGFUL STRUCTURES AND THE COMPLETE RECONSTRUCTION OF  #
#          THE SYSTEM USING ONLY THE COLLECTED SNAPSHOTS.          #
####################################################################


plt.pcolor(xgrid, tgrid, (X-dmd.reconstructed_data.T).real)
fig = plt.colorbar()



plt.pcolor(xgrid, tgrid, (X-dmd.reconstructed_data.T).real)
fig = plt.colorbar()



plt.pcolor(xgrid, tgrid, (X-dmd.reconstructed_data.T).real)
fig = plt.colorbar()

