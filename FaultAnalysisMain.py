import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydmd import DMD
from Simulation.Parameters import SET_PARAMS
from Fault_prediction.Fault_utils import Dataset_order
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
from numpy.linalg import inv, eig, pinv
from scipy.linalg import svd, svdvals
from scipy.integrate import odeint, ode, complex_ode
from warnings import warn

SET_PARAMS.load_as == ".csv"
Plot = True
# Firstly the data must be extracted from the csv file. 
# Afterwards the DMD operations must be executed.
Y, Y_buffer, X, X_buffer, Orbit = Dataset_order("None", binary_set = True, buffer = False, categorical_num = False)


t = len(Y) - 1
# Select the data for the sensor of interest
x1 = X[:-1,:3].T

# Select the data for the sensors that will be used to predict the next step of sensor x
x2 = X[1:,:3].T

# SVD of input matrix
U2,Sig2,Vh2 = svd(x1, False)

# based on Sig2 values truncation is set to 2
r = 2
U = U2[:,:r]

Sig = diag(Sig2)[:r,:r]
V = Vh2.conj().T[:,:r]

# build A tilde
Atil = dot(dot(dot(U.conj().T, x2), V), inv(Sig))
mu,W = eig(Atil)

radius = np.sqrt(mu[0].real**2 + mu[0].imag**2)

temp = np.linspace(0, 2*np.pi, 150)

circle_x = radius*cos(temp)
circle_y = radius*sin(temp)

# plot the complex numbers with radius around
plt.scatter(W.real, W.imag)
plt.plot(circle_x, circle_y)
plt.ylabel('Imaginary')
plt.xlabel('Real')
plt.show()

# build the DMD modes
Phi = dot(dot(dot(x2, V), inv(Sig)), W)

plt.plot(Phi)
plt.show()

# compute time evolution
b = dot(pinv(Phi), x1[:,0])
Psi = np.zeros([r, t], dtype='complex')
for i,_t in enumerate(range(t)):
    Psi[:,i] = multiply(power(mu, _t/1), b)

# compute DMD reconstruction
D2 = dot(Phi, Psi)
print(x1)
print(D2.real)
print(np.allclose(x1, D2.real)) # True

print(np.max(x1 - D2.real))