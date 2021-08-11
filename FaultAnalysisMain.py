import numpy as np
from Simulation.Parameters import SET_PARAMS
from Fault_prediction.Fault_utils import Dataset_order
import sys
np.set_printoptions(threshold=sys.maxsize)

SET_PARAMS.load_as == ".csv"
Plot = True
# Firstly the data must be extracted from the csv file. 
# Afterwards the DMD operations must be executed.
Y, Y_buffer, X, X_buffer, Orbit = Dataset_order("None", binary_set = True, buffer = False, categorical_num = False)

fromTimeStep = 0
ToTimeStep = 1000

sensor_number = 4

t = len(Y) - ToTimeStep - fromTimeStep
# Select the data for the sensor of interest
x1 = X[fromTimeStep:ToTimeStep-1,3*sensor_number:3*sensor_number + 3].T

# Select the data for the sensors that will be used to predict the next step of sensor x
x2 = X[fromTimeStep + 1:ToTimeStep,3*sensor_number:3*sensor_number + 3].T

# Select y, which impacts the vector x (Control DMD)
y1 = np.roll(X, 3*sensor_number)[fromTimeStep:ToTimeStep-1,3:].T

y2 = np.roll(X, 3*sensor_number)[fromTimeStep:ToTimeStep,3:].T

#! Without control matrix B
#! A = x2 @ np.linalg.pinv(x1)
#! x2_approximate = A @ x1

G = x2 @ np.linalg.pinv(np.concatenate((x1, y1)))

A = G[:,:3]
B = G[:,3:]

x2_approximate = A @ x1 + B @ y1

maximum = np.max(np.abs(x2 - x2_approximate))

print(maximum)

"""
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

"""