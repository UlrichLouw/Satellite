from Simulation.Parameters import SET_PARAMS
import math
import numpy as np
import Simulation.igrf_utils as igrf_utils
from scipy import interpolate


IGRF_FILE = r'Simulation/Simulation_data/IGRF13.shc'
igrf = igrf_utils.load_shcfile(IGRF_FILE, None)
f = interpolate.interp1d(igrf.time, igrf.coeffs)

pi = math.pi

def ecef2lla(R):
    # x, y and z are scalars or vectors in meters
    x, y, z = R
    x = np.array([x]).reshape(np.array([x]).shape[-1], 1)
    y = np.array([y]).reshape(np.array([y]).shape[-1], 1)
    z = np.array([z]).reshape(np.array([z]).shape[-1], 1)

    a=6378137
    a_sq=a**2
    e = 8.181919084261345e-2
    e_sq = 6.69437999014e-3

    f = 1/298.257223563
    b = a*(1-f)

    # calculations:
    r = np.sqrt(x**2 + y**2)
    ep_sq  = (a**2-b**2)/b**2
    ee = (a**2-b**2)
    f = (54*b**2)*(z**2)
    g = r**2 + (1 - e_sq)*(z**2) - e_sq*ee*2
    c = (e_sq**2)*f*r**2/(g**3)
    s = (1 + c + np.sqrt(c**2 + 2*c))**(1/3.)
    p = f/(3.*(g**2)*(s + (1./s) + 1)**2)
    q = np.sqrt(1 + 2*p*e_sq**2)
    r_0 = -(p*e_sq*r)/(1+q) + np.sqrt(0.5*(a**2)*(1+(1./q)) - p*(z**2)*(1-e_sq)/(q*(1+q)) - 0.5*p*(r**2))
    u = np.sqrt((r - e_sq*r_0)**2 + z**2)
    v = np.sqrt((r - e_sq*r_0)**2 + (1 - e_sq)*z**2)
    z_0 = (b**2)*z/(a*v)
    h = u*(1 - b**2/(a*v))
    phi = np.arctan((z + ep_sq*z_0)/r)
    lambd = np.arctan2(y, x)


    return phi*180/np.pi, lambd*180/np.pi, h

class orbit:
    def __init__(self):
        self.w_earth = SET_PARAMS.w_earth
        self.a_G0 = SET_PARAMS.a_G0

    def EFC_to_EIC(self, t):
        a_G = self.w_earth * t + self.a_G0   # angle in radians form the greenwich
        A = np.array(([[np.cos(a_G), -np.sin(a_G), 0.0], [np.sin(a_G), np.cos(a_G), 0.0], [0.0,0.0,1.0]]))
        return A

    def EIC_to_ORC(self, position_vector, velocity_vector):
        # position vector - Height from center of earth, 
        position_vector = position_vector/np.linalg.norm(position_vector)
        velocity_vector = velocity_vector/np.linalg.norm(velocity_vector)
        c = -position_vector   # position vector must be measured by sensors
        b = np.cross(velocity_vector, position_vector)/(np.linalg.norm(np.cross(velocity_vector, position_vector)))
        a = np.cross(b,c)
        A = np.reshape(np.array(([a],[b],[c])),(3,3))
        return A

class Earth:   
    def __init__(self):
        self.V = np.zeros((2))
        self.first = 1      # This is required for the geomagnetic_field_strength_func to initiate correctly
        self.coeffs = f(2021) 

    def scalar_potential_function(self, latitude, longitude, altitude):
        rs = altitude[0,0]
        theta = 90 - latitude[0,0]
        lambda_ = longitude[0,0]
        k = SET_PARAMS.k
        B = igrf_utils.synth_values(self.coeffs, rs, theta, lambda_, 10, 3)
        B = np.array((B[0],B[1],B[2]))

        return B