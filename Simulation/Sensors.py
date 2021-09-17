import numpy as np
import Simulation.Earth_model as Earth_model
from Simulation.Parameters import SET_PARAMS
from sgp4.api import Satrec, WGS72
from sgp4.api import jday
from skyfield.api import wgs84, EarthSatellite
import math
pi = math.pi


class Sensors:
    def __init__(self, s_list, t_list, J_t, fr):
        self.sat = Satrec()
        self.orbit = Earth_model.orbit()
        self.earth = Earth_model.Earth()
        self.satellite = self.sat.twoline2rv(s_list, t_list)
        e, self.r_sat, self.v_sat = self.satellite.sgp4(J_t, fr)  
        self.coordinates_to_earth = EarthSatellite(s_list, t_list)
        self.first = 0

    def sun(self, t):
        T_jc = (SET_PARAMS.J_t + SET_PARAMS.fr + t * 3.168808781403e-8 - 2452545)/36525
        M_o = 357.527723300 + 35999.050340*T_jc     #in degrees
        lambda_Mo = 280.460618400 + 36000.770053610*T_jc        #degrees
        lambda_e = lambda_Mo + 1.914666471*np.sin(M_o*pi/180) + 0.019994643*math.sin(2*M_o*pi/180)      #degrees
        epsilon =  23.439291 - 0.013004200*T_jc                 #degrees
        r_o = 1.00140612 - 0.016708617*np.cos(M_o*pi/180) - 0.000139589*np.cos(2*M_o*pi/180)        #degrees
        rsun = r_o * np.array(([np.cos(lambda_e*pi/180),np.cos(epsilon*pi/180)*np.sin(lambda_e*pi/180),np.sin(epsilon*pi/180)*np.sin(lambda_e*pi/180)]))
        rsun = rsun*(149597871)*1000
        norm_rsun = np.linalg.norm(rsun)
        S_EIC = rsun - self.r_sat_EIC
        norm_S_EIC = np.linalg.norm(S_EIC)
        norm_r_sat = max(np.linalg.norm(self.r_sat_EIC),SET_PARAMS.Radius_earth)
        theta_e = np.arcsin(SET_PARAMS.Radius_earth/norm_r_sat)
        theta_s = np.arcsin(SET_PARAMS.Radius_sun/norm_S_EIC)
        theta = np.arccos(np.dot(self.r_sat_EIC, rsun)/(norm_rsun*norm_r_sat))
        if (theta_e > theta_s) and (theta < (theta_e-theta_s)):
            self.in_sun_view = False
            S_EIC = np.zeros(3)
        else:
            self.in_sun_view = True
        return S_EIC, self.in_sun_view     #in m

    def magnetometer(self, t):
        self.latitude, self.longitude, self.altitude = Earth_model.ecef2lla(self.r_sat_EIC)
        B = self.earth.scalar_potential_function(self.latitude, self.longitude, self.altitude)
        B += np.random.normal(0,np.linalg.norm(B)*SET_PARAMS.Magnetometer_noise,B.shape)
        self.position = np.array([self.longitude[0][0], self.latitude[0][0], self.altitude[0][0]])

        return B

    def Earth(self, t):
        e, r_sat, v_sat = self.satellite.sgp4(SET_PARAMS.J_t, SET_PARAMS.fr + t/86400)
        self.r_sat_EIC = np.array((r_sat)) # convert r_sat to m
        self.v_sat_EIC = np.array((v_sat)) # v_sat to m/s
    
        self.A_EFC_to_EIC = self.orbit.EFC_to_EIC(t)
        self.r_sat_EFC = np.matmul(np.linalg.inv(self.A_EFC_to_EIC),self.r_sat_EIC)
        self.A_EIC_to_ORC = self.orbit.EIC_to_ORC(self.r_sat_EIC, self.v_sat_EIC)
        self.r_sat = np.matmul(self.A_EIC_to_ORC, self.r_sat_EIC)
        self.v_sat = np.matmul(self.A_EIC_to_ORC, self.v_sat_EIC)
        self.r_sat = self.r_sat/np.linalg.norm(self.r_sat)
        self.r_sat_EIC = self.r_sat_EIC*1000
        self.v_sat_EIC = self.v_sat_EIC*1000
        return self.r_sat, self.v_sat/np.linalg.norm(self.v_sat), self.A_EIC_to_ORC, self.r_sat_EIC
