import pandas as pd
import numpy as np
import collections
from Simulation.Parameters import SET_PARAMS
import pickle

######################################################
# BASIC DETECTION WITH THRESHOLDS AND IF STATEMENTS. #
# THE CURRENT METHOD OF FDIR SYSTEMNS ON SATELLITES  #
######################################################

class Basic_detection:
    def __init__(self):
        # Sun parameters
        self.sun_var_threshold = 0.4        

        # Earth parameters       
        self.earth_var_threshold = 0.4

        # Star parameters
        self.star_var_threshold = 0.4

        # Angular momentum parameters
        self.angular_var_threshold = 0.01 

        # Magetometer parameters
        self.magnetometer_var_threshold = 0.4

        self.sun_buffer = collections.deque(maxlen = SET_PARAMS.buffer_size)
        self.earth_buffer = collections.deque(maxlen = SET_PARAMS.buffer_size)
        self.star_buffer = collections.deque(maxlen = SET_PARAMS.buffer_size)
        self.magnetometer_buffer = collections.deque(maxlen = SET_PARAMS.buffer_size)
        self.angular_threshold = collections.deque(maxlen = SET_PARAMS.buffer_size)
        self.sensors = {"Sun": self.sun_buffer, 
                "Earth": self.earth_buffer, 
                "Star": self.star_buffer, 
                "Angular momentum of wheels": self.angular_threshold,
                "Magnetometer": self.magnetometer_buffer}

    def Per_Timestep(self, Data, Strategy):
        for sensor in self.sensors:
            self.sensors[sensor].append(Data[sensor][-1])
        self.Error = "None"
        self.sun_fault(self.sensors['Sun'])
        self.star_fault(self.sensors['Star'])
        self.earth_fault(self.sensors['Earth'])
        self.angular_momentum_fault(self.sensors['Angular momentum of wheels'])
        self.magnetometer_fault(self.sensors['Magnetometer'])
        return self.Error

    #####################################################
    # IF THE THRESHOLD OF THE SUN VECTOR IS LARGER THAT #
    #     A SPECIFIED VALUE THEN IT RETURN AN ERROR     #
    #####################################################

    def sun_fault(self, sun):
        sun = np.array((sun))
        current_sun = sun[0]
        mean_sun = np.mean(sun)
        var_sun = np.var(sun)
        norm_sun = np.linalg.norm(current_sun)

        if round(norm_sun,5) != 1 and round(norm_sun,5) != 0:
            self.Error = "SUN_BROKEN"

        if var_sun >= self.sun_var_threshold:
            self.Error = "SUN_BROKEN"


    ######################################################
    # IF THE THRESHOLD OF THE STAR VECTOR IS LARGER THAT #
    #     A SPECIFIED VALUE THEN IT RETURN AN ERROR      #
    ######################################################

    def star_fault(self, star):
        star = np.array((star))
        current_star = star[0]
        mean_star = np.mean(star)
        var_star = np.var(star)
        norm_star = np.linalg.norm(current_star)

        if round(norm_star,5) != 1:
            self.Error = "STAR_BROKEN"

        if var_star >= self.star_var_threshold:
            self.Error = "STAR_BROKEN"

        
    ########################################
    # IF THE EARTH VECTOR IS LARGER THAN A #
    # GIVEN THRESHOLD THEN RETURN AN ERROR #
    ########################################

    def earth_fault(self, earth):
        earth = np.array((earth))
        current_earth = earth[0]
        mean_earth = np.mean(earth)
        var_earth = np.var(earth)
        norm_earth = np.linalg.norm(current_earth)

        if round(norm_earth,5) != 1 and round(norm_earth,5) != 0:
            self.Error = "EARTH_BROKEN"

        if var_earth >= self.earth_var_threshold:
            self.Error = "EARTH_BROKEN"

    
    ######################################################
    # IF THE ANGULAR MOMENTUM IS LARGER THAN A SPECIFIED #
    #    VALUE OR REMAINS LARGER THAN RETURN AN ERROR    #
    ######################################################

    def angular_momentum_fault(self, angular_moment):
        angular = np.array((angular_moment))
        current_angular = angular[0]
        mean_angular = np.mean(angular)
        var_angular = np.var(angular)
        norm_angular = np.linalg.norm(current_angular)

        if var_angular >= self.angular_var_threshold:
            self.Error = "ANGULAR_BROKEN"

    
    ########################################################
    # IF THE MAGNETOMETER IS LARGER THAN A SPECIFIED VALUE #
    #        OR REMAINS LARGER THAN RETURN AN ERROR        #
    ########################################################

    def magnetometer_fault(self, magnetometer):
        magnetometer = np.array((magnetometer))
        current_magnetometer = magnetometer[0]
        mean_magnetometer = np.mean(magnetometer)
        var_magnetometer = np.var(magnetometer)
        norm_magnetometer = np.linalg.norm(current_magnetometer)

        if round(norm_magnetometer,5) != 1:
            self.Error = "MAGNETOMETER_BROKEN"

        if var_magnetometer >= self.magnetometer_var_threshold:
            self.Error = "MAGNETOMETER_BROKEN"


#####################################################################
#   THIS CLASS IS FOR MORE COMPLEX OPERATIONS THAN IF STATEMENTS    #
# SUCH AS CORRELATION WITH THE MATRIX AND OTHER STATISTICAL METHODS #
#####################################################################

class Correlation_detection:
    def __init__(self):
        ######################
        # STANDARD DEVIATION #
        ######################
        # Use three standard deviations away from the average 
        # of a buffer to flag an anomaly. The buffer will be 
        # used to determine the average as soon as the buffer
        # is full.

        ###############
        # CORRELATION #
        ###############
        # Use the theoretical correlation between sensors and
        # sensor vectors to determine whether the current data
        # is an anomaly or not.

        sun_threshold = 0.15
        earth_threshold = 0.1
        star_threshold = 0.25
        angular_threshold = 0.1 
        magnetometer_threshold = 0.2

    def Per_Timestep(self, args):
        sun, star, earth, angular_momentum = args

        self.sun_fault(sun)
        self.star_fault(star)
        self.earth_fault(earth)
        self.angular_momentum_fault(angular_momentum)

    #####################################################
    # IF THE THRESHOLD OF THE SUN VECTOR IS LARGER THAT #
    #     A SPECIFIED VALUE THEN IT RETURN AN ERROR     #
    #####################################################

    def sun_fault(self, sun):
        pass

    
    #####################################
    # IF THE STAR VECTOR IS LARGER THAN #
    # A GIVEN THRESHOLD RETURN AN ERROR #
    #####################################

    def star_fault(self, star):
        pass

    
    ########################################
    # IF THE EARTH VECTOR IS LARGER THAN A #
    # GIVEN THRESHOLD THEN RETURN AN ERROR #
    ########################################

    def earth_fault(self, earth):
        pass

    
    ######################################################
    # IF THE ANGULAR MOMENTUM IS LARGER THAN A SPECIFIED #
    #    VALUE OR REMAINS LARGER THAN RETURN AN ERROR    #
    ######################################################

    def angular_momentum_fault(self, angular_moment):
        pass

    
    ########################################################
    # IF THE MAGNETOMETER IS LARGER THAN A SPECIFIED VALUE #
    #        OR REMAINS LARGER THAN RETURN AN ERROR        #
    ########################################################

    def magnetometer_fault(self, magnetometer):
        pass

#######################################
# PREDICT WHETHER THERE IS AN ANOMALY #
#######################################
class DecisionTreePredict():
    def __init__(self, path):
        self.clf = pickle.load(open(path, 'rb'))
    
    def Predict(self, X):
        y_predict = self.clf.predict(X)[0]
        return y_predict

###################################################
# THIS CLASS IS THE MOST NOVEL DETECTION METHODS  #
# THIS CLASS WILL HAVE MULTIPLE METHODS TO CHOOSE #
# AND THESE METHODS WILL BE COMPARED TO SEE WHICH #
#       WORKS METHODS BEST FOR EACH SENSOR        #
###################################################

class Encompassing_detection:
    def __init__(self):
        sun_threshold = 0.15
        earth_threshold = 0.1
        star_threshold = 0.25
        angular_threshold = 0.1 
        magnetometer_threshold = 0.2

    def Per_Timestep(self, *args):
        data, strategy, k_nearest = args
        predictions = [1] * len(k_nearest)
        predicted_dictionary = {k_nearest[i]: predictions[i] for i in range(len(k_nearest))}
        return predicted_dictionary
        """
        sun, star, earth, angular_momentum = args

        self.sun_fault(sun)
        self.star_fault(star)
        self.earth_fault(earth)
        self.angular_momentum_fault(angular_momentum)
        """

    #####################################################
    # IF THE THRESHOLD OF THE SUN VECTOR IS LARGER THAT #
    #     A SPECIFIED VALUE THEN IT RETURN AN ERROR     #
    #####################################################

    def sun_fault(self, sun):
        pass

    
    #####################################
    # IF THE STAR VECTOR IS LARGER THAN #
    # A GIVEN THRESHOLD RETURN AN ERROR #
    #####################################

    def star_fault(self, star):
        pass

    
    ########################################
    # IF THE EARTH VECTOR IS LARGER THAN A #
    # GIVEN THRESHOLD THEN RETURN AN ERROR #
    ########################################

    def earth_fault(self, earth):
        pass

    
    ######################################################
    # IF THE ANGULAR MOMENTUM IS LARGER THAN A SPECIFIED #
    #    VALUE OR REMAINS LARGER THAN RETURN AN ERROR    #
    ######################################################

    def angular_momentum_fault(self, angular_moment):
        pass

    
    ########################################################
    # IF THE MAGNETOMETER IS LARGER THAN A SPECIFIED VALUE #
    #        OR REMAINS LARGER THAN RETURN AN ERROR        #
    ########################################################

    def magnetometer_fault(self, magnetometer):
        pass