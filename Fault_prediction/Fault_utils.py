import pandas as pd
import numpy as np
from Simulation.Parameters import SET_PARAMS
import collections

X_buffer = []
Y_buffer = []



def Binary_split(classified_data):
    for index in range(len(classified_data.index)):
        if classified_data["Current fault"][index] == "None":
            classified_data["Current fault"][index] = 0
        else:
            classified_data["Current fault"][index] = 1

    return classified_data

def Dataset_order(index, binary_set, buffer, categorical_num, controlInputx = True, ControlInput = False, onlySensors = False, use_previously_saved_models = False, columns_compare = None, columns_compare_to = None):
    
    shapeSize = 1

    if isinstance(index, int):
        if SET_PARAMS.load_as == ".xlsx":
            excel_file = SET_PARAMS.filename + ".xlsx"
            xls = pd.ExcelFile(excel_file)
            Data = pd.read_excel(xls, str(index))
        elif SET_PARAMS.load_as == ".csv":
            csv_file = SET_PARAMS.filename + str(index) + ".csv"
            Data = pd.read_csv(csv_file)
        else:
            pickle_file = SET_PARAMS.filename + str(index) + ".pkl"
            Data = pd.read_pickle(pickle_file)
    else:
        if SET_PARAMS.load_as == ".csv":
            csv_file = SET_PARAMS.path + index + ".csv"
            Data = pd.read_csv(csv_file)
        else:
            pickle_file = SET_PARAMS.path + index+ ".pkl"
            Data = pd.read_pickle(pickle_file)
    Data = Data.loc[:, ~Data.columns.str.contains("^Unnamed")]
    Data = Data.loc[:, ~Data.columns.str.contains("^TimeStep")]
    Data = Data.loc[:, ~Data.columns.str.contains("Euler Angles")]
    Data = Data.loc[:, ~Data.columns.str.contains("Angular velocity of satellite")]

    ReplaceDict = {'\n': '',
                    '[': '',
                    ']': '',
                    '  ': ' '
    }

    try:
        for replacement in ReplaceDict:
            Data['Moving Average'] = Data['Moving Average'].str.replace(replacement,ReplaceDict[replacement], regex = True)

        Data['Moving Average'] = Data['Moving Average'].apply(lambda x: np.fromstring(x, sep=' '))

        DataMA = pd.DataFrame(Data['Moving Average'].tolist()).add_prefix('Moving Average')

        Data = pd.concat([Data, DataMA], axis = 1)
    except:
        pass


    Data.drop(columns = ['Moving Average'], inplace = True)

    if binary_set and use_previously_saved_models == False:
        Orbit = Data.drop(columns = ["Predicted fault", 'Current fault', 'Current fault numeric'])
    elif categorical_num:
        Orbit = Data.drop(columns = ["Predicted fault", 'Current fault', 'Current fault binary'])
    else:
        Orbit = Binary_split(Data)

    if onlySensors and controlInputx:
        Orbit = Orbit.loc[:, ~Orbit.columns.str.contains("^Moving Average")]
    elif onlySensors:
        Orbit = Orbit.drop(columns = ['Wheel Control Torques_x',
                            'Wheel Control Torques_y', 'Wheel Control Torques_z'])

    if columns_compare != None:
        columns_to_keep = columns_compare + columns_compare_to
        Orbit = Orbit[columns_to_keep]
        X = Orbit[columns_compare].to_numpy()
        Y = Orbit[columns_compare_to].to_numpy()
    else:
        Orbit.drop(columns = ['Sun in view'], inplace = True)
        if onlySensors:
            X = Orbit.loc[:,Orbit.columns.str.contains('Sun') | Orbit.columns.str.contains('Magnetometer') |
                            Orbit.columns.str.contains('Earth') | 
                            Orbit.columns.str.contains('Star')].to_numpy() # Ignore the angular sensor
        else:
            X = Orbit.loc[:,Orbit.columns.str.contains('Sun') | Orbit.columns.str.contains('Magnetometer') |
                            Orbit.columns.str.contains('Earth') | 
                            Orbit.columns.str.contains('Star') | 
                            Orbit.columns.str.contains('Moving Average') ].to_numpy()  # Ignore the angular sensor
        Y = Orbit.loc[:,Orbit.columns.str.contains('fault')].to_numpy()

    if ControlInput:
        Y = Data.loc[:,Data.columns.str.contains('Control Torques')].to_numpy()
        shapeSize = 6

    buffer_x = collections.deque(maxlen = SET_PARAMS.buffer_size)
    buffer_correlation_sun_earth_magnetometer = collections.deque(maxlen = SET_PARAMS.buffer_size)
    y = Y[SET_PARAMS.buffer_size - 1:]
    buffer_y = []

    if buffer == True:
        for i in range(SET_PARAMS.buffer_size - 1):
            buffer_x.append(X[i,:])
            #buffer_correlation_sun_earth_magnetometer.append(X_correlation_sun_earth_magnetometer[i,:])

        for i in range(SET_PARAMS.buffer_size, X.shape[0]):
            buffer_x.append(X[i,:])
            #buffer_correlation_sun_earth_magnetometer.append(X_correlation_sun_earth_magnetometer[i,:])
            if use_previously_saved_models == True:
                buffer_y.append(np.fromstring(y[i-SET_PARAMS.buffer_size][1:-1], dtype = float, sep=','))
            #Binary_stat_fault(buffer_correlation_sun_earth_magnetometer)
            X_buffer.append(np.asarray(buffer_x).flatten())
            #X_buffer_replaced.append(np.asarray(X_buffer).flatten())

        X = np.asarray(X_buffer)
    if use_previously_saved_models == True:
        Y = np.asarray(buffer_y)
        Y = Y.reshape(X.shape[0], Y.shape[1])
        Y_buffer.append(Y)
    elif buffer == True:
        Y = np.asarray(Y[SET_PARAMS.buffer_size:]).reshape(X.shape[0],shapeSize)
        Y_buffer.append(Y)
    else:
        Y = np.asarray(Y).reshape(X.shape[0],shapeSize)
        Y_buffer.append(Y)

    return Y, Y_buffer, X, X_buffer, Orbit