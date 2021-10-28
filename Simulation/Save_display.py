from Simulation.Parameters import SET_PARAMS
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Function to save a csv file of simulation data
def save_as_excel(Data, sheetnames):
    with pd.ExcelWriter(SET_PARAMS.path + SET_PARAMS.filename + ".xlsx") as writer:
        i = 0
        for data in Data:
            df = pd.DataFrame(data, columns = data.keys())
            sheetname = sheetnames[i]
            df.to_excel(writer, sheet_name = sheetname, index = False)
            i += 1


####################
# SAVE AS CSV FILE #
####################

def save_as_csv(Data, path, orbit = None, filename = SET_PARAMS.filename, index = False):
    if isinstance(Data, pd.DataFrame):
        Data.reset_index(drop = True, inplace = True)
        Data.to_csv(path + filename + ".csv", index_label = "TimeStep", float_format="%.6f")
    else:
        # Convert array's to individual values in the dictionary

        df = pd.DataFrame(Data, columns = Data.keys())
        df.reset_index(drop = True, inplace = True)
        df.to_csv(path + filename + str(orbit) + ".csv", index_label = "TimeStep", float_format="%.6f")

#######################################################
# FUNCTION TO SAVE A PICKLE FILE OF SIMULATION DATA   #
#######################################################
def save_as_pickle(Data, orbit):
    df = pd.DataFrame(Data, columns = Data.keys())
    df.to_pickle(SET_PARAMS.filename + str(orbit) + ".pkl")

##########################################

##########################################
# FUNCTION TO VISUALIZE DATA AS GRAPHS   #
##########################################
def visualize_data(D, fault, path):
    doNotVisualize = ["Sun in view", "Current fault", "Current fault numeric"]
    singleVisualize = ["Pointing Accuracy", "Estimation Accuracy", "Mean", "Covariance", "Predicted fault", "Current fault binary", "Isolation Accuracy", "Prediction Accuracy", "Quaternion magnetitude error"]

    for data in D:
        if data in doNotVisualize:
            pass
        elif data == "Moving Average":
            newData = []
            for processedData in D[data]:
                newData.append(np.sum(processedData))
            
            y = np.array(newData)
            fig = make_subplots(rows=3, cols=1)
            x = y.shape[0]
            x = np.arange(0,x,1)
            y_min = np.amin(y)
            y_max = np.amax(y)

            fig.append_trace(go.Scatter(
                x=x,
                y=y,
                name = "x"
            ), row=1, col=1)

            fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
            fig.update_layout(height=600, width=600, title_text=str(data))
            fig.write_html(path + "/" + str(data)+".html")
        
        elif data in singleVisualize:
            y = np.array((D[data]))
            fig = make_subplots(rows=3, cols=1)
            x = y.shape[0]
            x = np.arange(0,x,1)
            y_min = np.amin(y)
            y_max = np.amax(y)

            fig.append_trace(go.Scatter(
                x=x,
                y=y,
                name = "x"
            ), row=1, col=1)

            fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
            fig.update_layout(height=600, width=600, title_text=str(data))
            fig.write_html(path + "/" + str(data)+".html")
        
        else:
            y = np.array((D[data]))
            fig = make_subplots(rows=3, cols=1)
            x = y.shape[0]
            x = np.arange(0,x,1)
            y_min = np.amin(y)
            y_max = np.amax(y)

            fig.append_trace(go.Scatter(
                x=x,
                y=y[:, 0],
                name = "x"
            ), row=1, col=1)

            fig.append_trace(go.Scatter(
                x=x,
                y=y[:,1],
                name = "y"
            ), row=2, col=1)

            fig.append_trace(go.Scatter(
                x=x,
                y=y[:,2],
                name = 'z'
            ), row=3, col=1)

            fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
            fig.update_yaxes(range=[y_min, y_max], row=2, col=1)
            fig.update_yaxes(range=[y_min, y_max], row=3, col=1)
            fig.update_layout(height=600, width=600, title_text=str(data))
            fig.write_html(path + str(data)+".html")