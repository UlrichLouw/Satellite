import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import StandardScaler

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def dLSTM(df):
    train_size = int(len(df) * 0.95)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    print(train.shape, test.shape)

    # scaler = StandardScaler()
    # scaler = scaler.fit(train)

    # train = scaler.transform(train)
    # test = scaler.transform(test)

    print(train)

    TIME_STEPS = 30

    # reshape to [samples, time_steps, n_features]

    X_train, y_train = create_dataset(train, train.loc[:, "Current fault binary"], TIME_STEPS)
    X_test, y_test = create_dataset(test, test.loc[:, "Current fault binary"], TIME_STEPS)

    print(X_train.shape)

    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        units=64, 
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
    model.add(keras.layers.LSTM(units=64, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
    model.compile(loss='mae', optimizer='adam')

    # history = model.fit(
    #     X_train, y_train,
    #     epochs=10,
    #     batch_size=32,
    #     validation_split=0.1,
    #     shuffle=False
    # )

    X_train_pred = model.predict(X_train)

    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

    X_test_pred = model.predict(X_test)

    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

    print(test_mae_loss)

    THRESHOLD = 0.65

    test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['Current fault binary'] = test[TIME_STEPS:]["Current fault binary"]

    anomalies = test_score_df[test_score_df.anomaly == True]
    anomalies.head()