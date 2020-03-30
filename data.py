import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

output_scale = 100

def get_day(x):
    try:    
        y = datetime.strptime(x,'%Y/%m/%d')
    except:
        try:
            y = datetime.strptime(x,'%m/%d/%Y')
        except:
            y = datetime.strptime(x,'%d-%m-%y')
    y = str(y).split(' ')[0]
    return y

def split(X, Y, ratio):
    """
    Split in train and test test set
    """
    train_len = int(len(X) * ratio)
    trainX = X[:train_len]
    trainY = Y[:train_len]
    testX = X[train_len:]
    testY = Y[train_len:]
    
    return trainX, trainY, testX, testY

def create_dataset(dataframe, win_size, delta_t, scaler_path, save_scaler = True):
    dataframe.reset_index(drop=True,inplace=True)

    data = dataframe[['toHigh','toLow','toClose','Ticks','Spread',\
                     'Date','Seconds']] 
    # Disable SettingWithCopyWarning
    pd.options.mode.chained_assignment = None
    data['Hour']   = dataframe['Seconds'].apply(lambda x: int(str(x).split(':')[0])/23.)
    data['Date']   = dataframe['Date'].apply(lambda x: get_day(x))
    data['Day']    = data['Date'].apply(lambda x: int(datetime.strptime(x, '%Y-%m-%d').weekday())/6)
    data['Month']  = data['Date'].apply(lambda x: (int(x.split('-')[1])-1)/11)
    data = data.drop(['Date','Seconds'],axis=1).values
    pd.options.mode.chained_assignment = 'warn'
    
    if save_scaler:
        # Create, set-up and save scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data)
        joblib.dump(scaler, scaler_path)
    else:
        # load scaler
        scaler = joblib.load(scaler_path)
    
    # scale data
    data = scaler.transform(data)
    
    #outputs
    hilo = dataframe[['toHigh','toLow']].values*output_scale

    dataX, dataY = [], []
    for i in range(0, len(data) - win_size, delta_t):
        a = data[i:(i+win_size), :]
        dataX.append(a)
        dataY.append(hilo[i + win_size])
    return np.array(dataX), np.array(dataY)

def get_win(df,win_size,scaler):
    df = df.tail(win_size)
    # Disable SettingWithCopyWarning
    pd.options.mode.chained_assignment = None
    data = df[['toHigh','toLow','toClose','Ticks','Spread']]
    # Convert datetime
    data.loc[:,'Hour']  = df.loc[:,'Datetime'].apply(lambda x: x.hour/23)
    data.loc[:,'Weekday'] = df.loc[:,'Datetime'].apply(lambda x: int(x.weekday())/6)
    data.loc[:,'Month']  = df.loc[:,'Datetime'].apply(lambda x: (x.month-1)/11)
    pd.options.mode.chained_assignment = 'warn'
    data = scaler.transform(data)
    return np.array(data)
