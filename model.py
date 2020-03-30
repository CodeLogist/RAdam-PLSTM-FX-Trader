
# common packages
import numpy as np
import pandas as pd
import os
import time
import math
from datetime import datetime

# NN packages
from keras.models import Model
from keras.layers import Dense, Input,Conv1D,Flatten
from keras.optimizers import Adam
from keras_radam import RAdam
from keras.callbacks import Callback, ModelCheckpoint
from phased_lstm_keras.PhasedLSTM import PhasedLSTM as PLSTM

# own packages
from data import *

output_scale = 100

def create_model_plstm(features, win_size):
    if win_size >= 12:
    # LSTM with timegate
        inputs = Input(shape=(win_size, features))
        x = PLSTM(32, activation='relu',implementation=1,return_sequences=True,name='plstm1')(inputs)
        x = Conv1D(64,activation='relu',kernel_size=5,padding='valid',name='conv1')(x)
        x = Conv1D(128,activation='relu',kernel_size=5,padding='valid',name='conv2')(x)
        x = Conv1D(256,activation='relu',kernel_size=3,padding='valid',name='conv3')(x)
    elif win_size >= 6 and win_size < 12:
        inputs = Input(shape=(win_size, features))
        x = PLSTM(32, activation='relu',implementation=1,return_sequences=True,name='plstm1')(inputs)
        x = Conv1D(64,activation='relu',kernel_size=3,padding='valid',name='conv1')(x)
        x = Conv1D(128,activation='relu',kernel_size=3,padding='valid',name='conv2')(x)
        x = Conv1D(256,activation='relu',kernel_size=2,padding='valid',name='conv3')(x)
    else:
        inputs = Input(shape=(win_size, features))
        x = PLSTM(32, activation='relu',implementation=1,return_sequences=True,name='plstm1')(inputs)
        x = Conv1D(64,activation='relu',kernel_size=2,padding='valid',name='conv1')(x)
        x = Conv1D(128,activation='relu',kernel_size=1,padding='valid',name='conv2')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu',name='dense1')(x)
    out= Dense(2,activation='linear',name='out')(x)
    model_PLSTM = Model(inputs,out)
    radam = RAdam(learning_rate=0.005)
    model_PLSTM.compile(optimizer=radam, loss='mse',metrics=['mae'])
    return model_PLSTM

def Train(data, model_PLSTM, weights, n_epoch = 1, batch = 256, log = False):
    
    X, Y = data
    X_train, Y_train, X_test, Y_test = split(X, Y, ratio=0.9)
    
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)

    chkpt2 = ModelCheckpoint(weights, save_best_only=True, save_weights_only=True, monitor='val_loss')
    
    history = model_PLSTM.fit(X_train,
                              Y_train,
                              shuffle=True,
                              epochs=n_epoch,
                              batch_size=batch,
                              validation_data=(X_test,Y_test),
                              verbose=1,
                              callbacks=[chkpt2])
    h = history.history

def Evaluate(data, model_PLSTM, log = False):
    X, Y = data
    _, _, X_test, Y_test = split(X, Y, ratio=0.9)
    
    mse,mae = model_PLSTM.evaluate(X_test, Y_test, verbose=2)
    if log:
        print('Mean Squared Error: ', mse)
        print('Mean Absolute Error: ',mae)
        print('RMSE: ',math.sqrt(mse))
    return mse,mae

def Test(data,opens, model_PLSTM, test_size, log = False):
    X,Y = data
    
    X = X[-test_size:]
    Y = Y[-test_size:]
    opens = opens[-test_size:]
    
    prediction_plstm = model_PLSTM.predict(X,verbose=1)
    if log:
        print('------------------------------------------------------------------')
        p_differences_high = []
        p_differences_low = []
        columns = [
            "Pred-NH",
            "Pred-NL",
            "Orig-NH",
            "Orig-NL",
            "Diff-High",
            "Diff-Low",
            "Perc-D-H",
            "Perc-D-L"
        ]
        print("{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t{}".format(*columns))
        for i in range(len(Y)):
            o = opens[i]
            next_high_PLSTM = o + prediction_plstm[i][0]/output_scale
            next_low_PLSTM  = o + prediction_plstm[i][1]/output_scale
            next_high_original = o + Y[i][0]/output_scale
            next_low_original = o + Y[i][1]/output_scale

            DH = next_high_original - next_high_PLSTM
            PDH = (abs(DH) / ((next_high_PLSTM + next_high_original) / 2)) * 100

            DL = next_low_original - next_low_PLSTM
            PDL = (abs(DL) / ((next_low_PLSTM + next_low_original) / 2)) * 100

            p_differences_high.append(PDH)
            p_differences_low.append(PDL)

            output = [
                next_high_PLSTM,
                next_low_PLSTM,
                next_high_original,
                next_low_original,
                "+" if DH > 0 else "",
                DH,
                "+" if DL > 0 else "",
                DL,
                PDH,
                PDL
            ]
            print("{:0.5f}\t\t{:0.5f}\t\t{:0.5f}\t\t{:0.5f}\t\t{}{:0.5f}\t\t{}{:0.5f}\t\t{:0.5f}\t\t{:0.5f}".format(
                *output
            ))

        print("\nMax Percentage error (High): \t\t{:0.5f} %".format(np.max(p_differences_high)))
        print("Min Percentage error (High): \t\t{:0.5f} %".format(np.min(p_differences_high)))
        print("Avg Percentage error (High): \t\t{:0.5f} %".format(np.average(p_differences_high)))

        print("\nMax Percentage error (Low): \t\t{:0.5f} %".format(np.max(p_differences_low)))
        print("Min Percentage error (Low): \t\t{:0.5f} %".format(np.min(p_differences_low)))
        print("Avg Percentage error (Low): \t\t{:0.5f} %".format(np.average(p_differences_low)))
        print()
    return prediction_plstm

def Predict(dataset, model_PLSTM):
    X = np.expand_dims(dataset, 0)

    prediction_plstm = model_PLSTM.predict(X, verbose=0)
    next_high_PLSTM = prediction_plstm[0][0]/output_scale
    next_low_PLSTM  = prediction_plstm[0][1]/output_scale
    return next_high_PLSTM, next_low_PLSTM
