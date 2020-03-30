import pandas as pd
import numpy as np
from math import log, pow, sqrt

def upd_atr(df, i, period = 3):
    if i >= len(df.index):
        return
    tr_col = 'ATR(1) B#0'
    atr_col = 'ATR(1) B#1'
    if i < 1:
        df.loc[i,tr_col] = 0
        df.loc[i,atr_col] = 0
        return
    # Calculate true range (BUFFER 0 of ATR indicator)
    h = df.at[i,'High'] if df.at[i,'High'] > df.at[i-1,'Close'] else df.at[i-1,'Close']
    l = df.at[i,'Low'] if df.at[i,'Low'] < df.at[i-1,'Close'] else df.at[i-1,'Close']
    df.loc[i,tr_col] = h - l
    if i < period:
        df.loc[i,atr_col] = 0
        return
    atr = 0
    for j in range(i,i-period,-1):
        atr += df.at[j,tr_col]
    df.loc[i,atr_col] = atr/period
    return
    
def upd_toHLC(df,i):
    df.loc[i,'toHigh'] = df.loc[i,'High'] - df.loc[i,'Open']
    df.loc[i,'toLow'] = df.loc[i,'Open'] - df.loc[i,'Low']
    df.loc[i,'toClose'] = df.loc[i,'Close'] - df.loc[i,'Open']
    return

def upd_rsi(df,i,period = 5,price = 'Close'):
    ag_col = 'AverageGain'
    al_col = 'AverageLoss'
    rsi_col = 'RSI'
    if(i >= len(df.index)):
        return
    if (i < period):
        df.loc[i,rsi_col] = 0
        df.loc[i,ag_col] = 0
        df.loc[i,al_col] = 0
        return
    # first value calculations
    if(i == period):
        ag = 0
        al = 0
        for j in range(i,i-period,-1):
            diff = df.at[j,price] - df.at[j-1, price]
            if diff > 0:
                ag += diff
            else:
                al -=diff
        ag /= period
        al /= period
        rsi = 100 - 100/(1 + ag/al)
        df.loc[i,rsi_col] = rsi
        df.loc[i,ag_col] = ag
        df.loc[i,al_col] = al
        return
    diff = df.at[i,price] - df.at[i-1, price]
    if diff > 0:
        ag = (df.loc[i-1,ag_col] * (period-1) + diff)/period
        al = df.loc[i-1,al_col] * (period-1)/period
    else:
        al = (df.loc[i-1,al_col] * (period-1) - diff)/period
        ag = df.loc[i-1,ag_col] * (period-1) / period
    rsi = 100 - 100/(1 + ag/al)
    df.loc[i,rsi_col] = rsi
    df.loc[i,ag_col] = ag
    df.loc[i,al_col] = al
    return

def upd_pifor(df, i, price = 'Close'):
    pifor_b1 = 'PIforJohnOHagan_(LOGARITHM) B#0'
    pifor_b2 = 'PIforJohnOHagan_(LOGARITHM) B#1'
    if i >= len(df.index):
        return
    if i < 1:
        df.loc[i, pifor_b1] = 0
        df.loc[i, pifor_b2] = 0
        return
    signal = (log(df.at[i,price]) - log(df.at[i-1,price]))*100
    if signal > 0:
        df.loc[i,pifor_b1] = signal
        df.loc[i,pifor_b2] = 0
    else:
        df.loc[i,pifor_b1] = 0
        df.loc[i,pifor_b2] = signal
    return 


def upd_stdev(df,i,period = 3, price = 'Close', mode = 'Simple'):
  
    if(i >= len(df.index)):
        return

    ma_col = 'MA(3)'
    stdev_col = 'StdDev(3) B#0'

    if (i < period):
        df.loc[i,ma_col] = 0
        df.loc[i,stdev_col] = 0
        return
    
    # Calculate simple movinge average
    temp = 0
    for j in range(i,i-period,-1):
        temp += df.at[j,price]
    temp /= period
    df.loc[i,ma_col] = temp

    # Calculate standard deviation
    temp = 0
    for j in range(i,i-period,-1):
        temp += pow(df.at[j,price] - df.at[i,ma_col],2)
    df.loc[i,stdev_col] = sqrt(temp/period)
    return


# LINEAR MOMENTUM
def upd_lm(df, i, period = 3):
    lm_col = 'LM (3) B#0'
    if(i >= len(df.index)):
        return
    if (i < period):
        df.loc[i,lm_col] = 0
        return
    point = 1e-5
    if (i == period):
        summ = 0
        for j in range(i,i-period,-1):
            seconds = df.at[j,'Datetime'] - df.at[j-1,'Datetime']
            seconds = seconds.total_seconds()
            distance = (df.at[j,'Close'] - df.at[j,'Open']) / point
            velocity = distance / seconds
            summ += velocity * df.at[j,'Ticks']
        df.loc[i,lm_col] = summ / period
        return
              
    seconds = df.at[i,'Datetime'] - df.at[i-1,'Datetime']
    seconds = seconds.total_seconds()
    distance = (df.at[i,'Close'] - df.at[i,'Open']) / point
    velocity = distance / seconds
    temp = velocity * df.at[i,'Ticks']
    # smothed MA
    #df.loc[i,lm] = (df.loc[i-1,lm] * (period-1) + temp) / period
    # Exponential MA
    df.loc[i,lm_col] = df.loc[i-1,lm_col] + (temp - df.loc[i-1,lm_col])*(2/(period+1))
    return
