from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from data import *
from model import create_model_plstm,Predict
from communication import *
from indicators import *
from predictor import Predictor

#COMMON PARAMS
win_size = 6 # window size
features = 8 # number of features
delta_t = 1
#GBPUSD SET
#weights_path = {'M15':'models/weights_6_M15.h5',
#                'M30': 'models/weights_6_M30.h5',
#                'H1':'models/weights_6_H1.h5'}
#scaler_path = {'M15': 'models/scaler_6_M15.model',
#               'M30': 'models/scaler_6_M30.model',
#               'H1': 'models/scaler_6_H1.model'}

weights_path = {'M15':'models/weights_6_M15_EURUSD.h5',
                'M30': 'models/weights_6_M30_EURUSD.h5',
                'H1':'models/weights_6_H1_EURUSD.h5'}
scaler_path = {'M15': 'models/scaler_6_M15_EURUSD.model',
               'M30': 'models/scaler_6_M30_EURUSD.model',
               'H1': 'models/scaler_6_H1_EURUSD.model'}



if __name__ == '__main__':
    print('Forex Phased LSTM Neural net version 1.09 (3 in 1)')
    # Sockets for sending and recieving information
    sub = subscriber(port = "tcp://*:5556")
    pub = publisher(port = "tcp://*:5557")
    magic = 47477
    # Create model and load weights
    predictors = {}
    predictors['M15'] = Predictor(features, win_size, weights_path['M15'], scaler_path['M15'])
    predictors['M30'] = Predictor(features, win_size, weights_path['M30'], scaler_path['M30'])
    predictors['H1'] = Predictor(features, win_size, weights_path['H1'], scaler_path['H1'])

    while True:
        print('*********************************************************************')
        print('Waiting for new message')
        msg = sub.recieve()
        frame, key = msg_to_frame(msg)
        print('Recieved: {}'.format(msg))

        # add new data to the common dataframe
        if (key == 'M15' or key == 'H1' or key == 'M30'):
            predictor = predictors[key]
        else:
            continue

        predictor.add_bar(frame)
        predictor.update_indicators()
        high,low = predictor.get_last_prediction()
        message = "{},{},{:5f},{:5f},{}".format(magic,key,high,low,predictor.last_update())
        pub.reply(message)
        time.sleep (0.25)
        # save this data to the csv. To see if we have 
        # some problems with such communication
        predictor.save_chart(key + '.csv')
