from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from data import *
from model import create_model_plstm,Predict
from communication import *
from indicators import *

#COMMON PARAMS
win_size = 6 # window size
features = 8 # number of features
delta_t = 1
weights_path = 'models/weights_6_minimal_set.h5'
scaler_path = 'models/scaler_6_minimal_set.model' 

magic = 47477


class Predictor:
    df = pd.DataFrame()

    def __init__(self,features, win_size, weights,scaler):
        self.win_size = win_size
        self.model = create_model_plstm(features, win_size)
        if os.path.exists(weights):
            try:
                self.model.load_weights(weights)
            except:
                print('Unable to Load weights')
        else:
            print('Cannot find file - %'.format(weights))
        self.scaler = joblib.load(scaler) # load scaler
        return

    def count_bars(self):
        return len(self.df.index)

    # add last bar to the dataframe
    def add_bar(self,candle):
        self.df =  self.df.append(candle, ignore_index=True, sort = False)
        return
    
    # update indicator values for last bar
    def update_indicators(self):
        i = len(self.df.index)-1
        upd_toHLC(self.df,i)
        #upd_atr(self.df, i, period=1)
        #upd_lm(self.df, i, period=3)
        #upd_pifor(self.df,i)
        #upd_stdev(self.df,i,period=3)
        return

    # get last predictions
    def get_last_prediction(self):
        i = len(self.df.index)-1
        if (i < win_size):
            return -1, -1
        else:
            X = get_win(self.df, self.win_size, self.scaler)
            high,low = Predict(X,self.model)
        high = self.df.loc[i,'Close'] + high
        low = self.df.loc[i,'Close'] - low
        print('Next Expected High: {:5f}\nNext Expected Low:  {:5f}'.format(high,low))
        self.df.loc[i,'Prediction_High'] = high
        self.df.loc[i,'Prediction_Low'] = low
        return high,low
    
    def last_update(self):
        return self.df.loc[self.count_bars() -1,'Datetime']
    # save all data to csv file
    def save_chart(self,path):
        self.df.to_csv(path,sep='\t', encoding='utf-16')
        return


if __name__ == '__main__':
    # Sockets for sending and recieving information
    sub = subscriber(port = "tcp://*:5556")
    pub = publisher(port = "tcp://*:5557")
    # Create model and load weights
    predictor = Predictor(features, win_size, weights_path, scaler_path)
    print('Forex Phased LSTM Neural net version 1.09')

    while True:
        print('*********************************************************************')
        print('Waiting for new message')
        msg = sub.recieve()
        frame, key = msg_to_frame(msg)
        print('Recieved: {}'.format(msg))

        # add new data to the common dataframe
        if (key == 'M15'):
            predictor.add_bar(frame)
            predictor.update_indicators()
            high,low = predictor.get_last_prediction()
            if (high != -1):
                i = predictor.count_bars() - 1
                message = "{},{},{:5f},{:5f},{}".format(magic,key,high,low,predictor.df.loc[i,'Datetime'])
                pub.reply(message)
                time.sleep (0.25)
                # save this data to the csv. To see if we have 
                # some problems with such communication
                predictor.save_chart('frame.csv')
            
