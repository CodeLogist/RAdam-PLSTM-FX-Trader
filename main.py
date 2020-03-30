from data import *
from model import *
import numpy

#COMMON PARAMS
win_size = 6 # window size
features = 8 # number of features
delta_t = 1
weights_path = {'M15':'Models/weights_6_M15_EURUSD.h5',
                'M30': 'Models/weights_6_M30_EURUSD.h5',
                'H1':'Models/weights_6_H1_EURUSD.h5'}
scaler_path = {'M15': 'Models/scaler_6_M15_EURUSD.model',
               'M30': 'Models/scaler_6_M30_EURUSD.model',
               'H1': 'Models/scaler_6_H1_EURUSD.model'}

# TRAIN/EVALUATE PARAMS
data_path = {'M15': 'data/EURUSD_M15_2010-2019.csv',
             'M30': 'data/EURUSD_M30_2010-2019.csv',
             'H1': 'data/EURUSD_H1_2010-2019.csv'}
batch_size = 256
epochs = 100
log = False

if __name__ == '__main__':
    print('Forex Phased LSTM Neural EURUSD net version 1.09')
    numpy.random.seed(41)

    timeframe = 'M30'
    
    mode = 'test' #[train, eval, test]
    test_set = 30

    # load data
    data = pd.read_csv(data_path[timeframe],encoding='utf-16',sep='\t')

    # calculate features
    data['toHigh'] = data['High'] - data['Open']
    data['toLow'] = data['Open'] - data['Low']
    data['toClose'] = data['Close'] - data['Open']

    opens = data['Open']
    opens = np.array(opens)

    if mode == 'train':
        dataset = create_dataset(data, win_size,delta_t,scaler_path[timeframe], save_scaler=True)
    else:
        dataset = create_dataset(data, win_size,delta_t,scaler_path[timeframe], save_scaler=False)

    # Create model and load weights
    model_PLSTM = create_model_plstm(features, win_size)
    if os.path.exists(weights_path[timeframe]):
        try:
            model_PLSTM.load_weights(weights_path[timeframe])
        except:
            print('Unable to Load weights')
    else:
        print('No weights file found: {}'.format(weights_path[timeframe]))

    start_time = time.time()
    if mode == 'train':
        Train(dataset, model_PLSTM, weights=weights_path[timeframe],n_epoch=epochs,batch=batch_size)
    elif mode == 'eval':
        Evaluate(dataset, model_PLSTM, log= True)
    elif mode == 'test':
        Test(dataset,opens, model_PLSTM, test_set,log=True)
    else:
        raise Exception("Invalid Mode {}. Select from 'train','evaluate','test_model'".format(mode))

    print("Time used: {} sec".format(time.time() - start_time))
