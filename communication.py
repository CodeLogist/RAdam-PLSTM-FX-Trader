import time
import zmq
from datetime import datetime
import pandas as pd
import numpy as np


DATE_FORMAT = '%Y.%m.%d %H:%M'
cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticks', 'Spread']

class subscriber:
    context = zmq.Context()
    socket = context.socket(zmq.SUB)

    def __init__(self, port = "tcp://*:5556"):
        self.socket.bind(port)
        self.socket.setsockopt_string(zmq.SUBSCRIBE,'')
        time.sleep(1)

    def recieve(self):
        message = self.socket.recv()
        message = message.decode('utf-8')
        return message
        

class publisher:
    context = zmq.Context()
    socket = context.socket(zmq.PUB)

    def __init__(self, port = "tcp://*:5557"):
        self.socket.bind(port)
        time.sleep(1)

    def reply(self, msg):
        self.socket.send_string(msg)
        time.sleep(0.5)


def msg_to_frame(msg):
    str_arr = msg.split('\t')
    # convert string values
    period = str_arr[1]
    dtm = datetime.strptime(str_arr[2],DATE_FORMAT)
    open = np.float64(str_arr[3])
    high = np.float64(str_arr[4])
    low = np.float64(str_arr[5])
    close = np.float64(str_arr[6])
    volume = int(str_arr[7])
    ticks = int(str_arr[8])
    spread = int(str_arr[9])
    frame = [(dtm, open, high, low, close, volume, ticks, spread)]
    frame = pd.DataFrame(frame, columns = cols)
    return frame, period