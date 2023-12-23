import numpy as np
import tensorflow as tf
from keras.utils import io_utils
from tensorflow.keras import backend as K
from tensorflow.python.platform import tf_logging as logging

class MinMaxScaler():

    def __init__(self):
        self.mini = None
        self.range = None

    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data

    def fit(self, data):    
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data
    
    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data


class DiscrMonitor(tf.keras.callbacks.Callback):

    def __init__(self, monitor='gen_loss', patience=10, cooldown=0, min_delta=1e-4):
        super(DiscrMonitor, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.wait = 0
        self.best = 0
        self.monitor_op = None

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """        
        self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
        self.best = np.Inf
    
        self.cooldown_counter = 0
        self.wait = 0

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 50:
            self._reset
    
    def on_epoch_end(self, epoch, logs=None):
        if self.monitor_op != None:
            logs = logs or {}
            logs['lr'] = K.get_value(self.model.optimizer.lr)
            current = logs.get(self.monitor)
            if current is None:
                logging.warning('Learning rate reduction is conditioned on metric `%s` '
                                'which is not available. Available metrics are: %s',
                                self.monitor, ','.join(list(logs.keys())))

            else:
                if self.in_cooldown():
                    self.cooldown_counter -= 1
                    self.wait = 0

                if self.monitor_op(current, self.best):
                    self.best = current
                    self.wait = 0
                elif not self.in_cooldown():
                    self.wait += 1
                    if self.wait >= self.patience:
                        dstep = K.get_value(self.model.d_step)
                        gstep = K.get_value(self.model.g_step)
                        if dstep > 1:
                            dstep -= 1.0
                            K.set_value(self.model.d_step, dstep)
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                        # elif gstep<=1:
                        #     gstep += 1.0
                        #     K.set_value(self.model.g_step, gstep)
                        #     self.cooldown_counter = self.cooldown
                        #     self.wait = 0


    def in_cooldown(self):
        return self.cooldown_counter > 0