import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout

# def discriminator(input_shape, mlp_units, dropout=0.0):
#     inputs = Input(input_shape)
#     x = inputs
#     for u in mlp_units:
#         x = Dense(u, activation='relu')(x)
#         x = Dropout(dropout)(x)
    
#     outputs = Dense(1, activation='sigmoid')(x)
#     return Model(inputs, outputs)

def disc_eva(input_shape, rnn_unit, dropout=0.0):
    inputs = Input(input_shape)
    x = inputs
    for u in rnn_unit[:-1]:
        x = LSTM(u, activation='tanh', return_sequences=True)(x)
        x = Dropout(dropout)(x)
    x = LSTM(rnn_unit[-1], activation='tanh')(x)
    x = Dropout(dropout)(x)
    # x = Dense(rnn_unit[-1], activation='relu')(x)
    # x = Dropout(dropout)(x)  
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

def fore_eva(input_shape, out_shape, rnn_unit):
    inputs = Input(input_shape)
    x = inputs
    for u in rnn_unit[:-1]:
        x = LSTM(u, activation='tanh', return_sequences=True)(x)
    x = LSTM(rnn_unit[-1], activation='tanh')(x)
    outputs = Dense(out_shape)(x)
    return Model(inputs, outputs)