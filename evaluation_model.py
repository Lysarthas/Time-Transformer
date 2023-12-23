import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, MultiHeadAttention, LayerNormalization, Conv1D, GlobalAveragePooling1D

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

def disc_eva_trans(input_shape, head_size, num_heads, n_filter, mlp_units, dropout=0.0):
    inputs = Input(input_shape)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x+inputs

    x = Conv1D(filters=n_filter, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = x+res

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(dropout)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs)


def fore_eva(input_shape, out_shape, rnn_unit):
    inputs = Input(input_shape)
    x = inputs
    for u in rnn_unit[:-1]:
        x = LSTM(u, activation='tanh', return_sequences=True)(x)
    x = LSTM(rnn_unit[-1], activation='tanh', return_sequences=True)(x)
    outputs = Dense(out_shape, activation='sigmoid')(x)
    return Model(inputs, outputs)

def fore_model(input_shape, out_shape, rnn_unit):
    inputs = Input(input_shape)
    x = inputs
    for u in rnn_unit[:-1]:
        x = LSTM(u, activation='tanh', return_sequences=True)(x)
    x = LSTM(rnn_unit[-1], activation='tanh')(x)
    outputs = Dense(out_shape)(x)
    return Model(inputs, outputs)

def fore_eva_trans(input_shape, head_size, num_heads, n_filter, mlp_units, out_shape, dropout=0.0):
    inputs = Input(input_shape)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x+inputs

    x = Conv1D(filters=n_filter, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = x+res

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="tanh")(x)
        x = Dropout(dropout)(x)
    outputs = Dense(out_shape)(x)
    return Model(inputs, outputs)