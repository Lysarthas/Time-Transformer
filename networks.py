import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, Conv1D, AveragePooling1D, SpatialDropout1D, LayerNormalization, MultiHeadAttention
from tensorflow.keras.layers import Conv1DTranspose, Flatten, MaxPooling1D

def timesformer_layer(tcn_inputs, trans_inputs, head_size, num_heads, filters, k_size, dilation, dropout=0.0):
    #Temporal Convolution
    x = Conv1D(filters, k_size, padding='causal', dilation_rate=dilation, activation='relu')(tcn_inputs)
    x = LayerNormalization(epsilon=1e-06)(x)
    tcn_out = SpatialDropout1D(dropout)(x)
    #Transformer
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(trans_inputs, trans_inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-06)(x)
    res = x + trans_inputs
    x = Conv1D(filters, kernel_size=1, activation='relu')(res)
    x = LayerNormalization(epsilon=1e-6)(x)
    trans_out = x + res
    #Cross Attention
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(trans_out, tcn_out)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-06)(x)
    chnl_trans = x + trans_out

    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(tcn_out, trans_out)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-06)(x)
    chnl_tcn = x + tcn_out

    return chnl_tcn, chnl_trans

def timesformer_dec(input_shape, ts_shape, head_size, num_heads, n_filters, k_size, dilations, dropout=0.0):
    inputs = Input(input_shape)
    x = inputs
    # De-Conv
    cnn_dim = n_filters[0] * ts_shape[0]/(2**len(n_filters))
    ts_len = ts_shape[0]
    ts_dim = ts_shape[1]
    x = Dense(cnn_dim, activation='relu')(x)
    x = Reshape((-1, n_filters[0]))(x)
    for f in n_filters[1:]:
        x = Conv1DTranspose(f, k_size, strides=2, padding='same', activation='relu')(x)
    x = Conv1DTranspose(ts_dim, k_size, strides=2, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(ts_dim * ts_len)(x)
    res = Reshape((ts_len, ts_dim))(x)

    tcn, trans = res, res
    for d in dilations:
        tcn, trans = timesformer_layer(tcn, trans, head_size, num_heads, ts_dim, k_size, d, dropout)
    x = tf.concat([tcn, trans], axis=-1)
    x = Flatten()(x)
    x = Dense(ts_dim * ts_len)(x)
    x = Reshape((ts_len, ts_dim))(x)
    outputs = x
    return Model(inputs, outputs, name='decoder')

def timesformer_enc(input_shape, latent_dim, head_size, num_heads, n_filters, k_size, dilations, dropout=0.0):
    inputs = Input(shape=input_shape)
    ts_len = input_shape[0]
    ts_dim = input_shape[1]
    tcn, trans = inputs, inputs
    for d in dilations:
        tcn, trans = timesformer_layer(tcn, trans, head_size, num_heads, ts_dim, k_size, d, dropout)
        tcn = AveragePooling1D(2)(tcn)
        trans = AveragePooling1D(2)(trans)
    x = tf.concat([tcn, trans], axis=-1)
    x = Flatten()(x)
    outputs = Dense(latent_dim)(x)
    return Model(inputs, outputs, name='encoder')


def cnn_enc(input_shape, latent_dim, n_filters, k_size, dropout=0.0):
    inputs = Input(shape=input_shape)
    x = inputs
    for f in n_filters:
        x = Conv1D(f, k_size, padding='same', strides=2, activation='relu')(x)
        x = Dropout(dropout)(x)
    x = Flatten()(x)
    outputs = Dense(latent_dim)(x)
    return Model(inputs, outputs, name='encoder')

def SeqCNNEnc(input_shape, latent_dim, n_filters, k_size, dropout=0.0):
    inputs = Input(shape=input_shape)
    x = inputs
    for f in n_filters:
        x = Conv1D(f, k_size, padding='same', strides=1, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    outputs = Dense(latent_dim)(x)
    return Model(inputs, outputs, name='seq_encoder')

def cnn_dec(input_shape, ts_shape, n_filters, k_size):
    inputs = Input(shape=input_shape)
    x = inputs
    cnn_dim = n_filters[0] * ts_shape[0]/(2**len(n_filters))
    ts_len = ts_shape[0]
    ts_dim = ts_shape[1]
    x = Dense(cnn_dim, activation='relu')(x)
    x = Reshape((-1, n_filters[0]))(x)
    for f in n_filters[1:]:
        x = Conv1DTranspose(f, k_size, strides=2, padding='same', activation='relu')(x)
    x = Conv1DTranspose(ts_dim, k_size, strides=2, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(ts_dim * ts_len)(x)
    outputs = Reshape((ts_len, ts_dim))(x)
    return Model(inputs, outputs, name='decoder')

def cautrans_dec(input_shape, ts_shape, n_block, head_size, num_heads, n_filters, k_size, dilations, dropout=0.0):
    inputs = Input(input_shape)
    x = inputs
    # De-Conv
    cnn_dim = n_filters[0] * ts_shape[0]/(2**len(n_filters))
    ts_len = ts_shape[0]
    ts_dim = ts_shape[1]
    x = Dense(cnn_dim, activation='relu')(x)
    x = Reshape((-1, n_filters[0]))(x)
    for f in n_filters[1:]:
        x = Conv1DTranspose(f, k_size, strides=2, padding='same', activation='relu')(x)
    x = Conv1DTranspose(ts_dim, k_size, strides=2, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(ts_dim * ts_len)(x)
    res = Reshape((ts_len, ts_dim))(x)

    # Attention
    for _ in range(n_block):
        x = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(res, res)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = x + res
        x = Conv1D(ts_dim, kernel_size=1, activation='relu')(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        res = x + res

    # TCN
    x = res
    for d in dilations:
        x = Conv1D(ts_dim, k_size, padding='causal', dilation_rate=d, activation='relu')(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = SpatialDropout1D(dropout)(x)
    x = Flatten()(x)
    x = Dense(ts_dim * ts_len)(x)
    x = Reshape((ts_len, ts_dim))(x)
    outputs = x + res if n_block == 0 else x
    return Model(inputs, outputs, name='decoder')

def discriminator(input_shape, hidden_unit, dropout=0.3):
    inputs = Input(input_shape)
    x = Dense(hidden_unit, activation='relu')(inputs)
    x = Dropout(dropout)(x)
    x = Dense(hidden_unit, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs, name='discriminator')