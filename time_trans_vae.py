
import os, warnings, sys
from re import T
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D,  Flatten, Dense, Conv1DTranspose, Reshape, Input, MultiHeadAttention, Dropout, SpatialDropout1D, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# from utils import get_mnist_data, draw_orig_and_post_pred_sample, plot_latent_space
from vae_base import BaseVariationalAutoencoder, Sampling 



class Time_Trans_VAE(BaseVariationalAutoencoder):    
    model_name = "VAE_INT"

    def __init__(self,  hidden_layer_sizes, dilations, k_size, head_size, num_heads, dropout, **kwargs):
        '''
            hidden_layer_sizes: list of number of filters in convolutional layers in encoder and residual connection of decoder. 
            trend_poly: integer for number of orders for trend component. e.g. setting trend_poly = 2 will include linear and quadratic term. 
            num_gen_seas: Number of sine-waves to use to model seasonalities. Each sine wae will have its own amplitude, frequency and phase. 
            custom_seas: list of tuples of (num_seasons, len_per_season). 
                num_seasons: number of seasons per cycle. 
                len_per_season: number of epochs (time-steps) per season.
            use_residual_conn: boolean value indicating whether to use a residual connection for reconstruction in addition to
            trend, generic and custom seasonalities.        
        '''

        super(Time_Trans_VAE, self).__init__(**kwargs)

        self.hidden_layer_sizes = hidden_layer_sizes
        self.dilations = dilations    #[1,2,4]
        self.k_size= k_size           #4
        self.head_size = head_size    #64
        self.num_heads = num_heads    #3
        self.dropout = dropout        #0.2
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        self.compile(optimizer=Adam())


    def _get_encoder(self):
        encoder_inputs = Input(shape=(self.seq_len, self.feat_dim), name='encoder_input')
        x = encoder_inputs            
        for i, num_filters in enumerate(self.hidden_layer_sizes):
            x = Conv1D(
                    filters = num_filters, 
                    kernel_size=self.k_size, 
                    strides=2, 
                    activation='relu', 
                    padding='same',
                    name=f'enc_conv_{i}')(x)

        x = Flatten(name='enc_flatten')(x)

        # save the dimensionality of this last dense layer before the hidden state layer. We need it in the decoder.
        self.encoder_last_dense_dim = x.shape[-1]       

        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

        encoder_output = Sampling()([z_mean, z_log_var])     
        self.encoder_output = encoder_output
        
        
        encoder = Model(encoder_inputs, [z_mean, z_log_var, encoder_output], name="encoder")
        # encoder.summary()
        return encoder


    def timesformer_layer(self, tcn_inputs, trans_inputs, head_size, num_heads, filters, k_size, dilation, dropout=0.0):
        #TCN
        x = Conv1D(filters, k_size, padding='causal', dilation_rate=dilation, activation='relu')(tcn_inputs)
        x = LayerNormalization(epsilon=1e-06)(x)
        tcn_out = SpatialDropout1D(dropout)(x)

        #Self Attention
        # x = tf.transpose(trans_inputs, perm=[0, 2, 1])
        x = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(trans_inputs, trans_inputs)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-06)(x)
        # x = tf.transpose(x, perm=[0, 2, 1])
        res = x + trans_inputs
        x = Conv1D(filters, kernel_size=1, activation='relu')(res)
        x = Dropout(dropout)(x)
        x = Conv1D(filters=trans_inputs.shape[-1], kernel_size=1)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        trans_out = x + res

    #Cross Attention
        x = MultiHeadAttention(
            key_dim=head_size, num_heads=1, dropout=dropout
        )(trans_out, tcn_out)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-06)(x)
        chnl_trans = x + trans_out

        x = MultiHeadAttention(
            key_dim=head_size, num_heads=1, dropout=dropout
        )(tcn_out, trans_out)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-06)(x)
        chnl_tcn = x + tcn_out

        return chnl_tcn, chnl_trans
    
    def _get_decoder(self):
        decoder_inputs = Input(shape=(int(self.latent_dim),), name='decoder_input')
        
        x = decoder_inputs
        dilations = self.dilations
        k_size = self.k_size
        head_size = self.head_size
        num_heads = self.num_heads
        dropout = self.dropout
# De-Conv
        tcn_dim = self.hidden_layer_sizes[-1] * self.seq_len/(2**len(dilations))
        ts_len = self.seq_len
        ts_dim = self.feat_dim
        x = Dense(tcn_dim, activation='relu')(x)
        x = Reshape((-1, self.hidden_layer_sizes[-1]))(x)
        for f in reversed(self.hidden_layer_sizes[:-1]):
            x = Conv1DTranspose(f, k_size, strides=2, padding='same', activation='relu')(x)
        x = Conv1DTranspose(ts_dim, k_size, strides=2, padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(ts_dim * ts_len)(x)
        res = Reshape((ts_len, ts_dim))(x)

        tcn, trans = res, res
        for d in dilations:
            tcn, trans = self.timesformer_layer(tcn, trans, head_size, num_heads, ts_dim, k_size, d, dropout)
        x = tf.concat([tcn, trans], axis=-1)
        x = Flatten()(x)
        x = Dense(ts_dim * ts_len)(x)
        x = Reshape((ts_len, ts_dim))(x)
# f_dim = x.shape[-1]*5
        # x = Conv1D(self.hidden_layer_sizes[0], kernel_size=k_size, padding='same', activation='relu')(x)
        # x = Dropout(dropout)(x)
        # x = Conv1D(ts_dim, kernel_size=1, padding='same')(x)
        outputs = x + res
        return Model(decoder_inputs, outputs, name='decoder')

    @classmethod
    def load(cls, model_dir) -> "Time_Trans_VAE":
        params_file = os.path.join(model_dir, f"{cls.model_name}_parameters.pkl")
        dict_params = joblib.load(params_file)
        vae_model = Time_Trans_VAE(**dict_params)
        vae_model.load_weights(model_dir)
        vae_model.compile(optimizer=Adam())
        return vae_model