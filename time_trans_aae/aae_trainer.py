from requests import head
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import networks
from aae import aae_model
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tools import MinMaxScaler
# from data_loading import real_data_loading

class train_monitor(tf.keras.callbacks.Callback):
    def __init__(self, ckp_path):
        self.path = ckp_path

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 100 == 0:
            ckp_num = int((epoch+1)/100)
            self.model.save_weights(self.path + '/checkpoint' +  str(ckp_num) + '/aae_ckp')

def ae_loss(ori_ts, rec_ts):
    return tf.keras.metrics.mse(ori_ts, rec_ts)

def dis_loss(y_true, y_pred):
#     l_real = -tf.reduce_mean(real)
#     l_fake = tf.reduce_mean(fake)
    return tf.keras.metrics.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)

def gen_loss(y_true, y_pred):
    return tf.keras.metrics.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)
#     return -tf.reduce_mean(fake)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sine', type=str)
    parser.add_argument('--valid_perc', default=0.1, type=float)
    parser.add_argument('--latent_dim', default=8, type=int)
    parser.add_argument('--D_step', default=1, type=int)
    parser.add_argument('--G_step', default=1, type=int)
    args = parser.parse_args()

    dataset = args.dataset
    valid_perc = args.valid_perc

    if dataset == 'ecochg':
        x = np.load('cm_data/cm_x.npy')
        y = np.load('cm_data/cm_y.npy')
        x = np.expand_dims(x[:, 22:], axis=-1)
        pos = x[y==1]
        neg = x[y==0]
        idx = np.random.permutation(len(neg))[:int(len(pos)*1.5)]
        neg = neg[idx]
        full_train_data = np.concatenate((pos, neg))
    else:
        full_train_data = np.load('dataset/'+dataset+'.npy')

    N, T, D = full_train_data.shape
    # valid_perc = 0.2
    # N = len(full_train_data)

    N_train = int(N * (1 - valid_perc))
    N_valid = N - N_train
    np.random.shuffle(full_train_data)
    train_data = full_train_data[:N_train]
    valid_data = full_train_data[N_train:]

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_data)
    x_valid = scaler.transform(valid_data)
    
    ts_shape = x_train.shape[1:]
    latent = args.latent_dim
    
    encoder = networks.cnn_enc(
        input_shape=ts_shape,
        latent_dim=latent,
        n_filters=[64, 128, 256],
        k_size=4,
        dropout=0.2
    )

    # decoder = networks.timesformer_dec(
    #     input_shape=latent,
    #     ts_shape=ts_shape,
    #     head_size=64,
    #     num_heads=3,
    #     n_filters=[128, 64],
    #     k_size=4,
    #     dilations=[1,4],
    #     dropout=0.2
    # )

    decoder = networks.cautrans_dec(
        input_shape=latent,
        ts_shape=ts_shape,
        n_block = 1,
        head_size=64,
        num_heads=3,
        n_filters=[128, 64],
        k_size=4,
        dilations=[1,4],
        dropout=0.2
    )

    discriminator = networks.discriminator(input_shape=latent, hidden_unit=32)

    ae_schedule = PolynomialDecay(initial_learning_rate=0.005, decay_steps=300, end_learning_rate=0.0025, power=0.5)
    dc_schedule = PolynomialDecay(initial_learning_rate=0.001, decay_steps=300, end_learning_rate=0.0001, power=0.5)
    ge_schedule = PolynomialDecay(initial_learning_rate=0.001, decay_steps=300, end_learning_rate=0.0001, power=0.5)
    ae_opt = tf.keras.optimizers.Adam(ae_schedule)
    dc_opt = tf.keras.optimizers.Adam(dc_schedule)
    ge_opt = tf.keras.optimizers.Adam(ge_schedule)

    train_perc = int(10*(1-valid_perc))
    path = "saved_model/"+dataset+str(train_perc)
    cbk = train_monitor(path)

    D_step = args.D_step
    G_step = args.G_step
    model = aae_model(encoder=encoder, decoder=decoder, discriminator=discriminator, latent_dim=latent, dis_steps=D_step, gen_steps=G_step)
    # model.summary()
    model.compile(rec_opt=ae_opt, rec_obj=ae_loss, dis_opt=dc_opt, dis_obj=dis_loss, gen_opt=ge_opt, gen_obj=gen_loss)
    history = model.fit(x_train, epochs=500, batch_size=128, callbacks = [cbk], verbose=0)

    his_doc = pd.DataFrame(history.history)
    with open("training _his.json", mode='w') as f:
        his_doc.to_json(f)
