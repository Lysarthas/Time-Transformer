import tensorflow as tf

class aae_model(tf.keras.Model):
    def __init__(self, encoder, decoder, discriminator, latent_dim, dis_steps=5, gen_steps=2, gp_weight=10.0, lambda_ae = 0.999, lambda_gen=0.001):
        super(aae_model, self).__init__()
        self.enc = encoder
        self.dec = decoder
        self.dis = discriminator
        self.z_dim = latent_dim
        self.d_step = tf.Variable(dis_steps, trainable=False, dtype='float32')
        self.g_step = tf.Variable(gen_steps, trainable=False, dtype='float32')
        self.gp_w = gp_weight
        self.lambda_ae = lambda_ae
        self.lambda_gen = lambda_gen

    def compile(self, rec_opt, rec_obj, dis_opt, dis_obj, gen_opt, gen_obj):
        super(aae_model, self).compile()
        self.rec_optimizer = rec_opt
        self.dis_optimizer = dis_opt
        self.gen_optimizer = gen_opt
        self.rec_loss_fn = rec_obj
        self.dis_loss_fn = dis_obj
        self.gen_loss_fn = gen_obj
        self.rec_loss_metric = tf.keras.metrics.Mean()
        self.dis_loss_metric = tf.keras.metrics.Mean()
        self.gen_loss_metric = tf.keras.metrics.Mean()
        self.dis_acc_metric = tf.keras.metrics.Mean()

    def summary(self):
        self.enc.summary()
        self.dec.summary()
        self.dis.summary()


    def gradient_penalty(self, batch_num, real_ts, fake_ts):
        alpha = tf.random.uniform(shape=[batch_num, 1], minval=0.0, maxval=1.0)
        dif = real_ts - fake_ts
        interpolate = fake_ts + alpha * dif

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolate)
            pred = self.dis(interpolate, training=True)

        grad = gp_tape.gradient(pred, [interpolate])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]) + 1e-08)
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    def reduce_d_step(self):
        self.d_step -= 1

    def train_step(self, batch_x):

        d_s = tf.keras.backend.get_value(self.d_step)
        g_s = tf.keras.backend.get_value(self.g_step)
        batch_size = tf.shape(batch_x)[0]

        # train autoencoder
        with tf.GradientTape() as rec_tape:
            latent_x = self.enc(batch_x, training=True)
            x_rec = self.dec(latent_x, training=True)
            l_ae = self.rec_loss_fn(batch_x, x_rec)
        
        trainable_variables = self.enc.trainable_variables+self.dec.trainable_variables
        rec_grad = rec_tape.gradient(l_ae, trainable_variables)
        self.rec_optimizer.apply_gradients(
            zip(rec_grad, trainable_variables)
        )

        #train discriminator
        for _ in range(int(d_s)):
            with tf.GradientTape() as dis_tape:
                real_distribution = tf.random.normal([batch_size, self.z_dim], 0.0, 1.0)
                latent_x = self.enc(batch_x, training=True)
                dis_real = self.dis(real_distribution, training=True)
                dis_fake = self.dis(latent_x, training=True)
                d_cost_real = self.dis_loss_fn(tf.ones_like(dis_real), dis_real)
                d_cost_fake = self.dis_loss_fn(tf.zeros_like(dis_fake), dis_fake)
                l_dis = 0.5*(d_cost_real+d_cost_fake)

            dis_grad = dis_tape.gradient(l_dis, self.dis.trainable_variables)
            self.dis_optimizer.apply_gradients(
                zip(dis_grad, self.dis.trainable_variables)
            )
            # clip_D = [p.assign(tf.clip_by_value(p, -0.05, 0.05)) for p in self.dis.weights]
        
        #train generator
        for _ in range(int(g_s)):
            with tf.GradientTape() as gen_tape:
                latent_x = self.enc(batch_x, training=True)
        #         dis_fake = self.dis(latent_x, training=True)
        #         l_gen = self.gen_loss_fn(dis_fake)
                dis_fake = self.dis(latent_x, training=True)
                l_gen = self.gen_loss_fn(tf.ones_like(dis_fake), dis_fake)

            gen_grad = gen_tape.gradient(l_gen, self.enc.trainable_variables)
            self.gen_optimizer.apply_gradients(
                zip(gen_grad, self.enc.trainable_variables)
            )

        self.rec_loss_metric.update_state(l_ae)
        self.dis_loss_metric.update_state(l_dis)
        self.gen_loss_metric.update_state(l_gen)

        return {
            "rec_loss": self.rec_loss_metric.result(),
            "dis_loss": self.dis_loss_metric.result(),
            "gen_loss": self.gen_loss_metric.result(),
            # "d_step": self.d_step
        }