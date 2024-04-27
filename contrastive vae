'''model configuration'''
def contrastive_vae(input_dim, intermediate_dim, latent_dim_s, latent_dim_z, alpha):
    
    class Sampling(layers.Layer):
    #Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
    class Zeros(layers.Layer):
        def call(self, inputs):
            batch = tf.shape(inputs)[0]
            dim = tf.shape(inputs)[1]
            return tf.zeros(shape=(batch, dim))
    
    # input layers
    input_shape = (input_dim, )
    fg_inputs = keras.Input(shape=input_shape, name='fg_input')
    bg_inputs = keras.Input(shape=input_shape, name='bg_input')    
    
    # build s_encoder
    s_h_layers = fg_inputs
    for dim in intermediate_dim:
        s_h_layers = layers.Dense(dim, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.1))(s_h_layers) # use_bias=False
        s_h_layers = layers.Dropout(0.3)(s_h_layers)
        # s_h_layers = layers.add([s_h_layers, block_1_output])

    fg_s_mean = layers.Dense(latent_dim_s, name='fg_s_mean')(s_h_layers)
    fg_s_log_var = layers.Dense(latent_dim_s, name='fg_s_log_var')(s_h_layers)
    fg_s = Sampling()([fg_s_mean, fg_s_log_var])
    
    s_encoder = keras.Model(fg_inputs, [fg_s_mean, fg_s_log_var, fg_s], name="s_encoder")
    
    
    # create zero_like_encoder
    #zero_shape = layers.Dense(latent_dim, name='zero_shape')(bg_inputs)
    #zeros = Zeros()(zero_shape)
    # zeros = layers.Lambda(lambda x: keras.ops.zeros(shape = (, latent_dim)))(bg_inputs) #  = batch size in training
    #zero_like_encoder = keras.Model(bg_inputs, zeros, name="zero_like_encoder")
    
    # build z_encoder
    # create hidden layers
    z_h_layers = keras.Sequential()
    for dim in intermediate_dim:
        z_h_layers.add(layers.Dense(dim, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.1)))
        z_h_layers.add(layers.Dropout(0.3))

    def z_encoder_func(inputs, mean_name, var_name):
        x_h = z_h_layers(inputs)
        x_mean = layers.Dense(latent_dim_z, name= mean_name)(x_h)
        x_log_var = layers.Dense(latent_dim_z, name= var_name)(x_h)
        x = Sampling()([x_mean, x_log_var])
        return x_mean, x_log_var, x
    
    fg_z_mean, fg_z_log_var, fg_z = z_encoder_func(fg_inputs, 'fg_z_mean', 'fg_z_log_var')
    bg_z_mean, bg_z_log_var, bg_z = z_encoder_func(bg_inputs, 'bg_z_mean', 'bg_z_log_var')
    
    z_encoder = keras.Model(fg_inputs, [fg_z_mean, fg_z_log_var, fg_z], name="z_encoder")
    
    
    #build decoder
    latent_inputs = keras.Input(shape=(latent_dim_s + latent_dim_z,), name='samples')

    cvae_h = latent_inputs
    for dim in reversed(intermediate_dim):
        cvae_h = layers.Dense(dim, activation='relu',kernel_regularizer=L1L2(l1=0.01, l2=0.1))(cvae_h)
        cvae_h = layers.Dropout(0.3)(cvae_h)

    cvae_outputs = layers.Dense(input_dim)(cvae_h)

    decoder = keras.Model(latent_inputs, cvae_outputs, name='decoder')
    
    #build decoder_bg
    #latent_inputs_bg = keras.Input(shape=(latent_dim,), name='samples_bg')
    #cvae_h_bg = latent_inputs_bg
    #for dim in reversed(intermediate_dim):
      #  cvae_h_bg = layers.Dense(dim, activation='relu',kernel_regularizer=L1L2(l1=0.01, l2=0.1))(cvae_h_bg)
     #   cvae_h_bg = layers.Dropout(0.3)(cvae_h_bg)

    #cvae_outputs_bg = layers.Dense(input_dim)(cvae_h_bg)

    #decoder_bg = keras.Model(latent_inputs_bg, cvae_outputs_bg, name='decoder_bg')


    # initialize model
    class cVAE(keras.Model):
        def __init__(self, s_encoder, z_encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.s_encoder = s_encoder
            self.z_encoder = z_encoder
            self.decoder = decoder
            # self.decoder_bg = decoder_bg
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
            

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        def train_step(self, data): # data = fg, bg
            fg, bg = data
            with tf.GradientTape() as tape:
                fg_s_mean, fg_s_log_var, fg_s = self.s_encoder(fg)
                fg_z_mean, fg_z_log_var, fg_z = self.z_encoder(fg)
                
                zero = Zeros()(fg_s)
                bg_z_mean, bg_z_log_var, bg_z = self.z_encoder(bg)
                
                reconstruction_fg = self.decoder(layers.Concatenate(axis=-1)([fg_s, fg_z]))
                reconstruction_bg = self.decoder(layers.Concatenate(axis=-1)([zero, bg_z]))
                
                reconstruction_loss = tf.reduce_mean(tf.square(fg - reconstruction_fg) * input_dim)
                reconstruction_loss += tf.reduce_mean(tf.square(bg - reconstruction_bg) * input_dim)
                
                kl_loss = tf.reduce_sum((-0.5 * (1 + fg_s_log_var - tf.square(fg_s_mean) - tf.exp(fg_s_log_var))), axis=-1)
                kl_loss += tf.reduce_sum((-0.5 * (1 + fg_z_log_var - tf.square(fg_z_mean) - tf.exp(fg_z_log_var))), axis=-1)
                kl_loss += tf.reduce_sum((-0.5 * (1 + bg_z_log_var - tf.square(bg_z_mean) - tf.exp(bg_z_log_var))), axis=-1)
                kl_loss = tf.reduce_mean(kl_loss)
                kl_loss *= alpha
                
                total_loss = reconstruction_loss + kl_loss
                
            grads = tape.gradient(total_loss, self.trainable_weights)
            
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)

            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }
        
    cvae = cVAE(s_encoder, z_encoder, decoder)
    cvae.compile(optimizer=keras.optimizers.Adam())
    
    return cvae,  s_encoder, z_encoder, decoder
