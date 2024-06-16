'''model configuration'''

def supervised_vae(input_dim, intermediate_dim, latent_dim, label_dim, alpha, gamma):
    
    class Sampling(layers.Layer):
    #Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    input_shape = (input_dim, )
    inputs = keras.Input(shape = input_shape)
    
    # build encoder
    if isinstance(intermediate_dim, int):
        x = layers.Dense(intermediate_dim, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.1))(inputs)
        x = layers.Dropout(0.3)(x)
    else:
        x = inputs
        for dim in intermediate_dim:
            x = layers.Dense(dim, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.1))(x)
            x = layers.Dropout(0.3)(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    
    #build decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    
    if isinstance(intermediate_dim, int):
        x = layers.Dense(intermediate_dim, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.1))(latent_inputs)
        x = layers.Dropout(0.3)(x)
    else:
        x = latent_inputs
        for dim in reversed(intermediate_dim):
            x = layers.Dense(dim, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.1))(x)
            x = layers.Dropout(0.3)(x)
            
    outputs = layers.Dense(input_dim)(x)
    
    decoder = keras.Model(latent_inputs, outputs, name="decoder")
    #decoder.summary()
    
    #build softmax classifier            
    h_output = layers.Dense(label_dim)(latent_inputs)
    # output_labels = keras.layers.activations.Softmax()(h_output)
    
    classifier = keras.Model(latent_inputs, h_output, name="SoftmaxC")

    class sVAE(keras.Model):
        def __init__(self, encoder, decoder, classifier, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.classifier = classifier
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
            self.prediction_loss_tracker = keras.metrics.Mean(name="prediction_loss")

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
                self.prediction_loss_tracker,
            ]

        def train_step(self, data):
            X , y = data
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(X)
                reconstruction = self.decoder(z)
                y_pred = self.classifier(z)
                
                reconstruction_loss = tf.reduce_mean(tf.square(X - reconstruction) * input_dim)
                
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))
                kl_loss *= alpha 
                
                cce = keras.losses.CategoricalCrossentropy()
                pred_loss = cce(y, keras.ops.softmax(y_pred))
                pred_loss *= gamma
                
                total_loss = reconstruction_loss + kl_loss + pred_loss
                
            grads = tape.gradient(total_loss, self.trainable_weights)
            
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            self.prediction_loss_tracker.update_state(pred_loss)
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
                "pred_loss": self.prediction_loss_tracker.result()
            }
        
    svae = sVAE(encoder, decoder, classifier)
    svae.compile(optimizer=keras.optimizers.Adam())
    
    return svae, encoder, decoder, classifier
