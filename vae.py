import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial
import math

import tensorflow as tf
from scipy.special import expit

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import layers
from keras.regularizers import l1,l2, L1L2


def standard_vae(input_dim, intermediate_dim, latent_dim, alpha):
    
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

    class VAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(
                name="reconstruction_loss"
            )
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
            

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                
                reconstruction_loss = tf.reduce_mean(tf.square(data - reconstruction) * input_dim)
                #reconstruction_loss *=  
                
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))
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
        
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    
    return vae,  encoder, decoder
