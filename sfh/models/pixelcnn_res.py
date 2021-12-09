"""Keras model implementing PixelCNN."""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def resblock(x, kernel_size, nb_filters, dilation_rate):
    
    
    fx = tf.keras.layers.Conv1D(filters=nb_filters,
               kernel_size=kernel_size,
               dilation_rate=dilation_rate[0],
               padding='causal',
               activation='relu')(x)    
    fx = tf.keras.layers.Conv1D(filters=nb_filters,
               kernel_size=kernel_size,
               dilation_rate=dilation_rate[1],
               padding='causal',
               activation='relu')(fx)    

    x = tf.keras.layers.Conv1D(filters=nb_filters,
               kernel_size=1,
               padding='same',
               activation='relu')(x) 
    out = tf.keras.layers.Add()([x,fx])
    #out = tf.keras.layers.ReLU()(out)
    return out

class MaskedConvLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(MaskedConvLayer, self).__init__()
        self.conv = layers.Conv1D(**kwargs)

    def build(self, input_shape):
        self.conv.build(input_shape)
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.ones(shape=kernel_shape)
        self.mask[-1, ...] = 0
        #maybe mask should be tf to make it faster?

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

def generate_model(kernel_size=3, n_dilations=5, optimizer='adam'):
    n_timesteps = 100
    num_components = 2
    # Shape of the distribution
    event_shape = [1]
    # Utility function to compute how many parameters this distribution requires
    params_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)


    paddings = tf.constant([[0,0],[1, 0], [0, 0]])

    #shift_layer = tf.keras.layers.Lambda(lambda x: tf.pad( tf.roll(x, shift=+1, axis=-2)[:,1:,:], paddings))
    #shift_layer = tf.keras.layers.Lambda(lambda x: tf.roll(x, shift=+1, axis=-2))

    shift_layer = tf.keras.layers.Lambda(lambda x: tf.pad( x, paddings))
    cut_layer = tf.keras.layers.Lambda(lambda x: x[:,:-1,:])

    inputs = keras.Input(shape=(100,1), dtype='float32')

    kernel_size=3

    #x = shift_layer(inputs)
    #x = cut_layer(x)
    x = MaskedConvLayer(filters=16,
           kernel_size=kernel_size,
           dilation_rate=2,
           padding='causal',
           activation='relu')(inputs)

    for dilation_rates, nb_filters in zip([[4, 8], [16, 32]], [32, 64]):

        x = resblock(x, kernel_size, nb_filters, dilation_rates)

        


    x = keras.layers.Dense(params_size)(x)
    out = tfp.layers.MixtureNormal(num_components, event_shape)(x)

    pixel_cnn = keras.Model(inputs, out)
    pixel_cnn.build(input_shape=(None, n_timesteps, 1))

    # Use the negative log-likelihood as loss function.
    def negloglik(y, q):
        return tf.reduce_sum(-q.log_prob(y), -1)

    pixel_cnn.compile(loss=negloglik, optimizer=optimizer)

    return pixel_cnn
