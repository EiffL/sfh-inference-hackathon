"""Keras model implementing PixelCNN."""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers


def generate_model(n_timesteps, *, n_components=2, kernel_size=3,
                   n_dilations=5):
    """Generate the PixelCNN Keras model.

    Parameters
    ----------
    n_timesteps: int
        Number of time steps.
    n_components: int, default 2
        Number of components in the Gausian mixture distribution.
    kernel_size: int, default 3
        Size of the convolution kernel.
    n_dilations: int, default 5
        Number of dilations.

    Returns
    -------
    Keras model

    """
    # Shape of the distribution
    event_shape = [1]
    # Compute how many parameters this distribution requires
    params_size = tfp.layers.MixtureNormal.params_size(
        n_components, event_shape)

    pixel_cnn = keras.Sequential()

    # Shift and cut
    pixel_cnn.add(
        keras.layers.Lambda(
            lambda x: tf.pad(x, paddings=tf.constant([[0, 0], [1, 0], [0, 0]]))
        )
    )
    pixel_cnn.add(
        keras.layers.Lambda(
            lambda x: x[:, :-1, :]
        )
    )

    pixel_cnn.add(
        keras.layers.Conv1D(
            filters=16,
            kernel_size=kernel_size,
            dilation_rate=1,
            padding='causal',
            activation='relu'
        )
    )

    for dilation_idx in range(n_dilations):
        dilation_rate = 2**(dilation_idx+1)
        nb_filters = 2**(dilation_idx+4)

        pixel_cnn.add(
            keras.layers.Conv1D(
                filters=nb_filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu')
        )

    pixel_cnn.add(keras.layers.Dense(params_size))
    pixel_cnn.add(tfp.layers.MixtureNormal(n_components, event_shape))

    pixel_cnn.build(input_shape=(None, n_timesteps, 1))

    # Use the negative log-likelihood as loss function.
    def negloglik(y, q):
        return tf.reduce_sum(-q.log_prob(y), -1)

    pixel_cnn.compile(loss=negloglik, optimizer='adam')

    return pixel_cnn
