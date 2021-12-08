"""Keras model implementing PixelCNN."""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers


def generate_model(n_timesteps, *, n_channels=1, n_components=2, kernel_size=3,
                   n_dilations=5, list_of_dilation_rates=None,
                   list_of_filters=None):
    """Generate the PixelCNN Keras model.

    Parameters
    ----------
    n_timesteps : int
        Number of time steps.
    n_channels : int, default 1
        Number of channels in the dataset
    n_components : int, default 2
        Number of components in the Gaussian mixture distribution.
    kernel_size : int, default 3
        Size of the convolution kernel.
    n_dilations : int, default 5
        Number of dilated convolutions to do. For each convolution, the
        dilation rate is 2**idx+1 and the number of filters is 2**idx+4.
    list_of_dilation_rates : list of int or None, default None
        List of the dilation rates to use in the dilated convolutions. If not
        None, the n_dilations is not used and filters must be given with the
        same size.
    list_of_filters : list of int or None, default None
        List of the filter number for each of the dilated convolutions. Must be
        of the same size as list_of_dilation_rates

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

    if list_of_dilation_rates is None:
        list_of_dilation_rates = [2**(i+1) for i in range(n_dilations)]
        list_of_filters = [2**(i+4) for i in range(n_dilations)]
    elif len(list_of_filters) != len(list_of_dilation_rates):
        raise ValueError(
            "filters and list_of_dilation_rates must have the same length")

    for dilation_rate, nb_filters in zip(list_of_dilation_rates,
                                         list_of_filters):
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

    pixel_cnn.build(input_shape=(None, n_timesteps, n_channels))

    # Use the negative log-likelihood as loss function.
    def negloglik(y, q):
        return tf.reduce_sum(-q.log_prob(y), -1)

    pixel_cnn.compile(loss=negloglik, optimizer='adam')

    return pixel_cnn
