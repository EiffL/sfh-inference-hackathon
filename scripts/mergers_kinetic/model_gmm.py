import tensorflow.keras as tfk
import tensorflow_probability as tfp

def create_model():


    # Number of components in the Gaussian Mixture
    num_components = 16
    # Shape of the distribution
    event_shape = [1]
    # Utility function to compute how many parameters this distribution requires
    params_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)
    # Loss function
    negloglik = lambda y, q: -q.log_prob(y)

    model = tfk.models.Sequential()
    model.add(tfk.layers.Conv2D(32, kernel_size=5, padding='same', input_shape=(128,128,2), activation='elu', strides=2))
    model.add(tfk.layers.BatchNormalization())
    
    model.add(tfk.layers.Conv2D(64, kernel_size=3, padding='same', activation='elu'))
    model.add(tfk.layers.BatchNormalization())
    
    model.add(tfk.layers.Conv2D(128, kernel_size=3, padding='same', strides=2, activation='elu'))
    model.add(tfk.layers.BatchNormalization())  

    model.add(tfk.layers.Conv2D(256, kernel_size=3, padding='same', activation='elu', strides=2))
    model.add(tfk.layers.BatchNormalization())

    model.add(tfk.layers.Conv2D(512, kernel_size=3, padding='same', activation='elu', strides=2))
    model.add(tfk.layers.BatchNormalization())
    
    model.add(tfk.layers.Flatten())
    model.add(tfk.layers.Dense(256))
    model.add(tfk.layers.Activation('relu'))
    model.add(tfk.layers.Dense(128))
    model.add(tfk.layers.Activation('relu'))
    model.add(tfk.layers.Dense(128))
    model.add(tfk.layers.Activation('tanh'))
    model.add(tfk.layers.Dense(params_size))
    
    model.compile(optimizer='adam', loss=negloglik)
    return model
