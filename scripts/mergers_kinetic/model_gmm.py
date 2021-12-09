import tensorflow as tf
import tensorflow_probability as tfp
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

def create_model():

    # Number of components in the Gaussian Mixture
    num_components = 4
    # Shape of the distribution
    event_shape = [1]
    # Utility function to compute how many parameters this distribution requires
    params_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)

    #Loss function
    negloglik = lambda x, rv_x: -rv_x.log_prob(x)

    #Define the model
    model = tfk.models.Sequential()
    #1st layer
    model.add(tfk.layers.Conv2D(32, kernel_size=3, padding='same',input_shape=(128,128,2), activation='relu', strides=2))
    model.add(tfk.layers.BatchNormalization())
    #2nd layer
    #model.add(tfk.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    #model.add(tfk.layers.BatchNormalization())
    #1st layer
    #model.add(tfk.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    #model.add(tfk.layers.BatchNormalization())
    #2nd layer
    #model.add(tfk.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    #model.add(tfk.layers.BatchNormalization())
    #Flatten layer
    model.add(tfk.layers.Flatten())
    #Dense
    #model.add(tfk.layers.Dense(512,activation='relu'))
    model.add(tfk.layers.Dense(256,activation='relu'))
    model.add(tfk.layers.Dropout(0.5))

    model.add(tfk.layers.Dense(128,activation='relu'))
    model.add(tfk.layers.Dropout(0.2))

    #Predict a distribution
    model.add(tfkl.Dense(params_size))
    model.add(tfp.layers.MixtureNormal(num_components, event_shape))
    
    model.compile(optimizer='adam',loss=negloglik)

    return model
