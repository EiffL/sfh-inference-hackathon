import tensorflow.keras as tfk

def create_model():
  model = tfk.models.Sequential()
  model.add(tfk.layers.Conv2D(32, kernel_size=5, padding='same',
                                   input_shape=(64,64,5), activation='elu', strides=2))
  model.add(tfk.layers.BatchNormalization())
  
  model.add(tfk.layers.Conv2D(64, kernel_size=3, padding='same', 
                              activation='elu'))
  model.add(tfk.layers.BatchNormalization())
  
  model.add(tfk.layers.Conv2D(128, kernel_size=3, padding='same', strides=2, 
                                   activation='elu'))
  model.add(tfk.layers.BatchNormalization())  

  model.add(tfk.layers.Conv2D(256, kernel_size=3, padding='same', 
                                   activation='elu', strides=2))
  model.add(tfk.layers.BatchNormalization())

  model.add(tfk.layers.Conv2D(512, kernel_size=3, padding='same', 
                                   activation='elu', strides=2))
  model.add(tfk.layers.BatchNormalization())
  
  model.add(tfk.layers.Flatten())
  model.add(tfk.layers.Dense(512))
  model.add(tfk.layers.Activation('relu'))
  model.add(tfk.layers.Dense(256))
  model.add(tfk.layers.Activation('relu'))
  model.add(tfk.layers.Dense(1))
  
  model.compile(optimizer='adam', # learning rate will be set by LearningRateScheduler
                loss=tfk.metrics.mse)
  return model
