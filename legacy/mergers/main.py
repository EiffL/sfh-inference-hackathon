import tensorflow as tf
import tensorflow.keras as tfk
from utils.input_pipeline import input_fn
from models.model_mse import create_model


# Parameters for the datasets
BATCH_SIZE              = 64
# Parameters for the training
SAVE_CHECKPOINTS        = True 
CHECKPOINTS_PATH        = './checkpoints_mse/'
SAVE_MODEL              = True
MODEL_SAVE_NAME         = 'model_mse.sav'
LEARNING_RATE           = 0.0001
LEARNING_RATE_EXP_DECAY = 0.9
STEPS_PER_EPOCH         = 20000//BATCH_SIZE
N_EPOCHS                = 100



# Enable multi-GPU distributed training
mirrored_strategy = tf.distribute.MirroredStrategy()

# Create the training and testing datasets
dataset_training = input_fn(mode='train', batch_size=BATCH_SIZE)
dataset_testing = input_fn(mode='test', batch_size=BATCH_SIZE)

# Call create_model to generate the model
with mirrored_strategy.scope():
    model = create_model()
# Print the model architecture
model.summary()



# Callback to decrease the learning rate during the training
lr_decay = tfk.callbacks.LearningRateScheduler(lambda epoch: LEARNING_RATE * LEARNING_RATE_EXP_DECAY**epoch,verbose=True)
# Callback to save weights during training
cp_callback = tfk.callbacks.ModelCheckpoint(filepath=CHECKPOINTS_PATH, verbose=1, save_weights_only=True)

# Train the model
with mirrored_strategy.scope():
    if(SAVE_CHECKPOINTS):
        model.fit(dataset_training, validation_data=dataset_testing, steps_per_epoch=STEPS_PER_EPOCH, epochs=N_EPOCHS, callbacks=[lr_decay, cp_callback])
    else:
        model.fit(dataset_training, validation_data=dataset_testing, steps_per_epoch=STEPS_PER_EPOCH, epochs=N_EPOCHS, callbacks=[lr_decay])

# Save model when training finishes 
if(SAVE_MODEL):    
    model.save(MODEL_SAVE_NAME)