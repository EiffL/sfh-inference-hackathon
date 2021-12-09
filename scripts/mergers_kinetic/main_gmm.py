import tensorflow as tf
import tensorflow.keras as tfk
from input_pipeline import input_fn
from model_gmm import create_model as create_model_gmm


# Enable multi-GPU distributed training
mirrored_strategy = tf.distribute.MirroredStrategy()

# Create the train and testing datasets
dataset_training = input_fn(mode='train', batch_size=128)
dataset_testing = input_fn(mode='test', batch_size=128)

# Call to create_model to generate the model
with mirrored_strategy.scope():
    model = create_model_gmm()
# Print model architecture
model.summary()

# "Hyperparameters"
LEARNING_RATE=0.0001 ; LEARNING_RATE_EXP_DECAY=0.9
STEPS_PER_EPOCH=20000//128
EPOCHS = 100

# Callback to decrease the learning rate during the training
lr_decay = tfk.callbacks.LearningRateScheduler(lambda epoch: LEARNING_RATE * LEARNING_RATE_EXP_DECAY**epoch,verbose=True)
# Callback to save weights during training
cp_callback = tfk.callbacks.ModelCheckpoint(filepath='./model_checkpoints_gmm/', verbose=1, save_weights_only=True)

# Train the model
with mirrored_strategy.scope():
    model.fit(dataset_training, validation_data=dataset_testing, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[cp_callback,lr_decay])

# Save model when training finishes 
    model.save('./model_gmm')