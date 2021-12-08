import tensorflow.keras as tfk
from input_pipeline import input_fn
from model import create_model


# Create the train and testing datasets
dataset_training = input_fn(mode='train', batch_size=64)
dataset_testing = input_fn(mode='test', batch_size=64)

# Call to create_model to generate the model
cnn_model = create_model()
# Print model architecture
cnn_model.summary()


# "Hyperparameters"
LEARNING_RATE=0.001 ; LEARNING_RATE_EXP_DECAY=0.9
STEPS_PER_EPOCH=20000//64
EPOCHS = 10

# Callback to decrease the learning rate during the training
lr_decay = tfk.callbacks.LearningRateScheduler(lambda epoch: LEARNING_RATE * LEARNING_RATE_EXP_DECAY**epoch,verbose=True)
# Callback to save weights during training
cp_callback = tfk.callbacks.ModelCheckpoint(filepath='./model_checkpoints/', verbose=1, save_weights_only=True)

# Train the model
cnn_model.fit(dataset_training, validation_data=dataset_testing, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_decay, cp_callback])

# Save model when training finishes 
cnn_model.save('./cnn_model')