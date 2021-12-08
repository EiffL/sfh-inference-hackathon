import tensorflow.keras as tfk
from input_pipeline import input_fn
from model import create_model



dataset_training = input_fn('train')
dataset_testing = input_fn('test')

cnn_model = create_model()
cnn_model.summary()


# Hyperparameters
LEARNING_RATE=0.001 ; LEARNING_RATE_EXP_DECAY=0.9
STEPS_PER_EPOCH=20000//64
EPOCHS = 10

lr_decay = tfk.callbacks.LearningRateScheduler(lambda epoch: LEARNING_RATE * LEARNING_RATE_EXP_DECAY**epoch,verbose=True)
cnn_model.fit(dataset_training, validation_data=dataset_testing, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_decay,])