import tensorflow as tf
from tensorflow.keras import layers as tfkl
import tensorflow_datasets as tfds
# import my_dataset
import matplotlib.pyplot as plt


def Model(nb_summaries):
    """ Creates a small fully connected network
    """
    return tf.keras.Sequential([
        tfkl.Input(shape=(143)),
        tfkl.Dense(16, activation='relu'),
        tfkl.Dense(32, activation='relu'),
        tfkl.Dense(64, activation='relu'),
        tfkl.Dense(128, activation='relu'),
        tfkl.Dense(nb_summaries, activation='softplus')
        ])


model = Model(1)
print(model.summary)

def preprocessing(example):
    img = tf.math.asinh(example['image'] / tf.constant(scaling) / 3. )
  # We return the image as our input and output for a generative model
  return img, img

def input_fn(mode='train', batch_size=64):
    """
    mode: 'train' or 'test'
    """
    if mode == 'train':
        dataset = tfds.load('sfhsed', split='train[:80%]')
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10000)
    else:
        dataset = tfds.load('sfhsed', split='train[80%:]')

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(preprocessing) # Apply data preprocessing
    dataset = dataset.prefetch(-1)  # fetch next batches while training current one (-1 for autotune)
    return dataset


dset = input_fn()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MSE)

history = model.fit(x=dset['flux'], y=dset['quantile'][4], epochs=20)

plt.plot(history.loss)
