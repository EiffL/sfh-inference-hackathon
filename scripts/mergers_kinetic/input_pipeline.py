import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as tfk
from sfh.datasets.mergers import kinetic



# Global variables used as parameter for preprocessing the data
std_velocity       = 52.98044618680991
mean_velocity      = -0.4956044
mean_velocity_disp = 114.94696
std_velocity_disp  = 34.80651110327937
std_stellar_light  = 30467.57983575861
stellar_light_compression = 3.0

def preprocessing(example):
  import numpy as np

  img0 = tf.math.asinh(example['image'][:,0,:,:] / tf.constant(std_stellar_light) * tf.constant(stellar_light_compression) ) / tf.constant(stellar_light_compression)
  img1 = (example['image'][:,1,:,:] - tf.constant(mean_velocity)) / tf.constant(std_velocity)
  img2 = (example['image'][:,2,:,:] - tf.constant(mean_velocity_disp))/ tf.constant(std_velocity_disp)
  last_major_merger = example['last_major_merger'] / tf.constant(13.6) # Scale ages between 0 and 1

  print(img0.shape)
  return tf.stack([img0, img1, img2], axis=1), last_major_merger

def input_fn(mode='train', batch_size=64):
  """
  mode: 'train' or 'test'
  """
  # Jean-Zay datasets diretory:
  data_dir='/gpfsscratch/rech/qrc/commun/tensorflow_datasets'
  #data_dir='/Users/benjamin/SCRATCH/sfh/content/data/'


  if mode == 'train':
    dataset = tfds.load('mergers_kinetic', split='train[:80%]', data_dir=data_dir) 
    dataset = dataset.repeat()
    dataset = dataset.shuffle(10000)
  else:
    dataset = tfds.load('mergers_kinetic', split='train[:80%]', data_dir=data_dir)

  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.map(preprocessing) # Apply data preprocessing
  dataset = dataset.prefetch(-1) # fetch next batches while training current one (-1 for autotune)

  return dataset