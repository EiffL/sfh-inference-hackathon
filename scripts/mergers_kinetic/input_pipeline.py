import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as tfk
from sfh.datasets.tng100 import tng100



# Global variables used as parameter for preprocessing the data
mean_stellar_light = 167915.38
mean_velocity      = 1.2179685
mean_velocity_disp = 80.03517
std_velocity       = 47.085832102830075
std_velocity_disp  = 29.16650153197798
std_stellar_light  = 18212.175615239412
stellar_light_compression = 3.0

def preprocessing(example):
  import numpy as np

  

  img0 = example['stellar_light']
  img1 = example['velocity_map']
  img2 = example['velocity_dispersion_map']

  # Replace NaNs by zeros
  img0 = tf.where(tf.math.is_nan(img0), tf.zeros_like(img0), img0)
  img1 = tf.where(tf.math.is_nan(img1), tf.zeros_like(img1), img1)
  img2 = tf.where(tf.math.is_nan(img2), tf.zeros_like(img2), img2)
  # Replace InFs by zeros
  img0 = tf.where(tf.math.is_inf(img0), tf.zeros_like(img0), img0)
  img1 = tf.where(tf.math.is_inf(img1), tf.zeros_like(img1), img1)
  img2 = tf.where(tf.math.is_inf(img2), tf.zeros_like(img2), img2)

  # Normalize data
  img0 = tf.math.asinh(img0 / tf.constant(std_stellar_light) * tf.constant(stellar_light_compression) ) / tf.constant(stellar_light_compression)
  img1 = (img1 - tf.constant(mean_velocity)) / tf.constant(std_velocity)
  img2 = (img2 - tf.constant(mean_velocity_disp))/ tf.constant(std_velocity_disp)

  # Reduce images size (drop stellar light)
  img = tf.stack([img1, img2], axis=-1)
  # Resize images
  img = tf.image.resize(img, [128, 128])

  t2 = example['last_major_merger']
  if(t2<0.8):
    t2=0.8
  return img, tf.constant(t2)

def input_fn(mode='train', batch_size=64):
  """
  mode: 'train' or 'test'
  """
  # Jean-Zay datasets diretory:
  data_dir='/gpfsscratch/rech/qrc/commun/tensorflow_datasets'
  #data_dir='/Users/benjamin/SCRATCH/sfh/content/data/'


  if mode == 'train':
    dataset = tfds.load('tng100', split='train[:80%]', data_dir=data_dir) 
    dataset = dataset.map(preprocessing) # Apply data preprocessing
    dataset = dataset.repeat()
    dataset = dataset.shuffle(10000)
  else:
    dataset = tfds.load('tng100', split='train[80%:]', data_dir=data_dir)
    dataset = dataset.map(preprocessing) # Apply data preprocessing
  
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(-1) # fetch next batches while training current one (-1 for autotune)

  return dataset