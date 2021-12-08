import tensorflow as tf
import tensorflow_datasets as tfds
from sfh.datasets.mergers import kinetic

# Using a mapping function to apply preprocessing to our data
def preprocessing(example):
  img = example['image']
  return img, example['last_major_merger'], example['object_id']

def input_fn(mode='train', batch_size=64):
  """
  mode: 'train' or 'test'
  """
  # Jean-Zay datasets diretory:
  data_dir='/gpfsscratch/rech/qrc/commun/tensorflow_datasets'

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