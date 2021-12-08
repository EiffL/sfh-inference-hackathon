import numpy as np
import tensorflow as tf
import tensorflow_dataset ast tfds

def preprocessing(example):
    
    return tf.reshape(tf.cast(example['SFR_Max'], dtype=tf.float32),(-1,100,1)), \
           tf.reshape(tf.cast(example['SFR_Max'], dtype=tf.float32),(-1,100,1))

def input_fn(mode='train', batch_size=64, dataset_name='sfh'):
    """
    mode: 'train' or 'test'
    """
    if mode == 'train':
        dataset = tfds.load(dataset_name, split='train[:80%]')
        #dataset = dataset.repeat()
        dataset = dataset.shuffle(10000)
    else:
        dataset = tfds.load(dataset_name, split='train[80%:]')
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(preprocessing) # Apply data preprocessing
    dataset = dataset.prefetch(-1)       # fetch next batches while training current one (-1 for autotune)
    return dataset, tf.data.experimental.cardinality(dset).numpy()

def predictor(model, sample_size, nsteps=100):
    res = np.zeros((sample_size, nsteps,1))
    for i in range(nsteps):
        tmp = model(res).sample()
        res[0,i] = tmp[0,i]
    return res

def pass_sample(model, sample, n_pass=100):
    sample = tf.reshape(sample,(1, 100, 1))
    mean = model(sample).mean()
    std = model(sample).stddev()
    p_sample = model(sample).sample()
    return mean, std, p_sample