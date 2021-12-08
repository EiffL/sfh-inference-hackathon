import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def preprocessing(example):
    
    return tf.reshape(example['SFR_Max'],(-1,100,1)), \
           tf.reshape(example['SFR_Max'],(-1,100,1))

def input_fn(mode='train', batch_size=64, dataset_name='sfh', data_dir=None):
    """
    mode: 'train' or 'test'
    """
    if mode == 'train':
        dataset = tfds.load(dataset_name, split='train[:80%]', data_dir=data_dir)
        #dataset = dataset.repeat()
        dataset = dataset.shuffle(10000)
    else:
        dataset = tfds.load(dataset_name, split='train[80%:]', data_dir=data_dir)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(preprocessing) # Apply data preprocessing
    dataset = dataset.prefetch(-1)       # fetch next batches while training current one (-1 for autotune)
    return dataset, tf.data.experimental.cardinality(dset).numpy()

def predictor(model, sample_size, nsteps=100):
    """
    """
    res = np.zeros((sample_size, nsteps,1))
    for i in range(nsteps):
        tmp = model(res).sample()
        res[0,i] = tmp[0,i]
    return res

def pass_sample(model, sample):
    """
    For now only works with 1 sample, not batch !
    """
    sample = tf.reshape(sample,(1, 100, 1))
    mean = model(sample).mean()
    std = model(sample).stddev()
    p_sample = model(sample).sample()
    return mean, std, p_sample

def finish_sample(model, sample, nsteps=100, mode='sample'):
    """
    
    For now only works with 1 sample, not batch !
    
    mode should be either 'sample' or 'mean'
    sample : a 1d Tensor SFR sequence, of cardinality<nsteps
    nsteps : total length of output sequence
    """
    assert sample.numpy().reshape((-1,)).shape[0]<nsteps
    n_remain = nsteps - sample.numpy().reshape((-1,)).shape[0]
    res = np.zeros((1, nsteps,1))
    res[0,:nsteps-n_remain,0] = sample.numpy()
    for i in range(nsteps-n_remain, nsteps):
        if mode=='sample':
            tmp = model(res).sample()
        if mode=='mean':
            tmp = model(res).mean()
        res[0,i] = tmp[0,i]
    return res

