import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def preprocessing(example):
    return tf.reshape(example['SFR_Max'],(-1,100,1)), \
           tf.reshape(example['SFR_Max'],(-1,100,1))

def preprocessing_wmass(example):
    mass = example['Mstar'][:,-1]
    mass_half = example['Mstar_Half'][:,-1]
    tiler = tf.constant([100])
    mass = tf.reshape(tf.tile(mass, tiler),(-1,100,1))
    mass_half = tf.reshape(tf.tile(mass_half, tiler),(-1,100,1))
    sfr = tf.reshape(example['SFR_Max'],(-1,100,1))
    res = tf.concat([sfr, mass, mass_half], axis=2)
    return res, res

def input_fn(mode='train', batch_size=64, 
             dataset_name='sfh', data_dir=None,
             include_mass=False):
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
    if include_mass:
        dataset = dataset.map(preprocessing_wmass) # Apply data preprocessing
    else : 
        dataset = dataset.map(preprocessing)
    dataset = dataset.prefetch(-1)       # fetch next batches while training current one (-1 for autotune)
    return dataset

def predictor(model, sample_size, nsteps=100, n_channels=1, mode='sample'):
    """
    mode should be either 'sample' or 'mean'
    """
    assert mode in ['sample', 'mean']
    res = np.zeros((sample_size, nsteps, n_channels))
    if n_channels==3:
        res[:,:,1] = np.random.uniform(10**(8.5-10), 1, size=(sample_size,))
        res[:,:,2] = res[:,:,1]*np.random.uniform(0.6, 0.7, size=(sample_size,))
    for i in range(nsteps):
        if mode=='sample':
            tmp = model(res).sample()
        if mode=='mean':
            tmp = model(res).mean()
        res[:,i,0] = tmp[:,i]
    return res[:,:,0]

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

