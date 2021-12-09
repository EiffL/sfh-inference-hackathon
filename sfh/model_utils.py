import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def preprocessing(example):
    return tf.reshape(example['SFR_Max'],(-1,100,1)), \
           tf.reshape(example['SFR_Max'],(-1,100,1))

def preprocessing_wmass(example):
    sfr = tf.math.add(tf.reshape(example['SFR_Max'],(-1,100,1)), 1e-5)
    mass = tf.reshape(tf.math.log(example['Mstar'][:,0]), (-1,1))
    sfr_z0 = tf.reshape(sfr[:,0,0], (-1,1))
    time = example['time']
    
    tiler = tf.constant([1, 100])
    
    mass = tf.reshape(tf.tile(mass, tiler),(-1,100,1))
    sfr_z0 = tf.reshape(tf.tile(sfr_z0, tiler),(-1,100,1))
    time = tf.reshape(time,(-1,100,1))
    
    res = tf.concat([sfr, mass, sfr_z0, time], axis=2)
    return res, res

def preprocessing_wmass_atan(example):
    sfr = tf.scalar_mul(2/np.pi, tf.math.atan(tf.math.add(tf.reshape(example['SFR_Max'],(-1,100,1)), 1e-5)))
    mass = tf.reshape(tf.math.log(example['Mstar'][:,0]), (-1,1))
    #print(sfr[:,0].get_shape, example['Mstar'][:,0].get_shape())

    sfr_z0 = tf.reshape(sfr[:,0,0], (-1,1))
    time = example['time']
    tiler = tf.constant([1, 100])
    
    mass = tf.reshape(tf.tile(mass, tiler),(-1,100,1))
    sfr_z0 = tf.reshape(tf.tile(sfr_z0, tiler),(-1,100,1))
    time = tf.reshape(time,(-1,100,1))
    
    res = tf.concat([sfr, mass, sfr_z0, time], axis=2)
    return res, res

def input_fn(mode='train', batch_size=64, 
             dataset_name='sfh', data_dir=None,
             include_mass=False, arctan=False):
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
    if include_mass and arctan:
        dataset = dataset.map(preprocessing_wmass_atan) # Apply data preprocessing
    elif include_mass:
        dataset = dataset.map(preprocessing_wmass)
    else : 
        dataset = dataset.map(preprocessing)
    dataset = dataset.prefetch(-1)       # fetch next batches while training current one (-1 for autotune)
    return dataset

def predictor(model, sample_size, nsteps=100, n_channels=1, mode='sample', 
              arctan=False, M0=None, SFR0=None, time=None):
    """
    mode should be either 'sample' or 'mean'
    M0 : have to be in natural Mstar unit
    SFR0 : should be in the natural dataset unit
    """
    assert mode in ['sample', 'mean']
    res = np.zeros((sample_size, nsteps, n_channels))
    if n_channels>=3:
        if M0 is not None: res[:,:,1] = np.log(M0/1e10)
        else : res[:,:,1] = np.log(10**(np.random.uniform(8.5, 10, size=(1,))-10))
        if SFR0 is not None: res[:,:,2] = np.arctan(SFR0)*2/np.pi
        else: res[:,:,2] = np.arctan(np.random.uniform(0.0001, 1, size=(1,)))*2/np.pi
        res[:,0,0] =  res[:,0,2]
    if n_channels==4:
        res[:,:,-1] = time
    for i in range(1,nsteps):
        if mode=='sample':
            tmp = model(res).sample()
        if mode=='mean':
            tmp = model(res).mean()
        res[:,i,0] = tmp[:,i]
    if not arctan :
        return res[:,:,0]
    else :
        return np.tan(res[:,:,0]*np.pi/2), res[:,0,1], res[:,0,2]

def pass_sample(model, sample):
    """
    For now only works with 1 sample, not batch !
    """
    sample = tf.reshape(sample,(1, 100, 4))
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
    assert sample.numpy().reshape((-1,4)).shape[0]<nsteps
    n_remain = nsteps - sample.numpy().reshape((-1,4)).shape[0]
    res = np.zeros((1, nsteps,4))
    res[0,:nsteps-n_remain,0] = sample.numpy()
    for i in range(nsteps-n_remain, nsteps):
        if mode=='sample':
            tmp = model(res).sample()
        if mode=='mean':
            tmp = model(res).mean()
        res[0,i] = tmp[0,i]
    return res

