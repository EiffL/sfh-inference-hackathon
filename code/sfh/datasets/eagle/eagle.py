import os
import glob
import tensorflow as tf
import h5py as h5py
import tensorflow_datasets as tfds
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy.io import fits
import numpy as np
import pandas as pd # To extract the SnapNumLastMajorMerger values from TNG100_SDSS_MajorMergers.csv

_DESCRIPTION = """
#Data of EAGLE galaxies
"""

_CITATION = ""
_URL = "https://github.com/mhuertascompany/sfh-inference-hackathon"

## My functions added ##


def find_summaries(mass, time, percentiles=np.linspace(0.1, 0.9, 9)):

    ''' compute the half mass and the half time of a galaxy 
          Input: 
                - mass: array. The mass history of the galaxy.
                - time: array. The corresponding time for the galaxy history.
                - percentiles: array. The summaries you want to predict by default 0.1, 0.2,..., 0.9. 
          Output: the time of the summaries, the corresponding masses, and the index of the mass/time summary.
    '''

    summary_masses = []
    summary_times = []
    summary_indices = []
    for percentile in percentiles:
        summary_mass = min(mass, key=lambda x: abs(x-mass[0]*percentile))  # find mass closest to the half mass
        summary_masses.append(summary_mass)
        summary_mass_indices = np.where(mass == summary_mass)[0]  # find the corresponding indices
        summary_mass_index = summary_mass_indices[0]  # chose the first index for the half mass
        summary_indices.append(summary_mass_index)
        summary_time = time[summary_mass_index]  # find the corresponding half time
        summary_times.append(summary_time)

    return np.array(summary_times).astype('float32')
#######################

class Eagle(tfds.core.GeneratorBasedBuilder):
  """Eagle galaxy dataset"""  

  VERSION = tfds.core.Version("4.0.0")
  RELEASE_NOTES = {'4.0.0': 'Sort wl.',}
  MANUAL_DOWNLOAD_INSTRUCTIONS = "Nothing to download. Dataset was generated at first call."
  
  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    N_TIMESTEPS = 100
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        homepage=_URL,
        citation=_CITATION,
        # Two features: image with 3 channels (stellar light, velocity map, velocity dispersion map)
        #  and redshift value of last major merger
        features=tfds.features.FeaturesDict({
            #'noiseless_griz': tfds.features.Tensor(shape=(128, 128, 4), dtype=tf.float32),
            #'stellar_light': tfds.features.Tensor(shape=(512, 512), dtype=tf.float32),
            #'velocity_map': tfds.features.Tensor(shape=(512, 512), dtype=tf.float32),
            #'velocity_dispersion_map': tfds.features.Tensor(shape=(512, 512), dtype=tf.float32),
            "sed": tfds.features.Tensor(shape=(125,), dtype=tf.float32),
            "time": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            #"SFR_halfRad": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            #"SFR_Rad": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            "SFR_Max": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            #"Mstar_Half": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            "Mstar": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            'mass_quantiles': tfds.features.Tensor(shape=(9,), dtype=tf.float32),
            'inds_valid': tfds.features.Tensor(shape=(143,), dtype=tf.bool),
            'wl_sort': tfds.features.Tensor(shape=(143,), dtype=tf.float32),
            #'last_major_merger': tf.float32,
            'object_id': tf.int32
        }),
        supervised_keys=('noiseless_griz', 'last_major_merger'), 
    )

  def _split_generators(self, dl):
    """Returns generators according to split"""
    return {tfds.Split.TRAIN: self._generate_examples(str(dl.manual_dir))}

  def _generate_examples(self, root_path):
    """Yields examples."""

    
    # read EAGLE hd5y + filter names + filter wavelength
    hf = h5py.File(root_path+'/dataMagnitudes_2000kpc_EMILES_PDXX_DUST_CH_028_z000p000.hdf5', 'r')
    wl = np.loadtxt(root_path+"/wl.csv")
    inds = np.argsort(wl)
    wl_sort = wl[inds]
    text_file = open(root_path+"/fnames.csv", "r")
    fname_list = text_file.readlines()
    sfh = hf.get('Data/SFhistory')
    tbins = hf.get('Data/SFbins')
    time = (tbins[1:] + tbins[:-1] )/2.
    deltat=tbins[1:] - tbins[:-1]

    mstar  = hf.get('Data/StellarMassNew')  
    
    # sfh
    #sfh = hf.get('Data/SFhistory')
    nobjects = sfh.shape[0]
    #tbins = hf.get('Data/SFbins')
    #time = (tbins[1:] + tbins[:-1] )/2.
    
   
        
   

    for i in range(len(mstar)):
        object_id = i

        try:
            
            
            if np.log10(mstar[i])<9.5:
                continue

            # sed
    
            mag = [] 
            for f in fname_list:
                mag.append(hf['Data'][f.strip()][i])
            app_mag = np.array(mag)+5*(np.log10(20e6)-1) #assume at 20pc
            flux = 10**(.4*(-app_mag+8.90)) #convert to Jy
            flux = flux[inds]  # sorting the flux
            inds_valid = np.isfinite(flux)
            flux = flux[inds_valid]
            
            
            example = {'sed': flux}
            example.update({'inds_valid': np.array(inds_valid).astype('bool')})
            example.update({'wl_sort': np.array(wl_sort).astype('float32')})
            #example.update({'sed': np.array(flux).astype('float32')})

        
            #mstar growth
     
    
            mgrowth = np.cumsum(deltat*sfh[i])
            
   

            #sfh
            time_norm = time / np.max(time)*100
            xvals = np.linspace(0, 100, 100)
            yinterp = np.interp(xvals, time_norm, sfh[i]) 
            minterp = np.interp(xvals, time_norm, mgrowth)    
            example.update({'time': np.array(xvals).astype('float32')})
            example.update({'SFR_Max': np.array(yinterp).astype('float32')})
            example.update({'Mstar': np.array(minterp).astype('float32')})
            
            
            #quantiles
            mass_history_summaries = find_summaries(example['Mstar'],
                                                example['time'])
            #last_over_max = example['Mstar'][0]/np.max(example['Mstar'])
            example.update({'mass_quantiles': mass_history_summaries,
                        #'last_over_max': last_over_max,
                        'object_id': object_id})
            
            
            


    

            yield object_id, example
        except:
            continue      
