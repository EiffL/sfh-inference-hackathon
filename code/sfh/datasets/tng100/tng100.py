import os
import glob
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy.io import fits
import numpy as np
import pandas as pd # To extract the SnapNumLastMajorMerger values from TNG100_SDSS_MajorMergers.csv

_DESCRIPTION = """
#Data of TNG100 galaxy
"""

_CITATION = ""
_URL = "https://github.com/mhuertascompany/sfh-inference-hackathon"

## My functions added ##
def stack_bands(img_dir,gal_id):
  """
  For a given image path and galaxy id, stacks the four bands g,r,i,z into a single image
  Input: img_dir (str): path to the directory containing the noisy images
         gal_id (int): number of the image of galaxy for which you want to stack bands
  Output: im (numpy ndarray): resulting image with the four stacked bands
  """
  filters=['g','r','i','z']
  filenames=[img_dir+filters[i]+"/broadband_"+str(gal_id)+'.fits_'+filters[i]+"_band.fits" for i in range(len(filters))]
  #Stack the bands together
  im=[fits.getdata(filename, ext=0) for filename in filenames]
  im_size = min([min(i.shape) for i in im])
  im = np.stack([i[:im_size, :im_size] for i in im], axis=-1).astype('float32')
  return im

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

class Tng100(tfds.core.GeneratorBasedBuilder):
  """TNG100 galaxy dataset"""  

  VERSION = tfds.core.Version("2.0.0")
  RELEASE_NOTES = {'2.0.0': 'Change flux units. wl is sort.',}
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
            'noiseless_griz': tfds.features.Tensor(shape=(128, 128, 4), dtype=tf.float32),
            #'stellar_light': tfds.features.Tensor(shape=(512, 512), dtype=tf.float32),
            #'velocity_map': tfds.features.Tensor(shape=(512, 512), dtype=tf.float32),
            #'velocity_dispersion_map': tfds.features.Tensor(shape=(512, 512), dtype=tf.float32),
            "sed": tfds.features.Tensor(shape=(143,), dtype=tf.float32),
            "time": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            "SFR_halfRad": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            "SFR_Rad": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            "SFR_Max": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            "Mstar_Half": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            "Mstar": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            'mass_quantiles': tfds.features.Tensor(shape=(9,), dtype=tf.float32),
            'wl_sort': tfds.features.Tensor(shape=(143,), dtype=tf.float32),
            'last_over_max': tf.float32,
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

    # Create new dataframe with the columns 'Illustris_ID' and 'SnapNumLastMajorMerger'
    # of the TNG100_SDSS_MajorMergers.csv file
    #mergers_data = pd.read_csv(root_path+'/mergers/TNG100_SDSS_MajorMergers.csv', 
     #                          usecols=['Illustris_ID','SnapNumLastMajorMerger'],
      #                         index_col='Illustris_ID')

    # Create new dataframe with equivalence values between Snapshots numbers and redshifts, age, loopback time
    snaps = pd.read_csv(root_path+"/snaps.csv", 
                        index_col=0, names=['sn', 'z', 'age', 'lbt'])
    a = 1./(1. + np.array(snaps['z'][::-1]))

    # Opening sed data
    phot_cat = Table.read(root_path+"/phot_TNG100_dylan_143.csv")
    phot_cat['subhaloIDs'] = phot_cat['subhaloIDs'].astype('int') # convert to int

    #sorting wavelength
    wl = np.loadtxt(root_path+"/wl.csv")
    inds = np.argsort(wl)
    wl_sort = wl[inds]

    for filename in glob.glob(root_path+"/cats_SFH/*.csv"):
      object_id = int((filename.split("_")[-1].split('.')[0])) # Extracting object id
      

      try:
        # Opening multiband TNG image 
        img = stack_bands(root_path+'/images/TNG100/sdss/sn99/noiseless/', object_id)
        example = {'noiseless_griz': img.astype('float32')}

        # Opening kinematic data
        #kin_image = fits.getdata(root_path+'/mergers/maps/sn99/moments_TNG100-1_99_%d_stars_i0__32.fits'%object_id, ext=0)
        #example.update({'stellar_light': kin_image[0].astype('float32'),
         #               'velocity_map': kin_image[1].astype('float32'),
         #               'velocity_dispersion_map': kin_image[2].astype('float32')})

        # Retrieve sed row for given galaxy
        row = phot_cat[phot_cat['subhaloIDs'] == object_id][0]
        mag = np.array(list(row.values())[1:]).astype('float32')
        app_mag = np.array(mag)+5*(np.log10(20e6)-1) #assume at 20pc
        flux = 10**(.4*(-app_mag+8.90))
        flux = flux[inds]
        example.update({'sed': np.array(flux).astype('float32')})
        example.update({'wl_sort': np.array(wl_sort).astype('float32')})
        
        # Opening sfh data and interpolating it on common grid, in reverse time
        sfh = Table.read(filename)
        example.update({k: interp1d(sfh['time'], sfh[k], bounds_error=False, fill_value=0.)(a).astype('float32') 
                        for k in ['SFR_halfRad', 'SFR_Rad', 'SFR_Max', 'Mstar_Half', 'Mstar']})
        example.update({'time': np.array(a).astype('float32')})

        # Get snapshot number of the last major merger for the current object_id from the mergers_data dataframe
        #SnapNumLastMajorMerger = mergers_data.loc[object_id,'SnapNumLastMajorMerger']
        # Convert snapshot number to lookback time using the snaps dataframe
        #example.update({'last_major_merger': 1./(1.+snaps.loc[SnapNumLastMajorMerger,'z']).astype('float32')})

        # Compute mass history summaries
        mass_history_summaries = find_summaries(example['Mstar_Half'],
                                                example['time'])
        last_over_max = example['Mstar_Half'][0]/np.max(example['Mstar_Half'])
        example.update({'mass_quantiles': mass_history_summaries,
                        'last_over_max': last_over_max,
                        'object_id': object_id})

        yield object_id, example
      except:
        continue      
