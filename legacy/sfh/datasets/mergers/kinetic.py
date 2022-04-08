"""Mergers_kinetic dataset."""

import os
import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
#Kinetic data of TNG100 galaxy mergers dataset

Data is extracted from:
 - CSV file TNG100_SDSS_MajorMergers.csv
 - FITS files mergers-maps-sn99-moments_TNG100-1_99_<object_id>_stars_<projection_id>__32.fits

The snaps.csv file contains equivalences between Snapshots numbers and redshifts, age and loopback time

"""

_CITATION = ""
_URL = "https://github.com/EiffL/sfh-inference-hackathon"



class MeregersKineticConfig(tfds.core.BuilderConfig):
  """BuilderConfig for MergersKinetic."""
  def __init__(self, *, dataset_size=1, **kwargs):
    """BuilderConfig for MergersKinetic
    Args:
      dataset_size: max number of element in the dataset.
      **kwargs: keyword arguments forwarded to super.
    """
    super(MeregersKineticConfig, self).__init__(
      description=("MergersKinetic dataset with max %d images" %(dataset_size)),
      version=tfds.core.Version("2.0.0"),
      release_notes={"2.0.0": "New split API (https://tensorflow.org/datasets/splits)"},
      **kwargs
    )

    self.dataset_size = dataset_size
    # Paths to the data
    #data_path="/Users/benjamin/SCRATCH/sfh/content/data"
    data_path=os.path.expandvars("$ALL_CCFRWORK/SFH/tng100/")
    self.fits_dir_path = data_path+"/mergers/maps/sn99/"
    self.majormergers_database = data_path+"/mergers/TNG100_SDSS_MajorMergers.csv"
    self.snaps_file = os.path.join(os.path.dirname(__file__), './')+"/snaps.csv"

class MergersKinetic(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Mergers_kinetic dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Initial release.',}
  MANUAL_DOWNLOAD_INSTRUCTIONS = "Nothing to download. Dataset was generated at first call."
  
  BUILDER_CONFIGS = [
    MeregersKineticConfig(name='full', dataset_size=1000000), # Arbitrary large value here.
    MeregersKineticConfig(name='medium', dataset_size=5000),
    MeregersKineticConfig(name='small', dataset_size=2500),
    MeregersKineticConfig(name='tiny', dataset_size=250),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        homepage=_URL,
        citation=_CITATION,
        # Two features: image with 3 channels (stellar light, velocity map, velocity dispersion map)
        #  and redshift value of last major merger
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Tensor(shape=(3, 512, 512), dtype=tf.float32),
            'last_major_merger': tf.float32,
            'object_id': tf.int32,
        }),
        supervised_keys=('image', 'last_major_merger'), 
    )

  def _split_generators(self, dl):
    """Returns generators according to split"""
    return {tfds.Split.TRAIN: self._generate_examples(self.builder_config.dataset_size)}
    
  def _generate_examples(self, dataset_size):
    """Yields examples."""
    from astropy.io import fits # To open FITS files
    from astropy.cosmology import Planck13 # To convert redshift to loopback time
    import pandas as pd # To extract the SnapNumLastMajorMerger values from TNG100_SDSS_MajorMergers.csv

    # Create new dataframe with the columns 'Illustris_ID' and 'SnapNumLastMajorMerger'
    # of the TNG100_SDSS_MajorMergers.csv file
    mergers_data =  pd.read_csv(self.builder_config.majormergers_database, usecols=['Illustris_ID','SnapNumLastMajorMerger'],index_col='Illustris_ID')
    # Create new dataframe with equivalence values between Snapshots numbers and redshifts, age, loopback time
    snaps = pd.read_csv(self.builder_config.snaps_file, index_col=0, names=['sn', 'z', 'age', 'lbt'])
    i = 0
    for fits_file in os.listdir(self.builder_config.fits_dir_path):
      if (i>dataset_size):
        break
      try:
        # Get object_id from the current FITS file name
        object_id = int(fits_file.split('_')[3])
        # Extract image data from the FITS file
        image = fits.getdata(self.builder_config.fits_dir_path+fits_file, ext=0)
        # Get snapshot number of the last major merger for the current object_id from the mergers_data dataframe
        SnapNumLastMajorMerger = mergers_data.loc[object_id,'SnapNumLastMajorMerger']
        # Convert snapshot number to lookback time using the snaps dataframe
        last_major_merger = snaps.loc[SnapNumLastMajorMerger,'lbt']
        # Yiel with i because in our case object_id will be the same for the 4 different projections
        i += 1
        yield i, {'image': image.astype("float32"), 'last_major_merger': last_major_merger, 'object_id': object_id}
      except:
        print("File ", fits_file, " not added to the dataset: object_id=", object_id, 
        " SnapNumLastMajorMerger=",SnapNumLastMajorMerger, " last_major_merger: ", last_major_merger)

