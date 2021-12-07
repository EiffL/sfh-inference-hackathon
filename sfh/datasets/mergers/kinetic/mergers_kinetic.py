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

# TODO(Mergers_kinetic): BibTeX citation
_CITATION = """
"""


class MergersKinetic(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Mergers_kinetic dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  # TODO(Mergers_kinetic): MANUAL_DOWNLOAD_INSTRUCTIONS have to be adapted for Jean-Zay 
  MANUAL_DOWNLOAD_INSTRUCTIONS = """TBD"""
  
  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        # Two features: image with 3 channels (stellar light, velocity map, velocity dispersion map)
        #  and redshift value of last major merger
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Tensor(shape=(3, 512, 512), dtype=tf.float32),
            'last_major_merger': tf.float32,
        }),
        supervised_keys=('image', 'last_major_merger'), 
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    data_path = dl_manager.manual_dir 

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # Send paths to fits files and the TNG100_SDSS_MajorMergers.csv to _generate_examples method
            gen_kwargs={"fits_dir_path": os.path.join(data_path, "/mergers/maps/sn99/"),
                        "majormergers_path": os.path.join(data_path, "/mergers/")
            },
        ),
    ]

  def _generate_examples(self, fits_dir_path, majormergers_path):
    """Yields examples."""
    from astropy.io import fits # To open FITS files
    from astropy.cosmology import Planck13 # To convert redshift to loopback time
    import pandas as pd # To extract the SnapNumLastMajorMerger values from TNG100_SDSS_MajorMergers.csv

    # Create new dataframe with the columns 'Illustris_ID' and 'SnapNumLastMajorMerger'
    # of the TNG100_SDSS_MajorMergers.csv file
    mergers_data =  pd.read_csv(majormergers_path+"TNG100_SDSS_MajorMergers.csv",
      usecols=['Illustris_ID','SnapNumLastMajorMerger'],
      index_col='Illustris_ID'
    )
    # Create new dataframe with equivalence values between Snapshots numbers and redshifts, age, loopback time
    snaps = pd.read_csv("snaps.csv", index_col=0, names=['sn', 'z', 'age' 'lbt'])
    
    for i, fits_file in enumerate(os.listdir(fits_dir_path)):
      # Get object_id from the current FITS file name
      object_id = int(fits_file.split('_')[3])
      # Extract image data from the FITS file
      image = fits.getdata(fits_dir_path+fits_file, ext=0)
      # Get snapshot number of the last major merger for the current object_id from the mergers_data dataframe
      napNumLastMajorMerger = mergers_data.loc[object_id,'SnapNumLastMajorMerger']
      # Convert snapshot number to lookback time using the snaps dataframe
      last_major_merger = snaps.loc[napNumLastMajorMerger,'lbt']
      # Yiel with i because in our case object_id will be the same for the 4 different projections
      yield i, {'image': image.astype("float32"), 'last_major_merger': last_major_merger}

