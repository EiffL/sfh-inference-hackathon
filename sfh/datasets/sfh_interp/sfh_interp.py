"""sfh_interp dataset."""

import os

import tensorflow_datasets as tfds
from astropy.table import Table, vstack

import tensorflow as tf
import numpy as np
# TODO(sfh_interp): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
# SFH_INTERP Dataset

Dataset for generative models. Data is extracted from csv files of TNG100 snapshots
For each galaxy, the following sequence are stored into the dataset:
 - time
 - SFR_halfRad
 - SFR_Rad
 - SFR_Max
 - Mstar_Half
 - Mstar
 - Mask, 1 if value is original from raw data, 2 if interpolated, 0 if expanded at high z

"""

# TODO(sfh_interp): BibTeX citation
_CITATION = """
"""

N_TIMESTEPS = 100

class SfhInterp(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for sfh_interp dataset."""
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(sfh): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "time": tfds.features.Tensor(
                        shape=(N_TIMESTEPS,), dtype=tf.dtypes.float64),
                    "SFR_halfRad": tfds.features.Tensor(
                        shape=(N_TIMESTEPS,), dtype=tf.dtypes.float64),
                    "SFR_Rad": tfds.features.Tensor(
                        shape=(N_TIMESTEPS,), dtype=tf.dtypes.float64),
                    "SFR_Max": tfds.features.Tensor(
                        shape=(N_TIMESTEPS,), dtype=tf.dtypes.float64),
                    "Mstar_Half": tfds.features.Tensor(
                        shape=(N_TIMESTEPS,), dtype=tf.dtypes.float64),
                    "Mstar": tfds.features.Tensor(
                        shape=(N_TIMESTEPS,), dtype=tf.dtypes.float64),
                    "Mask": tfds.features.Tensor(
                        shape=(N_TIMESTEPS,), dtype=tf.dtypes.int32),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=(None),  # Set to `None` to disable
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(sfh): Downloads the data and defines the splits
        path = dl_manager.extract(os.path.join(dl_manager.manual_dir, "cats_SFH"))#download_and_extract("https://todo-data-url")

        # TODO(sfh): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path),
        }

    @staticmethod
    def _generate_examples(path):
        """Yields examples."""
        # To complete the partial SFH (the ones not starting at the beginning
        # of the Universe) we need to have all the SnapNums with the associated
        # time.  We take those from a know SFH.
        empty_sfh = Table.read(path / "TNG100_mainprojenitors_102694.csv")
        empty_sfh['SFR_halfRad'] = 0.
        empty_sfh['SFR_Rad'] = 0.
        empty_sfh['SFR_Max'] = 0.
        empty_sfh['Mstar_Half'] = 0.
        empty_sfh['Mstar'] = 0

        for filename in path.glob("*.csv"):

            object_id = filename.stem.split("_")[-1]
            #print(filename)
            sfh = Table.read(filename)
            mask = np.zeros((N_TIMESTEPS,), dtype=np.int32)
            mask[99-sfh['SnapNUm']] = 1.
            last_val = np.argwhere(mask==1.)[-1][0]
            tmp_sfh = empty_sfh.copy() 

            for k in empty_sfh.colnames:
                tmp_sfh[k][99-sfh['SnapNUm']] = sfh[k]
                tmp_interp = (tmp_sfh[k][np.argwhere(mask[:last_val]==0.)+1]+tmp_sfh[k][np.argwhere(mask[:last_val]==0.)-1])/2
                tmp_sfh[k][np.argwhere(mask[:last_val]==0.)] = tmp_interp

            yield object_id, {
                "time": tmp_sfh['time'].value,
                "SFR_halfRad": tmp_sfh['SFR_halfRad'].value,
                "SFR_Rad": tmp_sfh['SFR_Rad'].value,
                "SFR_Max": tmp_sfh['SFR_Max'].value,
                "Mstar_Half": tmp_sfh['Mstar_Half'].value,
                "Mstar": tmp_sfh['Mstar'].value,
                "Mask": mask
            }

