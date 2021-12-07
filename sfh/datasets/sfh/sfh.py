"""sfh dataset."""

import os

from astropy.table import Table, vstack

import tensorflow as tf
import tensorflow_datasets as tfds

# TODO(sfh): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
# SFH Dataset

Dataset for generative models. Data is extracted from csv files of TNG100 snapshots
For each galaxy, the following sequence are stored into the dataset:
 - time
 - SFR_halfRad
 - SFR_Rad
 - SFR_Max
 - Mstar_Half
 - Mstar

Plus : N_age, just an int to indicate to the model how many of the timesteps are relevants
"""

# TODO(sfh): BibTeX citation
_CITATION = """
"""

N_TIMESTEPS = 100

class Sfh(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for sfh dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = "TBD"

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
                    "N_age": tfds.features.Tensor(
                        shape=(1,), dtype=tf.dtypes.int32),
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

    def _generate_examples(self, path):
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

            sfh = Table.read(filename)
            n_age = len(sfh)

            # Add the missing SnapNums
            sfh_min_snapnum = sfh['SnapNUm'].min()
            sfh = vstack([
                sfh,
                empty_sfh[empty_sfh['SnapNUm'] < sfh_min_snapnum]],
            )

            yield object_id, {
                "time": sfh['time'].value,
                "SFR_halfRad": sfh['SFR_halfRad'].value,
                "SFR_Rad": sfh['SFR_Rad'].value,
                "SFR_Max": sfh['SFR_Max'].value,
                "Mstar_Half": sfh['Mstar_Half'].value,
                "Mstar": sfh['Mstar'].value,
                "N_age": [n_age]
            }
