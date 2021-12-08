import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from sfhsed import utils

''' Dataset builder '''
# TODO(my_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
"""

# TODO(my_dataset): BibTeX citation
_CITATION = """
"""


class MyDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""

    VERSION = tfds.core.Version('1.0.4')
    RELEASE_NOTES = {
        '1.0.4': '9 quantiles',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'flux': tfds.features.Tensor(shape=(143,), dtype=tf.float32),
                'mass': tfds.features.Tensor(shape=(100,), dtype=tf.float32),
                'time': tfds.features.Tensor(shape=(100,), dtype=tf.float32),
                'quantile': tfds.features.Tensor(shape=(9,), dtype=tf.float32),
                'object_id': tf.float32,
                # These are the features of your dataset like images, labels ...
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('flux', 'quantile'),  # e.g. ('image', 'label')
            # homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(my_dataset): Downloads the data and defines the splits
        # dl_manager is a tfds.download.DownloadManager that can be used to
        # download and extract URLs
        tng_dir_path = "/gpfswork/rech/qrc/commun/SFH/tng100"
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={'tng_dir_path': tng_dir_path},
            ),
        ]

    def _generate_examples(self, tng_dir_path):
        """Yields examples."""

        # Create new dataframe with the columns 'Illustris_ID' and 'SnapNumLastMajorMerger'
        # of the TNG100_SDSS_MajorMergers.csv file
        times, data, wl = utils.create_data_array(path=tng_dir_path)
        table = utils.SubHalos(data, wl, times)

        # recover fluxes
        for i in range(len(table)):
            subhalo = table[i]
            flux = subhalo.fluxes
            quantile = subhalo.quantiles
            object_id = subhalo.shid
            mass = subhalo.mstar
        time = utils.subhalo.times
        # Yiel with i because in our case object_id will be the same for the 4 different projections
        yield i, {'flux': flux.astype("float32"),
                  'mass': mass.astype("float32"),
                  'time': time.astype('float32'),
                  'quantile': quantile.astype('float32'),
                  'object_id': object_id}
