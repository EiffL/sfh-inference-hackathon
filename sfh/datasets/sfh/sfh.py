"""sfh dataset."""

import tensorflow_datasets as tfds
import pandas as pd
# TODO(sfh): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
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

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(sfh): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "time": tfds.features.Tensor(shape=(N_TIMESTEPS, 1)),
                    "SFR_halfRad": tfds.features.Tensor(shape=(N_TIMESTEPS, 1)),
                    "SFR_Rad": tfds.features.Tensor(shape=(N_TIMESTEPS, 1)),
                    "SFR_Max": tfds.features.Tensor(shape=(N_TIMESTEPS, 1)),
                    "Mstar_Half": tfds.features.Tensor(shape=(N_TIMESTEPS, 1)),
                    "Mstar": tfds.features.Tensor(shape=(N_TIMESTEPS, 1)),


                    #"label": tfds.features.ClassLabel(names=["no", "yes"]),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=(None),  # Set to `None` to disable
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def csv_to_np(self, path):

        keys = ["time", "SFR_halfRad", "SFR_Rad", "SFR_Max", "Mstar_Half", "Mstar"]
 
        data = {}
        df = pd.read_csv(path)
        timesteps = df[['SnapNum']].values
        
        for k in keys:
            d = np.zeros((N_TIMESTEPS, 1))
            d[timesteps, 0] = df[[k]].values
            data[k] = d

        return data


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
        # TODO(sfh): Yields (key, example) tuples from the dataset
        for f in path.glob("*.csv"):
            yield "key", csv_to_np(path)
