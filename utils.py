import statsmodels.api as sm
import numpy as np


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

    return summary_times, summary_masses, summary_indices


def plot_with_summaries(axs, index, flux, wl, sfh):
    '''
    Plot the galaxy infos (SED, SFR, Mass) with the summaries
    input:
           axs: the axes of the plt.subplots with
           index: the ith index of the galaxies you want to plot
           flux: the flux of the galaxy
           wl: the wavelenghthes (x axis)
           sfh: the star fornation history
    output: nothing, but plot the figure
    '''

    axs[index, 0].scatter(np.array(wl)[np.array(wl) < 10**3], np.log10(flux), s=10)
    axs[index, 0].set_xlabel("wavelength [$\mu m$]")
    axs[index, 0].set_xscale('log')
    axs[index, 0].set_ylabel("$\log(f)$ [Jy]")
    axs[index, 1].set_xlabel("Time")
    axs[index, 1].set_ylabel("SFR")
    axs[index, 1].plot(sfh.time, sfh.SFR_halfRad)
    axs[index, 2].plot(sfh.time, np.log10(sfh.Mstar_Half)+10)
    half_times, half_masses, half_indices = find_summaries(sfh.Mstar_Half, sfh.time)
    for i in range(len(half_masses)):
        axs[index, 2].vlines(sfh.time[half_indices[i]], 6.5, np.max(np.log10(sfh.Mstar_Half)+10)+0.5)
        axs[index, 2].text(sfh.time[half_indices[i]], 6.3, (i+1)/10, rotation='vertical')
    axs[index, 2].set_xlabel("Time")
    axs[index, 2].set_ylabel("Mstar")


def smoothen(x, y):
    # smoothen data with frac = 0.2 to adjuct if needed
    lowess = sm.nonparametric.lowess(y, x, frac=0.2)
    return lowess


# here begins my_dataset.py
"""my_dataset dataset."""
import numpy as np
from astropy.table import Table
import tensorflow_datasets as tfds

# TODO(my_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
"""

# TODO(my_dataset): BibTeX citation
_CITATION = """
"""


class MyDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          'flux':tfds.features.Tensor(shape=(1, 143 ), dtype=tf.float32),
          'quantile':tfds.features.Scalar(shape=(1), dtype=tf.float32),
            # These are the features of your dataset like images, labels ...
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # e.g. ('image', 'label')
        #homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(my_dataset): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    catalog_path = 'path'
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={'catalog_path': catalog_path},
        ),
    ]

def _generate_examples(self, fits_dir_path):
    """Yields examples."""
    from astropy.io import fits # To open FITS files
    from astropy.cosmology import Planck13 # To convert redshift to loopback time
    import pandas as pd # To extract the SnapNumLastMajorMerger values from TNG100_SDSS_MajorMergers.csv

    # Create new dataframe with the columns 'Illustris_ID' and 'SnapNumLastMajorMerger'
    # of the TNG100_SDSS_MajorMergers.csv file
    flux_mass_data =  Table.read(fits_dir_path)
    # recover fluxes
    for t in flux_mass_data:
      flux = t[1:144]
      quantile = t['quantile']
      object_id = t['halo_id']
    # Yiel with i because in our case object_id will be the same for the 4 different projections
        yield t, {'flux': flux.astype("float32"), 'quantile': quantile.astype('float32'), 'object_id': object_id}