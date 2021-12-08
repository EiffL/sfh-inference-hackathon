#import statsmodels.api as sm
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

print("Importing utils.py...")

class SubHalos:
    def __init__(self, data, wl, times):
        self._data = data
        self._wavelengths = wl
        self._times = times

    @classmethod
    def fromfile(cls, filename):
        raw = np.load(filename)
        data = raw['data']
        wl = raw['wl']
        times = raw['times']
        return cls(data, wl, times)

    def __getitem__(self, i):
        return SubHalo(self._data[i], wl=self._wavelengths, times=self._times)

    def __len__(self):
        return len(self._data)
    
    @property
    def shf_times(self):
        return self._times

    @property
    def wl(self):
        return self._wavelengths


class SubHalo:
    def __init__(self, row, wl=None, times=None):
        self._shid = row[0]
        self._mags = row[1:144]
        self._mstar = row[244:344]
        self._sfr = row[144:244]
        self._quantiles = row[344:353]
        self._wl = wl
        self._times = times

    @property
    def shid(self):
        return self._shid

    @property
    def mags(self):
        return self._mags

    @property
    def fluxes(self):
        # compute apparent magnitude, assuming 20pc
        app_mags = self.mags + 5.0 * (np.log10(20e6) - 1.0)
        # convert to Jy
        fluxes = 10**(0.4 * (-app_mags + 8.90)) 
        return fluxes

    @property
    def mstar(self):
        return self._mstar

    @property
    def sfr(self):
        return self._sfr

    @property
    def wl(self):
        return self._wl

    @property
    def times(self):
        return self._times

    @property
    def quantiles(self):
        return self._quantiles


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

"""Create a pickle containing:
    - wavelength array (143 elements) corresponding to SED fluxes
    - time and reshift arrays (100 elements each) with snapshot times and 
      redshifts
    - a 2D array with one row per galaxy, each row containing:
      shid, 143xFluxes, 100xM_star_Half, 100xSFR_HakfRad
"""

import os
import pandas as pd
import numpy as np

TNG100_PATH="/gpfswork/rech/qrc/commun/SFH/tng100"


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


def read_tng_file(shid: int):
    """Read TNG100 SFH file for the given sub-halo *shid* 
    """
    shid = int(shid)
    filename = os.path.join(
            TNG100_PATH,
            'cats_SFH',
            f'TNG100_mainprojenitors_{shid}.csv'
        )
    sfh = pd.read_csv(filename)
    return sfh


def read_phot_cat(path=None):
    """Read the photometry catalog
    """
    if path is None:
        path = TNG100_PATH
    filename = os.path.join(TNG100_PATH, 'phot_TNG100_dylan_143.csv')
    phot_cat = pd.read_csv(filename)
    phot_cat.set_index("subhaloIDs", inplace=True)
    return phot_cat


def interpolate_sfh(sfh):
    """Linearly interpolate missing lines in SFH data.

    Algo: loop over rows, if a row is missing (SnapNUm make a jump > 1),
    add a new row equal to the mean between previous and current row.
    (NB: We don't try to add missing rows at the end.)
    """
    new_rows = []
    prev = None
    for i, row in sfh.iterrows():
        if prev is None:
            new_rows.append(row)
            prev = i, row
            continue
        prev_i, prev_row = prev
        gap = prev_row['SnapNUm'] - row['SnapNUm']
        if gap > 1:
            #print(gap)
            assert gap == 2
            # create an interpolated row
            new_row = 0.5 * (prev_row + row)
            #new_row['Unnamed: 0'] = i + shift
            new_rows.append(new_row)
        new_rows.append(row)
        prev = i, row
    new_shf = pd.DataFrame(new_rows, index=list(range(len(new_rows))))
    # We remove this comlum as we give the index explicitly above
    new_shf.drop(columns='Unnamed: 0', inplace=True)
    return new_shf


def create_data_array(path=None, limit=None):
    """Create the array of data using the given photometry catalog.
    Optionally limit the number of rows to the given limit.
    """
    phot_cat = read_phot_cat(path=path)

    if limit is None:
        limit = len(phot_cat)
    # 0: SubHaloID
    # 1-144: fluxes
    # 144-244: SFR
    # 244-344: Mstar
    # 344-353: slots for summaries data (to be generated later)
    # 353: last/max flag
    # 354: missing sfh flag

    dims = (limit, 1 + 143 + 2*100 + 9 + 2)
    arr = np.zeros(dims)
    # I know there are 100 snapshots maximum
    all_sfh_times = [set() for i in range(100)]
    for index, (shid, fluxes) in enumerate(phot_cat.iterrows()):
        if index % 1000 == 0:
            print(f"Processing {index}/{limit}")
        if index >= limit:
            break
        arr[index, 0] = shid
        arr[index, 1:144] = fluxes.array
        try:
            sfh = read_tng_file(shid)
        except:
            arr[index, 354] = 1
            continue
        # want to get the time of SFH (common to all SFH)
        snap_num = np.asarray(sfh['SnapNUm'], dtype=int)
        times = np.asarray(sfh['time'], dtype=float)
        for i, t in zip(snap_num, times):
            all_sfh_times[i].add(t)
        # interpolate sfh
        sfh = interpolate_sfh(sfh)
        # populate the data array
        sfr_half_rad = sfh['SFR_halfRad'].array
        n = sfr_half_rad.shape[0]
        arr[index, 144:144+n] = sfr_half_rad
        sfr_mstar_half = sfh['Mstar_Half'].array
        n = sfr_mstar_half.shape[0]
        arr[index, 244:244+n] = sfr_mstar_half
        summaries, _, _ = find_summaries(sfr_mstar_half, sfh['time'])
        arr[index, 344:353] = summaries
        arr[index, 353] = sfr_mstar_half[0] / max(sfr_mstar_half)

    times = np.asarray([s.pop() for s in all_sfh_times])[::-1]
    wl = read_wavelength(path=path)
    idx_sort = np.argsort(wl)
    arr[:, 1:144] = arr[:, 1:144][:, idx_sort]
    wl = wl[idx_sort]
    return times, arr, wl


def read_wavelength(path=None):
    if path is None:
        path = TNG100_PATH
    wl_filename = os.path.join(path, 'wl.csv')
    wl = np.loadtxt(wl_filename, delimiter=',')
    return wl
