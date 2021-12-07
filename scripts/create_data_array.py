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

def create_data_array(phot_cat, limit=None):
    """Create the array of data using the given photometry catalog.
    Optionally limit the number of rows to the given limit.
    """
    if limit is None:
        limit = len(phot_cat)
    # 0: SubHaloID
    # 1-144: fluxes
    # 144-244: SFR
    # 244-344: Mstar
    # 344-353: slots for summaries data (to be generated later)
    dims = (limit, 1 + 143 + 2*100 + 9)
    arr = np.zeros(dims)
    # I know there are 100 snapshots maximum
    all_sfh_times = [set() for i in range(100)]
    for index, (shid, fluxes) in enumerate(phot_cat.iterrows()):
        if index >= limit:
            break
        arr[index, 0] = shid
        arr[index, 1:144] = fluxes.array
        try:
            sfh = read_tng_file(shid)
        except:
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
    times = np.asarray([s.pop() for s in all_sfh_times])
    return times, arr

def read_wavelength(path=None):
    if path is None:
        path = TNG100_PATH
    wl_filename = os.path.join(path, 'wl.csv')
    wl = np.loadtxt(wl_filename, delimiter=',')
    return wl


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
            description='Create data array from TNG100 files')
    parser.add_argument('--path', type=str,
            help='path to TNG100 data', default=TNG100_PATH)
    parser.add_argument('--limit', type=int,
            help='limit to the first *limit* rows')
    parser.add_argument('--output', type=str,
            help='output file', default='data.npz')
    args = parser.parse_args()

    print(f"Path: {args.path}")
    print(f"Limit: {args.limit}")
    phot_cat = read_phot_cat(path=args.path)
    times, data = create_data_array(phot_cat, limit=args.limit)
    wl = read_wavelength(path=args.path)
    # Let's sort the wl, fluxes
    idx_sort = np.argsort(wl)
    data[:, 1:144] = data[:, 1:144][:, idx_sort]
    wl = wl[idx_sort]
    print(f"Writing to: {args.output}")
    np.savez(args.output, data=data, times=times, wl=wl)
