import utils
import numpy as np


if __name__ == '__main__':
    import argparse
    TNG100_PATH = "/gpfswork/rech/qrc/commun/SFH/tng100"
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
    times, data = utils.create_data_array(path=args.path, limit=args.limit)
    wl = utils.read_wavelength(path=args.path)
    # Let's sort the wl, fluxes
    idx_sort = np.argsort(wl)
    wl = wl[idx_sort]
    print(f"Writing to: {args.output}")
    np.savez(args.output, data=data, times=times, wl=wl)
