# The target directory
data_dir = '/gpfsscratch/rech/qrc/commun/dataset_sfhse'

from sfh.datasets.sfhsed import sfhsed
dset = tfds.load('sfhsed', split='train', data_dir=data_dir)

# That's it! It will create dataset according to sfhsed.py module
