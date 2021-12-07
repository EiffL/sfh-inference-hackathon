# JB Marquette - Hackathon AstroInfo 2021
# Checking once for all the times in cats_SFH csv files and saving updated ones

import os
from astropy.table import Table, setdiff
from glob import iglob
import sys

dir_work = sys.argv[1] # The local directory where the files are
suffix_file = sys.argv[2] # The suffix to add to outputs, e.g. 'JB.csv' to keep the file type

ref_table = 'TNG100_mainprojenitors_102694.csv' # The reference table containing all times (100)
ref_data = Table.read(os.path.join(dir_work, ref_table), format='csv') # Reading corresponding data

# Loop on the tables
tables = iglob(os.path.join(dir_work, '*.csv'))
for table in tables:
    if suffix_file in table: # Protecting from possible already modified tables
        continue
    data = Table.read(table, format='csv')
    if len(data) == 100: # Table with all 100 times are skipped
        continue
    print('Processing ', table) # OK, we got a candidate, let's see the time column differences
    diff_time = setdiff(ref_data, data, keys=['time']) # Yielding rows which are in reference file only (1st argument)

    # Loop on columns to be padded
    for diff_col in diff_time.colnames[3:]: # All float values from column 'redshift' are padded to 0.0
        diff_time[diff_col] = 0.0

    # Additional paddings on integer columns
    diff_time['col0'] = 99 - diff_time['SnapNUm'] # col0 is reverse of SnapNUm
    diff_time['SubfindID'] = 0 # The column SubfindID (integer) is padded to 0

    # Loop on rows to be added to the current file
    for diff_row in diff_time:
        data.add_row(diff_row)

    data.sort('SnapNUm', reverse=True) # Reverse sort on SnapNUm to fit the reference file structure
    data.write(table[:-3] + suffix_file, format='csv', overwrite=True) # Saving with a different suffix

print('Process completed successfully')
exit(0)
