import pandas as pd
import os
from datetime import timedelta
import math

from timeslot_and_imputation import multiproc_read_records
from utils import strptime


def remove_after_sore(records, header):
    
    soretime = records[0][header.index('SORETIME')]
    if not isinstance(soretime, str):
        if math.isnan(soretime):
            return records
        else:
            raise ValueError
            
    # case group
    new_records = []
    for record in records:
        td = (strptime(soretime) - strptime(record[header.index('EVENTTIME')])).total_seconds()
        if td > 60*60*24:
            new_records.append(record)
        else:
            break

    return new_records



if __name__ == '__main__':
    input_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/01_outlier.csv'
    output_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/02_remove_after_event.csv'

    records, header = multiproc_read_records(input_fname, 'EVENTTIME', debug=False)
    
    with open(output_fname, 'wt') as f:
        f.writelines(','.join(header) + '\n')
        for idx, (key, record) in enumerate(records.items()):
            if idx % 100 == 0:
                print ('{} | {}/{}'.format(key, idx, len(records)))

            new_records = remove_after_sore(record, header)
            if len(new_records) == 0:
                print ('{} is excluded!'.format(key))
            for nr in new_records:
                f.writelines(','.join(str(r) for r in nr) + '\n')

