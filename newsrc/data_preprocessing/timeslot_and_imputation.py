import pandas as pd
import os
from datetime import timedelta
from joblib import Parallel, delayed
import time
from tqdm import tqdm
import numpy as np 
import math
import copy

from datetime import timedelta
from utils import strptime
from constants import DTYPES

SORTKEY = 'EVENTTIME'
DICTKEY = 'ICUSTAY_ID'

def hashing_with_sorting(records):
    # 1. hasing and sorting
    sortidx = list(records.head()).index(SORTKEY)
    records_dict = {}
    for i, record in records.iterrows():
        r_key = record[DICTKEY]
        if r_key not in records_dict:
            records_dict[r_key] = [list(record)]
        else:
            # sorting: most of items are already sorted
            current_time = strptime(record[SORTKEY])
            for idx, _record in enumerate(reversed(records_dict[r_key])):
                last_time = strptime(_record[sortidx])
                if current_time - last_time > timedelta(0):
                    break
            records_dict[r_key].insert(len(records_dict[r_key]) - idx, list(record))

    return records_dict

def merge_and_sort(record_a, record_b, sort_idx):
    def get_sortkey(elem):
        return elem[sort_idx]
    record_a.extend(record_b)
    return sorted(record_a, key=get_sortkey)

def averaging(bucket):
    if len(bucket) == 0:
        return None
    
    num_features = len(bucket[0])
    output = np.zeros(num_features)
    for n in range(num_features):
        fs = [f[n] for f in bucket if not math.isnan(f[n])]
        if len(fs) > 0:
            output[n] = np.mean(fs)
        else:
            output[n] = float('nan')
    return output

def time_sloting(records, header):
    refined = []
    tidx = header.index(SORTKEY)
    icu_id = records[0][header.index('ICUSTAY_ID')]
    soretime = records[0][header.index('SORETIME')]
    if not isinstance(soretime, str):
        if math.isnan(soretime):
            soretime = 'N/A'
    new_header = ['ICUSTAY_ID', 'SORETIME'] + header[tidx:]

    prev_time = strptime(records[0][tidx])
    while (len(records) > 0):
        bucket = []
        while (len(records) > 0):
            tdiff = (strptime(records[0][tidx]) - prev_time).total_seconds() < 60*60
            if tdiff:
                assert tdiff >= 0
                bucket.append(records.pop(0)[tidx+1:])
            else:
                break

        bucket = averaging(bucket)
        post_heads = [icu_id, soretime, prev_time.strftime('%Y-%m-%d %H:%M')] 

        if bucket is not None:
            refined.append(post_heads + list(bucket))
        else:
            refined.append(post_heads + [float('nan') for _ in range(len(header[tidx+1:]))])
        prev_time = prev_time + timedelta(hours=1)
            
    return refined, new_header

def get_stats(fname):
    df = pd.read_csv(fname)
    stats = {}
    for idx, row in df.iterrows():
        stats[row['item']] = row['avg']
    return stats

def imputation(records, header, stats):
    VHSI = 3 # value header starting idx
    imputed = []
    value_header = header[VHSI:]

    # set initial value
    prev_values = copy.copy(records[0][VHSI:])
    for idx, h in enumerate(value_header):
        if math.isnan(prev_values[idx]):
            prev_values[idx] = stats[h]
    imputed.append(records[0][:VHSI] + copy.copy(prev_values))

    for record in records[1:]:
        # ignore icustay_id / timestamp
        irecord = record[VHSI:]
        for idx, h in enumerate(value_header):
            if math.isnan(irecord[idx]):
                irecord[idx] = prev_values[idx]
        
        prev_values = irecord
        imputed.append(record[:VHSI] + copy.copy(irecord))

    return imputed


def multiproc_read_records(fname, sortkey, n_jobs=25, chunksize=100000, nrows=None, debug=True):
    # multiproc divides data and merge, so sorting is needed
    # sortkey: EVENTTIME
    if debug:
        chunksize = 1000
        nrows = 100000

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs)(delayed(hashing_with_sorting)(
        records=df_chunk) for df_chunk in tqdm(pd.read_csv(fname, dtype=DTYPES,
            chunksize=chunksize, nrows=nrows)))

    small_records = pd.read_csv(fname, nrows=10)
    sortidx = list(small_records.head()).index(sortkey)

    merged = {}
    for result in results:
        for key in result.keys():
            if key in merged:
                merged[key] = merge_and_sort(merged[key], result[key], sortidx)
            else:
                merged[key] = result[key]
    print ('loadded in {:.3f} seconds'.format(time.time() - t0))
    return merged, list(small_records.head())
    

if __name__ == '__main__':
    input_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/01_outlier.csv'
    output_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/02_timeslot.csv'
    stats_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/stats_refined.csv'
    stats = get_stats(stats_fname)
    
#    results = Parallel(n_jobs=25)(delayed(hashing_with_sorting)(
#        records=df_chunk) for df_chunk in tqdm(pd.read_csv(input_fname, dtype=DTYPES, chunksize=1000,
#            nrows=100000)))
#    
#
#    small_records = pd.read_csv(input_fname, nrows=10)
#    sortidx = list(small_records.head()).index(SORTKEY)
#
#    merged = {}
#    for result in results:
#        for key in result.keys():
#            if key in merged:
#                merged[key] = merge_and_sort(merged[key], result[key], sortidx)
#            else:
#                merged[key] = result[key]
    merged, header = multiproc_read_records(input_fname, 'EVENTTIME', debug=True)
    
    with open(output_fname, 'wt') as f:
        for i, (key, records) in enumerate(merged.items()):
            if i % 100 == 0:
                print ('{}  | {} / {}'.format(key, i, len(merged)))
            trecords, new_header = time_sloting(records, header)
            if i == 0:
                f.writelines(','.join(new_header) + '\n')
            irecords = imputation(trecords, new_header, stats)
            #output.extend(irecords)
            for ir in irecords:
                f.writelines(','.join(str(r) for r in ir) + '\n')

#    data = pd.DataFrame(output)
#    data.to_csv(output_fname, header=new_header)

    

