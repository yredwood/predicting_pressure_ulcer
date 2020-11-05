import pandas as pd
import os
from datetime import timedelta
from joblib import Parallel, delayed
import time
from tqdm import tqdm
import numpy as np 
import math

from datetime import timedelta
from utils import strptime

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
    new_header = ['ICUSTAY_ID'] + header[tidx:]

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
        post_heads = [icu_id, prev_time.strftime('%Y-%m-%d %H:%M')] 
        if bucket is not None:
            refined.append(post_heads + list(bucket))
        else:
            refined.append(post_heads + [float('nan') for _ in range(len(header[tidx+1:]))])
        prev_time = prev_time + timedelta(hours=1)
            
    return refined, new_header



if __name__ == '__main__':
    input_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/01_outlier.csv'
    output_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/02_timeslot.csv'
    s0 = time.time()
    
    results = Parallel(n_jobs=25)(delayed(hashing_with_sorting)(
        records=df_chunk) for df_chunk in tqdm(pd.read_csv(input_fname, chunksize=1000, nrows=100000)))
    
    small_records = pd.read_csv(input_fname, nrows=10)
    sortidx = list(small_records.head()).index(SORTKEY)

    merged = {}
    for result in results:
        for key in result.keys():
            if key in merged:
                merged[key] = merge_and_sort(merged[key], result[key], sortidx)
            else:
                merged[key] = result[key]

    print ('loading time: ', time.time() - s0)
    
    time_sliced = []
    for key, records in merged.items():
        trecords, new_header = time_sloting(records, list(small_records.head()))
        time_sliced.extend(trecords)

    data = pd.DataFrame(time_sliced)
    data.to_csv(output_fname, header=new_header)

    

