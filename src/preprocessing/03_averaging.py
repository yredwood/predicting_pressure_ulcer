import csv
import os
import numpy as np
import copy
from utils import read_csv, write_csv, hashing
from operator import itemgetter
from datetime import datetime, timedelta

INFO_HEADER = [
    'case/control', 'ICUSTAY_ID', 'START_TIME', 'END_TIME',
    'TIMESTAMP', 'TIME_from_START', 'TIME_to_END'
]

VITAL_SIGNS_HEADER = [
    'BRANDEN_SCORE', 'GCS', 'HR', 'RR', 'TEMPERATURE',
    'SBP', 'DBP', 'MBP', 'SaO2', 'SpO2'
]

LAB_HEADER = [
    'Lactate', 'Oxygen Saturation', 'pCO2', 'pH', 'pO2',
    'Albumin', 'Bicarbonate', 'Total Bilirubin', 'Creatinine',
    'Glucose', 'Potassium', 'Sodium', 'Troponin I', 'Troponin T',
    'Urea Nitrogen', 'Hematocrit', 'Hemoglobin', 'INR(PT)',
    'Neutrophils', 'Platelet Count', 'White Blood Cells'
]

dir_path = './datasets'
input_fname = os.path.join(dir_path, '02_imputation.csv')
output_fname = os.path.join(dir_path, '03_averaging.csv')

strptime = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

def _sorting(recordss, sort_key, reverse=True):
    for key, records in recordss.items():
        records = sorted(records, key=itemgetter(sort_key))
        recordss[key] = records

def _excluding(recordss, thr_len=6):
    subsetss = {}
    for key, records in recordss.items():
        if len(records) < thr_len:
            continue
        subsetss[key] = records
    return subsetss

def _mean_feature(records, header):
    features = {}
    for record in records:
        for name in header:
            features[name] = features.get(name, list())
            '''
            value = int(record[name]) if name in ['BRANDEN_SCORE', 'GCS'] \
                    else float(record[name])
            '''
            value = float(record[name])
            features[name].append(value)

    for name in header:
        features[name] = np.mean(features[name])
    return features

def _averaging(records):

    refined = []
    start_time = strptime(records[0]['TIMESTAMP'])
    end_time = strptime(records[-1]['TIMESTAMP'])

    info_feature = {}
    for name in INFO_HEADER:
        info_feature[name] = records[0][name]

    bucket = []

    i = 0
    prev_j = 0
    while(True):
        start_time = start_time if i == 0 else start_time + timedelta(hours=1)
        if start_time > end_time:
            break

        bucket = []
        for j in range(prev_j, len(records)):
            record = records[j]
            time = strptime(record['TIMESTAMP'])
            time_diff = (time-start_time).total_seconds()
            if time >= start_time and time_diff <= 60*60:
                bucket.append(record)
            else:
                prev_j = j
                break

        if len(bucket) > 0:
            mean_feature = _mean_feature(bucket, VITAL_SIGNS_HEADER + LAB_HEADER)
            info_feature['TIMESTAMP'] = start_time
            refined_record = {**info_feature, **mean_feature}
            refined.append(refined_record)
        else:
            info_feature['TIMESTAMP'] = start_time
            refined_record = {**info_feature, **mean_feature}
            refined.append(refined_record)

        i += 1

    return refined

def main():
    records = read_csv(input_fname)
    recordss = hashing(records, 'ICUSTAY_ID')
    print('#icu_stays: %d'%len(recordss))

    recordss = _excluding(recordss)
    print('#icu_stays: %d (excluding stays having <= %d records)'\
          %(len(recordss), 6))

    _sorting(recordss, 'TIMESTAMP')

    recordss_avg = {}
    #for key, records in recordss.items():
    for i, key in enumerate(sorted(recordss.keys())):
        records = recordss[key]
        records_avg = _averaging(records)
        recordss_avg[key] = records_avg
        print('%d/%d'%(i, len(recordss.keys())))

    recordss_csv = []
    for key in sorted(recordss_avg.keys()):
        records = recordss_avg[key]
        recordss_csv.extend(records)

    write_csv(output_fname, INFO_HEADER + VITAL_SIGNS_HEADER + \
              LAB_HEADER, recordss_csv)

if __name__ == '__main__':
    main()
