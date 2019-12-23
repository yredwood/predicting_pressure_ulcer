import csv
import os
import numpy as np
import datetime
import copy
from utils import read_csv, write_csv, hashing
from operator import itemgetter

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
input_fname = os.path.join(dir_path, '01_outlier_removal.csv')
output_fname = os.path.join(dir_path, '02_imputation.csv')

def _sorting(recordss, sort_key, reverse=True):
    for key, records in recordss.items():
        records = sorted(records, key=itemgetter(sort_key))
        recordss[key] = records

def _imputation(records, header):
    prev_val = copy.copy(records[0])
    for record in records:
        for key in header:
            val = record[key]
            if val == 'NA':
                record[key] = prev_val[key]
            else:
                prev_val[key] = val

def _get_normal_value(recordss, header):
    print ('getting normal value...')
    values = np.zeros([len(header)])
    counts = np.zeros([len(header)])
    for key, records in recordss.items():
        for record in records:
            for idx, key in enumerate(header):

                try:
                    val = float(record[key])
                    values[idx] += val
                    counts[idx] += 1
                except:
                    pass

    means = values / counts
    return means

def _impute_NAs(recordss, header, avgs):
    print ('imputing N/As with normal values...')
    for key, records in recordss.items():
        for record in records:
            for idx, key in enumerate(header):

                try:
                    val = float(record[key])
                except:
                    record[key] = str(avgs[idx])

def _subset_extractor(records, header):
    subset = []
    for record in records:
        subset_ = {}
        for name in header:
            subset_[name] = record[name]
        subset.append(subset_)
    return subset

def main():
    records = read_csv(input_fname)
    records = _subset_extractor(records, INFO_HEADER + VITAL_SIGNS_HEADER \
                                + LAB_HEADER)
    recordss = hashing(records, 'ICUSTAY_ID')
    _sorting(recordss, 'TIMESTAMP')
    print('#icu_stays: %d'%len(recordss))

    for key, records in recordss.items():
        _imputation(records, VITAL_SIGNS_HEADER + LAB_HEADER)

    avgs = _get_normal_value(recordss, VITAL_SIGNS_HEADER + LAB_HEADER)
    _sorting(recordss, 'TIMESTAMP', reverse=True)
    for key, records in recordss.items():
        _imputation(records, VITAL_SIGNS_HEADER + LAB_HEADER)

    _impute_NAs(recordss, VITAL_SIGNS_HEADER + LAB_HEADER, avgs)

    _sorting(recordss, 'TIMESTAMP')

    recordss_csv = []
    for key in sorted(recordss.keys()):
        records = recordss[key]
        recordss_csv.extend(records)

    write_csv(output_fname, INFO_HEADER + VITAL_SIGNS_HEADER + \
               LAB_HEADER, recordss_csv)

if __name__ == '__main__':
    main()
