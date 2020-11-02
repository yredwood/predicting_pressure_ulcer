import csv
import os
import numpy as np
from utils import read_csv, write_csv

from headers import INFO_HEADER, VITAL_SIGNS_HEADER, LAB_HEADER

import pdb


'''
INFO_HEADER = ['case/control', 'ICUSTAY_ID', 'START_TIME', 'END_TIME',
               'TIMESTAMP', 'TIME_from_START', 'TIME_to_END']

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
'''

RANGE = {
    # (min, max)
    'BRANDEN_SCORE'    : (6.0, 23.0),
    'GCS'              : (3.0, 15.0),
    'HR'               : (0.0, 300.0),
    'RR'               : (0.0, 100.0),
    'TEMPERATURE'      : (13.7, 50.0),
    'SBP'              : (0.0, 400.0),
    'DBP'              : (0.0, 400.0),
    'MBP'              : (0.0, 400.0),
    'SaO2'             : (0.0, 100.0),
    'SpO2'             : (0.0, 100.0),
    'FiO2'             : (0.0, 100.0),
    'Oxygen Saturation': (25.0, 100)
}

RANGE_C = {
    'Glucose'          : 3555,
    'Troponin I'       : 575
}

RANGE_S = {
        'pO2'          : 400
}

'''
# need more?
    'Lactate'          : (),
    'Oxygen Saturation': (),
    'pCO2'             : (),
    'pH'               : (),
    'pO2'              : (),
    'Albumin'          : (),
    'Bicarbonate'      : (),
    'Total Bilirubin'  : (),
    'Creatinine'       : (),
    'Glucose'          : (),
    'Potassium'        : (),
    'Sodium'           : (),
    'Troponin I'       : (),
    'Troponin T'       : (),
    'Urea Nitrogen'    : (),
    'Hematocrit'       : (),
    'Hemoglobin'       : (),
    'INR(PT)'          : (),
    'Neutrophils'      : (),
    'Platelet Count'   : (),
    'White Blood Cells': ()
'''

input_root = './input_datasets'
input_fname = os.path.join(input_root, 'LAB_CHART_EVENTS_TABLE.csv')

output_root = './datasets'
if not os.path.exists(output_root):
    os.makedirs(output_root)
output_fname = os.path.join(output_root, '01_outlier_removal.csv')

def _subset_extractor(record, header):
    info_record = {}
    for name in header:
        info_record[name] = record[name]
    return info_record

def _refiner(record):
    refined_record = record.copy()
    for key, value in RANGE.items():
        value_ = record[key]
        try:
            value_ = float(value_)
        except ValueError:
            refined_record[key] = 'NA'
            continue
        if value_ < value[0] or value_ > value[1]:
            refined_record[key] = 'NA'
        else:
            refined_record[key] = value_

    for key, value in RANGE_C.items():
        value_ = record[key]
        try:
            value_ = float(value_)
        except ValueError:
            refined_record[key] = 'NA'
            continue
        refined_record[key] = 'NA' if value_ == value else value_

    for key, value in RANGE_S.items():
        value_ = record[key]
        try:
            value_ = float(value_)
        except ValueError:
            refined_record[key] = 'NA'
            continue

        refined_record[key] = 'NA' if value_ > value else value_

    return refined_record

def main():
    refined = []
    records = read_csv(input_fname)
    for record in records:
        _refined_record = _refiner(record)
        _vital_record = _subset_extractor(_refined_record, VITAL_SIGNS_HEADER)
        _info_record = _subset_extractor(_refined_record, INFO_HEADER)
        _lab_record = _subset_extractor(_refined_record, LAB_HEADER)
        
        refined.append({**_vital_record, **_info_record,
                                **_lab_record})

    write_csv(output_fname, INFO_HEADER + VITAL_SIGNS_HEADER \
               + LAB_HEADER, refined)


if __name__ == '__main__':
    main()


