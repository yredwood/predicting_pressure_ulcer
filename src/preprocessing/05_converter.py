import csv
import os
import numpy as np
import copy
import pickle
from utils import read_csv, write_csv, hashing
from operator import itemgetter
from datetime import datetime, timedelta
import pdb

from headers import INFO_HEADER, VITAL_SIGNS_HEADER, LAB_HEADER

#INFO_HEADER = [
#    'case/control', 'ICUSTAY_ID', 'START_TIME', 'END_TIME',
#    'TIMESTAMP', 'TIME_from_START', 'TIME_to_END'
#]
#
#VITAL_SIGNS_HEADER = [
#    'BRANDEN_SCORE', 'GCS', 'HR', 'RR', 'TEMPERATURE',
#    'SBP', 'DBP', 'MBP', 'SaO2', 'SpO2'
#]
#
#LAB_HEADER = [
#    'Lactate', 'Oxygen Saturation', 'pCO2', 'pH', 'pO2',
#    'Albumin', 'Bicarbonate', 'Total Bilirubin', 'Creatinine',
#    'Glucose', 'Potassium', 'Sodium', 'Troponin I', 'Troponin T',
#    'Urea Nitrogen', 'Hematocrit', 'Hemoglobin', 'INR(PT)',
#    'Neutrophils', 'Platelet Count', 'White Blood Cells',
#    'Position Change', 'Pressure Reducing Device',
#]

LAB_HEADER = LAB_HEADER + ['Position Change', 'Pressure Reducing Device']

dir_path = './datasets'
input_fname = os.path.join(dir_path, '04_add_features.csv')
output_fname = os.path.join(dir_path, '05_converter.pkl')

def main():
    records = read_csv(input_fname)
    recordss = hashing(records, 'ICUSTAY_ID')
    print('#icu_stays: %d'%len(recordss))
    mats = {}
    ys = {}

    for key, records in recordss.items():
        mat = []
        for record in records:
            vec = [float(record[name]) for name in VITAL_SIGNS_HEADER + LAB_HEADER]
            vec = np.asarray(np.expand_dims(vec, axis=0))
            mat.append(vec)
        mat = np.concatenate(mat, axis=0)
        mat = mat.astype(np.float32)
        mats[key] = mat
        ys[key] = records[0]['case/control']

    data = {'mats': mats, 'ys': ys}
    with open(output_fname, 'wb') as fp:
        pickle.dump(data, fp)

if __name__ == '__main__':
    main()
