import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import copy
from operator import itemgetter
from utils import read_csv, write_csv, hashing

from headers import INFO_HEADER, VITAL_SIGNS_HEADER, LAB_HEADER
import pdb

'''
imputing data from the statistics file and then make numpy array
for training data
'''

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
#    'Neutrophils', 'Platelet Count', 'White Blood Cells'
#]

HEADER = VITAL_SIGNS_HEADER + LAB_HEADER + ['Position Change', 'Pressure Reducing Device']

parser = argparse.ArgumentParser()
#parser.add_argument('--results_path', type=str, default='./results')
args = parser.parse_args()

output_root = './datasets'

input_fname = os.path.join(output_root, '03_add_position_features.csv')
output_fname = os.path.join(output_root, '04_imputing.csv')


def main():
    records = read_csv(input_fname)

    # first get statistics
    neg_values = {}
    for record in records:
        if record['case/control'] == 'case':
            continue
        
        for name in HEADER:
            val = record[name]
            if val != 'NA':
                neg_values[name] = neg_values.get(name, list())
                neg_values[name].append(float(val))

    print ('negative mean/std')
    for name in HEADER:
        print ('{:40s} | mean: {:.3f}, std: {:.3f}'.format(
            name, np.mean(neg_values[name]), np.std(neg_values[name])))


    # and then imputing values with carry forward in the middle & negative mean with no value
    # multiproc needed
    recordss = hashing(records, 'ICUSTAY_ID')
    cnt = 0
    for key, records in recordss.items():
        print ('processing {}... {} / {}'.format(key, cnt, len(recordss)))
        cnt += 1

        records = sorted(records, key=itemgetter('TIMESTAMP'))
        # set intitial value
        prev_val = copy.copy(records[0])
        for key in HEADER:
            if prev_val[key] == 'NA':
                prev_val[key] = np.mean(neg_values[key])
        
        for record in records:
            for key in HEADER:
                val = record[key]
                if val == 'NA':
                    record[key] = prev_val[key]
                else:
                    prev_val[key ] = val

    recordss_csv = []
    for key in sorted(recordss.keys()):
        records = recordss[key]
        recordss_csv.extend(records)

    write_csv(output_fname, INFO_HEADER + HEADER, recordss_csv)




if __name__ == '__main__':
    main()
