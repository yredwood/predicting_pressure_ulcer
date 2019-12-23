import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from utils import read_csv, write_csv, hashing

import pdb

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
STATS_HEADER = VITAL_SIGNS_HEADER + LAB_HEADER

plt.style.use('seaborn')

parser = argparse.ArgumentParser()
parser.add_argument('--refined', action='store_true')
parser.add_argument('--data_path', type=str, default='./datasets')
parser.add_argument('--results_path', type=str, default='./results')
args = parser.parse_args()

data_path = args.data_path
results_path = args.results_path
if args.refined:
    input_fname = os.path.join(data_path, 'LAB_CHART_EVENTS_TABLE_refined.csv')
    output_fname = os.path.join(data_path, 'stats_refined.csv')
    results_path = os.path.join(results_path, 'refined')
else:
    input_fname = os.path.join(data_path, 'EVENTS_TABLE_sorted_and_averaged.csv')
    output_fname = os.path.join(data_path, 'stats.csv')
    results_path = os.path.join(results_path, 'original')

if not os.path.exists(results_path):
    os.makedirs(results_path)

def main():
    values = {}
    records = read_csv(input_fname)
    records_dict = hashing(records, 'ICUSTAY_ID')
    print('#icu_stays: %d'%len(records_dict))
    

    vals = [[] for _ in range(len(STATS_HEADER))]

    for key, record in records_dict.items():
        # single measure
        # only negative samples
        if record[0]['case/control'] == 'case':
            continue

        for measure in record:
            for head, value in measure.items():
                if head in STATS_HEADER:
                    hidx = STATS_HEADER.index(head)
                    try:
                        vals[hidx].extend([float(value)])
                    except:
                        pass
        pdb.set_trace()
    pdb.set_trace()
    output_csv = os.path.join(results_path, 'feature_stats_for_norm.csv')
    with open(output_csv, 'wt') as f:
        output_str = '{:15s},{:5s},{:5s}'.format('Feature', 'Avg', 'Std')
        f.writeline(output_str + '\n')
        for i, header in enumerate(STATS_HEADER):
            output_str = '{:15s},{:.3f},{:.3f}'.format(header,
                np.mean(vals[i]),
                np.std(vals[i]),
            )
            print (output_str)
            f.writeline(output_str + '\n')
            


    

if __name__ == '__main__':
    main()
