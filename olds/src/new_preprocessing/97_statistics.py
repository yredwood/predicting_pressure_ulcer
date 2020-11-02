import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from utils import read_csv, write_csv

from headers import INFO_HEADER, VITAL_SIGNS_HEADER, LAB_HEADER

plt.style.use('seaborn')

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

HEADER = VITAL_SIGNS_HEADER + LAB_HEADER

OUTPUT_HEADER = ['Variable', 'Missing rate', 'Average', 'Std.', 'Max', 'Min']

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./input_datasets/LAB_CHART_EVENTS_TABLE_sampled.csv')
parser.add_argument('--set_name', type=str, default='full', choices=['full', 'pos', 'neg'], 
        help='statistics for selected subset')
#parser.add_argument('--results_path', type=str, default='./results')
args = parser.parse_args()

output_root = './datasets/97_statitics'
if not os.path.exists(output_root):
    os.makedirs(output_root)

infname = args.data_path.split('/')[-1].replace('.csv', '')
output_fname = os.path.join(output_root, 'stats_{}_{}.csv'.format(args.set_name, infname))


def _count_na(data):
    cnt = 0
    for datum in data:
        if datum == 'NA':
            cnt += 1
    return cnt

def _exclude_na(data):
    subset = []
    for datum in data:
        try:
            datum = float(datum)
        except ValueError:
            continue
        subset.append(datum)
    return subset

def _subset(ctype):
    if args.set_name == 'full':
        return True
    if args.set_name == 'pos' and ctype == 'case':
        return True
    if args.set_name == 'neg' and ctype == 'control':
        return True
    return False

def main():
    values = {}
    records = read_csv(args.data_path)
    for record in records:
        if not _subset(record['case/control']):
            continue
        
        for name in HEADER:
            val = record[name]
            values[name] = values.get(name, list())
            values[name].append(val)

    stats = []
    for name in HEADER:
        value = values[name]
        n_total = len(value)
        n_na = _count_na(value)
        refined_value = _exclude_na(value)

#        fig, ax = plt.subplots()
#        ax.boxplot(refined_value)
#        plt.xticks([1], [name])
#        plt.savefig(os.path.join(results_path, '%s.png'%name))

        stat = {
            'Variable': name,
            'Missing rate': n_na/n_total,
            'Average': np.mean(refined_value),
            'Std.': np.std(refined_value),
            'Max' : np.max(refined_value),
            'Min' : np.min(refined_value)
        }

        stats.append(stat)
        print('%s, %.4f, %.4f, %.4f, %.4f, %.4f'%(
              name, n_na/n_total, np.mean(refined_value), np.std(refined_value),
              np.max(refined_value), np.min(refined_value)))

    write_csv(output_fname, OUTPUT_HEADER, stats)

if __name__ == '__main__':
    main()
