import csv
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from operator import itemgetter

from utils import read_csv, write_csv, hashing
from headers import INFO_HEADER, VITAL_SIGNS_HEADER, LAB_HEADER


'''
plot per-case time-line data with all of dynamic features
'''
import matplotlib
matplotlib.use('Agg')

LAB_HEADER = LAB_HEADER + ['Position Change', 'Pressure Reducing Device']

data_root = './datasets'
#input_fname = os.path.join(data_root, '03_add_position_features.csv')
input_fname = os.path.join(data_root, '04_imputing.csv')
print ('plot data from {}...'.format(input_fname))

output_root = os.path.join(data_root, '98_plot_data')
if not os.path.exists(output_root):
    os.makedirs(output_root)


def plot(xs, ys, headers, fname):
    
    num_features = len(xs)
    assert len(xs) == len(ys)

    fig = plt.figure(0, figsize=(10,num_features*2))
    outer = gridspec.GridSpec(num_features, 1)
    for hi in range(num_features):
        board = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=outer[hi])
        axes = plt.subplot(board[0])
        axes.plot(xs[hi], ys[hi], color='r', marker='o', markersize=4, label=headers[hi])
        axes.legend(loc='upper left')
    
    plt.tight_layout()
    print ('writing {}...'.format(fname))
    plt.savefig(os.path.join(output_root, fname))


def main():
    records = read_csv(input_fname)
    recordss = hashing(records, 'ICUSTAY_ID')
    
    for key, records in recordss.items():

        headers = VITAL_SIGNS_HEADER + LAB_HEADER
        xs = [[] for _ in range(len(headers))] # relative time for start point in hours
        ys = [[] for _ in range(len(headers))] # actual value of the feature at that point
        label = records[0]['case/control']

        records = sorted(records, key=itemgetter('TIMESTAMP'))

        for t, record in enumerate(records):
            for hi, hname in enumerate(headers):
                value = record[hname]
                if value != 'NA':
                    xs[hi].append(t)
                    ys[hi].append(float(value))
        
        fname = '{}_{}.png'.format(key, label)
        plot(xs, ys, headers, fname)

if __name__=='__main__':
    main()

