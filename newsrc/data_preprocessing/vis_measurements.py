import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd
import math

from joblib import Parallel, delayed

from timeslot_and_imputation import multiproc_read_records
from utils import strptime
matplotlib.use('Agg')

import pdb


def plot_with_colors(fname, color_points_dict, yticks=None):
    '''
    input:
        {
            color_markersize: [1., 3., ....],
            color_markersize: [3, 4., ...],
        }
    '''
    fig = plt.figure(figsize=(40,20))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    
    for color_markersize, points in color_points_dict.items():
        color, size = color_markersize.split('_')
        size = int(size)

        # points = [[1,2,3...], [4,5,1,...,], ...]
        for idx in range(len(points)):
            ax.plot(points[idx], [idx for _ in range(len(points[idx]))], color, markersize=size)
    
    if yticks is not None:
        ax.set_yticks([i for i in range(len(yticks))])
        ax.set_yticklabels(yticks)
    
    #plt.tight_layout()
    plt.savefig(fname)


def join_feature(record_a, header_a, record_b, header_b, fixed_header):
    '''
    join record_b into record_a. if record_b doesn't have any feature list in record_a,
    then nan will be filled. also, fixed header will not be filled with nan and 
    will be filled with record_a's values
    
    this function is not symetric! only record_a has fixed header
    '''
    
    joined_records = []

    additional_header = []
    for h in header_b:
        if h not in header_a:
            additional_header.append(h)

    for record in record_a:
        joined_records.append(record + [float('nan') for _ in range(len(additional_header))])

    for record in record_b:
        _rcd = []
        for h in header_a:
            if h in fixed_header:
                _rcd.append(record_a[0][header_a.index(h)])
            elif h in header_b:
                _rcd.append(record[header_b.index(h)])
            else:
                _rcd.append(float('nan'))
        
        for h in additional_header:
            _rcd.append(record[header_b.index(h)])

        joined_records.append(_rcd)


    # sorting needed
    new_header = header + additional_header
    return joined_records, new_header
        
def get_prd_points(records, header):

    start_time = strptime(records[0][header.index('EVENTTIME')])
    measure_points = []
    prd_1_points = []
    prd_0_points = []
    for record in records[1:]:
        current_time = strptime(record[header.index('EVENTTIME')])
        xp = (current_time - start_time).total_seconds() / 3600.
        measure_points.append(xp)

        prd = record[header.index('PRD')]
        if not math.isnan(prd):
            if prd:
                prd_1_points.append(xp)
            else:
                prd_0_points.append(xp)
    
    return measure_points, prd_1_points, prd_0_points

def get_sore_points(records, header):
    start_time = strptime(records[0][header.index('EVENTTIME')])
    measure_points = []
    for record in records[1:]:
        current_time = strptime(record[header.index('EVENTTIME')])
        xp = (current_time - start_time).total_seconds() / 3600.
        measure_points.append(xp)

    soretime = records[0][header.index('SORETIME')]
    if isinstance(soretime, float):
        soretime = []
    else:
        soretime = [ (strptime(soretime) - start_time).total_seconds() / 3600. ]
    return measure_points, soretime
    
def get_measurepoints(records, header, dynamic_event_id, static_event_id=None):
    '''
    input:
        records: list of measurements
        header: header

    output:
        points: list of timepionts relative to start point 
        soretime: soretime
    '''
    
    start_time = strptime(records[0][header.index('EVENTTIME')])
    #endtime = strptime(records[0][header.index
    points = []
    for record in records[1:]:
        current_time = strptime(record[header.index('EVENTTIME')])
        points.append((current_time - start_time).total_seconds() / 3600.)

    event_time = records[0][header.index(event_id)]
    if isinstance(soretime, float):
        event_time = None
    else:
        event_time = (strptime(soretime) - start_time).total_seconds() / 3600.

    return points, event_time

def prd_hashing(fname):
    output = {}
    df = pd.read_csv(fname)
    header = list(df.head())
    header[1] = 'EVENTTIME'
    header[2] = 'PRD'
    for idx, row in df.iterrows():
        key = str(row['ICUSTAY_ID'])
        output[key] = output.get(key, list()) 
        output[key].append( [key, row['CHARTTIME'], row['VALUE']] )
    
    return output, header
    
if __name__ == '__main__':

    # example 1. vis pressure ulcer onset point
    example_1 = True
    if example_1:

        input_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/01_outlier.csv'
        output_fname = '/home/mike/codes/predicting_pressure_ulcer/visutalization/measurement_points.png'
        input_prd = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/pressure_reducing_device.csv'
        record_dict, header = multiproc_read_records(input_fname, 'EVENTTIME', debug=True)
        
        mpoint_list, sore_list = [], []
        key_list = []
        for key, records in record_dict.items():
            mp, sp = get_sore_points(records, header)
            if len(sp) > 0:
                mpoint_list.append(mp)
                sore_list.append(sp)
                key_list.append(key)

        cpd = {
            'k._8': mpoint_list,
            'r*_12': sore_list,
        }
        plot_with_colors(output_fname, cpd, yticks=key_list)

    
    # example 2. vis prd points
    example_2 = False
    if example_2:
        input_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/01_outlier.csv'
        output_fname = '/home/mike/codes/predicting_pressure_ulcer/visutalization/prd_points.png'
        input_prd = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/pressure_reducing_device.csv'

        record_dict, header = multiproc_read_records(input_fname, 'EVENTTIME', debug=True)
        prd_records, prd_header = prd_hashing(input_prd)
        fixed_header = ['HAM_ID', 'ICUSTAY_ID', 'START_TIME', 'END_TIME', 'SORETIME']
        
        mpoint_list, ptrue_list, pfalse_list = [], [], []
        sore_list = []
        key_list = []
        for key, records in record_dict.items():
            if key in prd_records:
                joined_records, joined_header = join_feature(records, header, prd_records[key], prd_header, fixed_header)
                _, sp = get_sore_points(records, header)
                mp, pt, pf = get_prd_points(joined_records, joined_header)
                mpoint_list.append(mp)
                ptrue_list.append(pt)
                pfalse_list.append(pf)
                key_list.append(key)
                sore_list.append(sp)
            else:
                #print ('key {} not in prd.csv'.format(key))
                pass

            if len(mpoint_list) > 20:
                break
    
        cpd = {
            'k._8': mpoint_list,
            'b*_12': ptrue_list,
            'r*_12': pfalse_list,
            'g+_13': sore_list,
        }
        plot_with_colors(output_fname, cpd, yticks=key_list)



#    for key, records in record_dict.items():
#        records.extend(prd_records[key])
#        p, s = get_measurepoints(records, header, 'SORETIME')















