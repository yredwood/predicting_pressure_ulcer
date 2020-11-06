import pandas as pd
import math
from constants import RANGE, DTYPES

import pdb

def get_stats(fname, output_fname):
    df = pd.read_csv(fname, dtype=DTYPES)
    atatistic_header = 'item,avg,std,min,max,exist_ratio'
    lines = [statistic_header]
    for label, content in df.iteritems():
        if not isinstance(content[0], float):
            continue

        try:
#            string = [label, content.mean(), content.std(), content.min(), content.max(), 
#                    content.notna().sum() / len(content)]
            _1mom = 0.
            _2mom = 0.
            _min = 1e+8
            _max = -1e+8
            num_exist = 0
            for _, c in content.iteritems():
                if math.isnan(c):
                    continue
                _1mom += c
                _2mom += c**2
                if c < _min:
                    _min = c
                if c > _max:
                    _max = c
            
                num_exist += 1
            avg = _1mom / num_exist
            std = _2mom / num_exist - avg**2
            
            string = [avg, std, _min, _max, num_exist / len(content)]
            string = ['{:.3f}'.format(s) for s in string]
            string = ','.join([label] + string)
            print (string)
            lines.append(string)
        except:
            print ('{} passed'.format(label))
            pass
        

    with open(output_fname, 'wt') as f:
        f.writelines('\n'.join(lines))


def remove_outlier(fname, output_fname):
    df = pd.read_csv(fname, dtype=DTYPES)

    for label, content in df.iteritems():
        if not isinstance(content[0], float):
            continue

        if label not in RANGE.keys():
            continue

        bound = RANGE[label]
        outlier_index = content.lt(bound[0]) | content.gt(bound[1])

        print ('{:20s} | outlier: {}'.format(label, outlier_index.sum())) 
        #content[outlier_index].replace(float('nan'))
        df[label][outlier_index] = float('nan')
        
    df.to_csv(output_fname, index=False)



if __name__ == '__main__':
    input_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/mimic_extracted.csv'
    output_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/stats_original.csv'
    #get_stats(input_fname, output_fname)
        
    output_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/01_outlier.csv'
    outout_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/stats_refined.csv'
    remove_outlier(input_fname, output_fname)
    get_stats(output_fname, outout_fname)

