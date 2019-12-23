import csv
import os
import numpy as np
import copy
import pickle
from utils import read_csv, write_csv, hashing
from operator import itemgetter
from datetime import datetime, timedelta
import pdb

'''
HEADER = ['LOS', 'age_at_admission', 'CHF', 'Arrhy', 'VALVE', 'PULMCIRC',
          'PERIVASC', 'HTN', 'PARA', 'NEURO', 'CHRNLUNG', 'DM', 'HYPOTHY',
          'RENLFAIL', 'LIVER', 'ULCER', 'AIDS', 'LYMPH', 'METS', 'TUMOR',
          'ARTH', 'COAG', 'OBESE', 'WGHTLOSS', 'LYTES', 'BLDLOSS', 'ANEMDEF',
          'ALCOHOL', 'DRUG', 'PSYCH', 'DEPRESS']
'''
HEADER = ['age_at_admission', 'CHF', 'Arrhy', 'VALVE', 'PULMCIRC',
          'PERIVASC', 'HTN', 'PARA', 'NEURO', 'CHRNLUNG', 'DM', 'HYPOTHY',
          'RENLFAIL', 'LIVER', 'ULCER', 'AIDS', 'LYMPH', 'METS', 'TUMOR',
          'ARTH', 'COAG', 'OBESE', 'WGHTLOSS', 'LYTES', 'BLDLOSS', 'ANEMDEF',
          'ALCOHOL', 'DRUG', 'PSYCH', 'DEPRESS']  # zero-one values

STR_HEADER = ['Gender', 'Insurance2', 'Race2']
STR_HEADER_CLASS = [['F', 'M'], ['Private', 'Public', 'Self'],
        ['Non-WHITE', 'WHITE']]


dir_path = './datasets'
input_root = './input_datasets'
input_fname  = os.path.join(dir_path, '05_converter.pkl')
icd_fname    = os.path.join(input_root, 'icd_code.csv')
output_fname = os.path.join(dir_path, '06_feature_eng.pkl')

def _subset_extractor(records, header):
    subset = []
    for record in records:
        subset_ = {}
        for name in header:
            subset_[name] = record[name]
        subset.append(subset_)
    return subset

def main():
    icd_codes = read_csv(icd_fname)
    icd_codes = hashing(icd_codes, 'ICUSTAY_ID')

    with open(input_fname, 'rb') as fp:
        records = pickle.load(fp)

    mats = records['mats']
    labels = records['ys']

    xs = [] # static features
    ys = [] # case/control
    icustay_id = []
    for key, data in mats.items():
        avg = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        max_val = np.max(data, axis=0)
        min_val = np.min(data, axis=0)

        icd_code = icd_codes[key]
        assert len(icd_code) == 1
        _icd_code = icd_code[0]
        icd_code = [_icd_code[name] for name in HEADER]
        for _i in range(len(STR_HEADER)):
            c = STR_HEADER_CLASS[_i].index(_icd_code[STR_HEADER[_i]])
            icd_code.extend([str(c)])

        #feat = np.concatenate((avg, std, max_val, min_val, [float(icd_code[0])]))
        feat = np.concatenate((avg, std, max_val, min_val, icd_code))
        xs.append(np.expand_dims(feat, axis=0))

        if labels[key] == 'control':
            ys.append(0)
        elif labels[key] == 'case':
            ys.append(1)
        else:
            raise ValueError()
        
        icustay_id.append(key)

    cs = np.zeros(len(feat), dtype=int)
    cs[-(len(icd_code)-1):] = 1  # if its categorical (1) or not (0)
    xs = np.concatenate(xs, axis=0)
    ys = np.asarray(ys)

    print(xs.shape, ys.shape)
    data = {
        'xs': xs,
        'cs': cs,
        'ys': ys,
        'icustay_id': icustay_id,
    }
    print('ratio: %.4f'%(np.sum(ys==1) / len(ys)))
    print(xs[0], ys[0])
    print(xs[-1], ys[1])

    with open(output_fname, 'wb') as fp:
        pickle.dump(data, fp)

if __name__ == '__main__':
    main()
