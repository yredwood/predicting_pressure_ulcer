import pickle
import os

import numpy as np

from utils import read_csv, hashing
from headers import INFO_HEADER, VITAL_SIGNS_HEADER, LAB_HEADER

import pdb

DYNAMIC_HEADER = VITAL_SIGNS_HEADER + LAB_HEADER + ['Position Change', 'Pressure Reducing Device']

HEADER = ['age_at_admission', 'CHF', 'Arrhy', 'VALVE', 'PULMCIRC',
          'PERIVASC', 'HTN', 'PARA', 'NEURO', 'CHRNLUNG', 'DM', 'HYPOTHY',
          'RENLFAIL', 'LIVER', 'ULCER', 'AIDS', 'LYMPH', 'METS', 'TUMOR',
          'ARTH', 'COAG', 'OBESE', 'WGHTLOSS', 'LYTES', 'BLDLOSS', 'ANEMDEF',
          'ALCOHOL', 'DRUG', 'PSYCH', 'DEPRESS']  # zero-one values

STR_HEADER = ['Gender', 'Race2', 'Insurance2'] 
STR_HEADER_CLASS = [['F', 'M'], 
        ['Non-WHITE', 'WHITE'],
        ['Private', 'Public', 'Self']]

STATIC_HEADER = HEADER + STR_HEADER

dir_path = './datasets'
input_root = './input_datasets'
output_root = os.path.join(dir_path, '09_timeline_data')

def makedirs(name):
    if not os.path.exists(name):
        os.makedirs(name)

def read_pkl(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

def _subset_extractor(records, header):
    subset = []
    for record in records:
        subset_ = {}
        for name in header:
            subset_[name] = record[name]
        subset.append(subset_)
    return subset


if __name__=='__main__':
    try:
        seed_num = int(sys.argv[1])
    except:
        seed_num = 0

    makedirs(output_root)
    _05_fname = os.path.join(dir_path, '05_converter.pkl')
    _icd_fname = os.path.join(input_root, 'icd_code.csv')
    _meta_fname = os.path.join(dir_path, 'meta_data.pkl')

    

    dynamic_data = read_pkl(_05_fname)
    meta_data = read_pkl(_meta_fname)
    icd_codes = read_csv(_icd_fname)
    icd_codes = hashing(icd_codes, 'ICUSTAY_ID')

    # copy file
    save_pkl(meta_data, os.path.join(output_root, 'meta_data.pkl'))

    mats = dynamic_data['mats']
    labels = dynamic_data['ys']

    for key, data in mats.items():
        # only use test data
        if key not in meta_data['test_icuid']:
            continue

        data_length = data.shape[0]
        if data_length < 90:
            continue
        fname = os.path.join(output_root, key + '.pkl')
        
        icd_code = icd_codes[key]
        assert len(icd_code)==1
        _icd = icd_code[0]
        icd_code = [_icd[name] for name in HEADER]
        for _i in range(len(STR_HEADER)-1):
            c = STR_HEADER_CLASS[_i].index(_icd[STR_HEADER[_i]])
            icd_code.extend([str(c)])

        if _icd['Insurance2'] == 'Private':
            icd_code.extend(['1', '0'])
        elif _icd['Insurance2'] == 'Public':
            icd_code.extend(['0', '1'])
        else:
            icd_code.extend(['0', '0'])
        
        xs = []
        xd = []
        ys = []

        # starting from 12 to the end
        for time in range(72,data_length):
            data_t = data[:time]
            avg = np.mean(data_t, axis=0)
            std = np.std(data_t, axis=0)
            max_val = np.max(data, axis=0)
            min_val = np.min(data, axis=0)

            static_feat = np.concatenate((avg, std, max_val, min_val, icd_code))
            static_feat = np.array([float(_s) for _s in static_feat])


            # standardization
            avg = meta_data['dynamic_avg']
            std = meta_data['dynamic_std']
            data_t = (data_t - avg) / (std+1e-8)

            avg = meta_data['static_avg']
            std = meta_data['static_std']
            static_feat = (static_feat - avg) / (std+1e-8)
            
            xd.append(data_t)
            xs.append(static_feat)
            if labels[key] == 'case':
                ys.append(1)
            else:
                ys.append(0)
            
        save_data = {
            'xd': xd,
            'xs': xs,
            'y': ys
        }

        save_pkl(save_data, fname)
    













#
