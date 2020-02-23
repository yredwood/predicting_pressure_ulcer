import pickle
import numpy as np
import os
import sys

from headers import INFO_HEADER, VITAL_SIGNS_HEADER, LAB_HEADER
import pdb


dir_path = './datasets'
testset_ratio = 0.40
upsample = 1 # 1 to keep original ratio

# input 06_...pkl and 07_...pkl and outputs 
# lstm (or MLP) train-test dataset

# headers from 05
DYNAMIC_HEADER = VITAL_SIGNS_HEADER + LAB_HEADER + ['Position Change', 'Pressure Reducing Device']


# headers from 06
from headers import INFO_HEADER, VITAL_SIGNS_HEADER, LAB_HEADER
HEADER = ['age_at_admission', 'CHF', 'Arrhy', 'VALVE', 'PULMCIRC',
          'PERIVASC', 'HTN', 'PARA', 'NEURO', 'CHRNLUNG', 'DM', 'HYPOTHY',
          'RENLFAIL', 'LIVER', 'ULCER', 'AIDS', 'LYMPH', 'METS', 'TUMOR',
          'ARTH', 'COAG', 'OBESE', 'WGHTLOSS', 'LYTES', 'BLDLOSS', 'ANEMDEF',
          'ALCOHOL', 'DRUG', 'PSYCH', 'DEPRESS']  # zero-one values

STR_HEADER = ['Gender', 'Race2', 'Private Insurance', 'Public Insurance']
STATIC_HEADER = HEADER + STR_HEADER


def read_pkl(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data 


if __name__=='__main__':
    try:
        # get random seed as an argument
        seed_num = int(sys.argv[1])
    except:
        seed_num = 0

    _05_fname = os.path.join(dir_path, '05_converter.pkl')
    _06_fname = os.path.join(dir_path, '06_feature_eng.pkl')

    dynamic_data = read_pkl(_05_fname)
    static_data = read_pkl(_06_fname)

    # matching labels 
    dynamic, static, label = [], [], []
    icu_ids = []
    for icu_id in sorted(dynamic_data['mats'].keys()):

        dynamic.append(dynamic_data['mats'][icu_id])
        static_icu_index = static_data['icustay_id'].index(icu_id)
        static.append([float(_s) for _s in static_data['xs'][static_icu_index]])
        icu_ids.append(icu_id)

        if dynamic_data['ys'][icu_id] == 'case':
            label.append(1)
        elif dynamic_data['ys'][icu_id] == 'control':
            label.append(0)
        else:
            raise ValueError()

        assert label[-1] == static_data['ys'][static_icu_index]
    label = np.array(label)

    # normalize data
    xall = np.concatenate(dynamic, axis=0)
    dy_avg = np.mean(xall, 0)
    dy_std = np.std(xall, 0)
    dynamic = np.array([(_x - dy_avg) / (dy_std+1e-8) for _x in dynamic])

    # for static, only normalize non-categorical features
    # >>>>> also normalize categorical variables
    static = np.array(static)
    st_avg = np.mean(static, 0)
    st_std = np.std(static, 0)
    for i in range(len(static[0])):
        #if not static_data['cs'][i]:
        static[:,i] = (static[:,i] - st_avg[i]) / (st_std[i] + 1e-8)

    # split train-val-test datasets
    split_idx = int(len(dynamic) * testset_ratio)
    np.random.seed(seed_num)
    rind = np.random.permutation(len(dynamic))

    tri = rind[split_idx:]
    tei = rind[:split_idx]

    train_xd = [dynamic[_i] for _i in tri]
    train_xs = [static[_i] for _i in tri]
    train_y = [label[_i] for _i in tri]

    test_xd = [dynamic[_i] for _i in tei]
    test_xs = [static[_i] for _i in tei]
    test_y = [label[_i] for _i in tei]

    test_icuid = [icu_ids[_i] for _i in tei]
    
    # test - valid split
    valid_xd = test_xd[len(tei)//2:]
    valid_xs = test_xs[len(tei)//2:]
    valid_y = test_y[len(tei)//2:]

    test_xd = test_xd[:len(tei)//2]
    test_xs = test_xs[:len(tei)//2]
    test_y = test_y[:len(tei)//2]

    if upsample > 1:
        positive_idx = np.where(np.array(train_y)==1)[0]
        add_x, add_xs, add_y = [], [], []

        for _ in range(upsample):
            train_xd.extend(dynamic[p] for p in positive_idx)
            train_xs.extend(static[p] for p in positive_idx)
            train_y.extend([1 for _ in range(len(positive_idx))])

    # save train/test data as pickle
    train_data = {
            'xd': train_xd,
            'xs': train_xs,
            'y': train_y,
    }
    test_data = {
            'xd': test_xd,
            'xs': test_xs,
            'y': test_y,
    }
    valid_data = {
            'xd': valid_xd,
            'xs': valid_xs,
            'y': valid_y,
    }
            
    with open(os.path.join(dir_path, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)

    with open(os.path.join(dir_path, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)

    with open(os.path.join(dir_path, 'valid_data.pkl'), 'wb') as f:
        pickle.dump(valid_data, f)
        
    # define header: index dictionary
    dh2ind = {}
    sh2ind = {}
    for i, h in enumerate(DYNAMIC_HEADER):
        dh2ind[h] = [i]
        sh2ind[h] = [i + len(DYNAMIC_HEADER) * _i for _i in range(4)] 
        # avg, std, max, min

    for i, h in enumerate(STATIC_HEADER):
        sh2ind[h] = [len(DYNAMIC_HEADER) * 4 + i]

    
    meta_data = {
        'dynamic_header': DYNAMIC_HEADER,
        'static_header': STATIC_HEADER,
        'dh2ind': dh2ind,
        'sh2ind': sh2ind,
        'dynamic_avg': dy_avg,
        'dynamic_std': dy_std,
        'static_avg': st_avg,
        'static_std': st_std,
        'test_icuid': test_icuid,
    }

    # save datasets
    with open(os.path.join(dir_path, 'meta_data.pkl'), 'wb') as f:
        pickle.dump(meta_data, f)

















    #
