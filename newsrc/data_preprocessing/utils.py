from openpyxl import load_workbook
from constants import DTYPES, HEADER
import pandas as pd
import pdb
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime

def tsv_to_csv(fname, output_fname):
    df = pd.read_csv(fname, sep='\t', header=None)
    new_rows = []
    for row in df.iterrows():
        #row[0].split(',')
        
        row[1][0].split(',')
        row_list = row[1][0].split(',')
        for _r in row[1][1:5]:
            if isinstance(_r, str):
                _r = _r.replace(',', ' ')
            row_list.append(_r)

        for _r in row[1][5:]:
            try:
                _r = float(_r)
            except:
                _r = float('nan')

            row_list.append(_r)

        new_rows.append(row_list)

    output = pd.DataFrame(new_rows, columns=HEADER)
    output.to_csv(output_fname, index=False)

def _hashing(records, key):
    records_dict = {}
    for i, record in records.iterrows():
        r_key = record[key]
        records_dict[r_key] = records_dict.get(r_key, list())
        records_dict[r_key].append(record)
    return records_dict

def hashing(csv_fname, key, chunksize=100000, dtypes=None, num_workers=10):
    import pdb
    #for df_chunk in tqdm(pd.read_csv(csv_fname, dtype=dtypes, chunksize=100)):
    def _hash(records, key):
        records_dict = {}
        for i, record in records.iterrows():
            r_key = record[key]
            records_dict[r_key] = records_dict.get(r_key, list())
            #records_dict[r_key].append(record)
        return records_dict 

    results = Parallel(n_jobs=num_workers)(delayed(_hash)(
        records=df_chunk,
        key=key) for df_chunk in tqdm(pd.read_csv(csv_fname, dtype=dtypes, chunksize=chunksize)))

    overlapped_keys = []
    for r in results:
        for key in r.keys():
            if key not in overlapped_keys:
                overlapped_keys.append(key)
            else:
                print (key)

strptime_s = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
strptime_m = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M')
def strptime(x):
    if '.' in x:
        x = x[:x.index('.')]

    try:
        return strptime_s(x)
    except:
        return strptime_m(x)
    
def _test_hashing(csv_fname, output_fname=None):
    from constants import DTYPES
    import time
    s0 = time.time()
    #data = pd.read_csv(csv_fname, dtype=DTYPES)
    recordss = hashing(csv_fname, 'ICUSTAY_ID', dtypes=DTYPES, num_workers=40)
#    icu_ids = [str(r) for r in recordss.keys()]
#    # save it
#    if output_fname is not None:
#        with open(output_fname, 'wt') as f:
#            f.writelines('\n'.join(icu_ids))
        
    print ('elapsed time: ', s0 - time.time())

if __name__ == '__main__':
#    input_fname = '/home/mike/codes/predicting_pressure_ulcer/dataset/1026/MIMIC_extracted.tsv'
#    output_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/mimic_extracted.csv'
#    tsv_to_csv(input_fname, output_fname)

    input_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/01_outlier.csv'
    output_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/icustay_ids.txt'
    _test_hashing(input_fname, output_fname)




    
