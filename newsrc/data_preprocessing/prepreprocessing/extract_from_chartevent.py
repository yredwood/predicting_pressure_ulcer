import pandas as pd
import pdb

def collect_row(records, itemid_list):
    for i, record in records.iterrows():
        if record['ITEMID'] in itemid_list:
            print ('here we are, ') 
            print (record)
                    

    pass



if __name__ == '__main__':
    input_fname = '/nfs/jarvis/ext01/mike/DataSet/medical.datasets/physionet.org/files/mimiciii/1.4/CHARTEVENTS.csv.gz'
    records = pd.read_csv(input_fname)
    collect_row(records, [548, 224066, 227952])

