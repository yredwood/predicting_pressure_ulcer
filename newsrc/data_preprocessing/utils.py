from openpyxl import load_workbook
from constants import DTYPES
import pandas as pd
import pdb

HEADER = [
    'HADM_ID',
    'ICUSTAY_ID', 
    'START_TIME', 
    'END_TIME',
    'SORETIME',
    'EVENTTIME',
    'DBP',
    'MBP',
    'SBP',
    'activity',
    'friction/shear',
    'mobility',
    'moisture',
    'nutrition',
    'sensory/perception',
    'gcs-eyeopening',
    'gcs-motorresponse',
    'gcs-verbalresponse',
    'HR',
    'SaO2',
    'SpO2',
    'RR',
    'temperature',
    'ph/bloodgas',
    'pO2/bloodgas',
    'pCO2/bloodgas',
    'Bicarbonate',
    'CalculatedBicarbonate',
    'Hemoglobin (hematology)',
    'Hemoglobin (blood gas)',
    'Hematocrit',
    'CalculatedHematocrit',
    '_WBCCount',
    'WBC',
    'Neutrophils',
    'Platelet Count',
    'INR(PT)',
    'Glucose (ch)',
    'Glucose (blood gas)',
    'Lactate',
    'Sodium (ch)',
    'Sodium (whole blood)',
    'Potassium (ch)',
    'Potassium (whole blood)',
    'Total Calcium',
    'Phosphate',
    '_Total Protein',
    'Albumin',
    'ALT',
    'AST',
    'Total Bilirubin',
    'Urea Nitrogen',
    'Creatinine',
    'Uric Acid',
    'C-Reactive Protein',
    'Troponin I',
    'Troponin T'
]


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


def _test_read_csv():
    read_csv('/home/mike/codes/predicting_pressure_ulcer/preprocessed/mimic_extracted.csv')


if __name__ == '__main__':
    input_fname = '/home/mike/codes/predicting_pressure_ulcer/dataset/1026/MIMIC_extracted.tsv'
    output_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/mimic_extracted.csv'
    tsv_to_csv(input_fname, output_fname)





    
