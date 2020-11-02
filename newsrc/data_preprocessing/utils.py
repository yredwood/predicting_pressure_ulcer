from openpyxl import load_workbook
import pandas as pd
import pdb

def tsv_to_csv(fname, output_fname):
    df = pd.read_csv(fname, sep='\t', header=None)
    new_header = [
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
    new_rows = []
    for row in df.iterrows():
        #row[0].split(',')
        
        row[1][0].split(',')
        row_list = row[1][0].split(',')
        for _r in row[1][1:]:
            if isinstance(_r, str):
                _r = _r.replace(',', ' ')
            row_list.append(_r)
            print (row[1][3])

        new_rows.append(row_list)

    output = pd.DataFrame(new_rows, columns=new_header)
    output.to_csv(output_fname, index=False)


def _test_read_xlsx():
    fname = '/home/mike/codes/predicting_pressure_ulcer/dataset/mimic_extracted/data_wo_pid.xlsx'
    read_xlsx(fname)

def _test_read_tsv():
    fname = '/home/mike/codes/predicting_pressure_ulcer/dataset/1026/MIMIC_extracted.tsv'
    read_original_tsv(fname)

if __name__ == '__main__':
    input_fname = '/home/mike/codes/predicting_pressure_ulcer/dataset/1026/MIMIC_extracted.tsv'
    output_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/mimic_extracted.csv'
    tsv_to_csv(input_fname, output_fname)




    
