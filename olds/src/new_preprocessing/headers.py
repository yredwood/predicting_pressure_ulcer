INFO_HEADER = [
    'case/control', 'ICUSTAY_ID', 'START_TIME', 'END_TIME',
    'TIMESTAMP', 'TIME_from_START', 'TIME_to_END']

VITAL_SIGNS_HEADER = [
    'BRANDEN_SCORE', 'GCS', 'HR', 'RR', 'TEMPERATURE',
    'SBP', 'DBP', 'MBP', 'SaO2', 'SpO2']

LAB_HEADER = [
    'Lactate', 'Oxygen Saturation', 'pCO2', 'pH', 'pO2',
    'Albumin', 'Bicarbonate', 'Total Bilirubin', 'Creatinine',
    'Glucose', 'Potassium', 'Sodium', 'Troponin I', 'Troponin T',
    'Urea Nitrogen', 'Hematocrit', 'Hemoglobin', 'INR(PT)',
    'Neutrophils', 'Platelet Count', 'White Blood Cells',
    'sensory_perception', 'moisture', 'activity', 'mobility',
    'nutrition', 'friction_shear']

all_feature_list = [
    'BRANDEN_SCORE,GCS,HR,RR,TEMPERATURE,SBP,DBP,MBP,SaO2,SpO2,Lactate,Oxygen Saturation,pCO2,pH,pO2,Albumin,Bicarbonate,Total Bilirubin,Creatinine,Glucose,Potassium,Sodium,Troponin I,Troponin T,Urea Nitrogen,Hematocrit,Hemoglobin,INR(PT),Neutrophils,Platelet Count,White Blood Cells,sensory_perception,moisture,activity,mobility,nutrition,friction_shear,Position Change,Pressure Reducing Device',
    'age_at_admission,CHF,Arrhy,VALVE,PULMCIRC,PERIVASC,HTN,PARA,NEURO,CHRNLUNG,DM,HYPOTHY,RENLFAIL,LIVER,ULCER,AIDS,LYMPH,METS,TUMOR,ARTH,COAG,OBESE,WGHTLOSS,LYTES,BLDLOSS,ANEMDEF,ALCOHOL,DRUG,PSYCH,DEPRESS,LOT,Gender,Race2,Private Insurance,Public Insurance'
]
