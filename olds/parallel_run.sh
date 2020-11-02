gpu=$1
seed=$(( $2+0 ))


# full,wobraden,wostatic,wbraden

model_type=LSTM_full

full="BRANDEN_SCORE,Oxygen Saturation"
nobranden="BRANDEN_SCORE,Oxygen Saturation,sensory_perception,moisture,activity,mobility,nutrition,friction_shear"
nostatic="BRANDEN_SCORE,Oxygen Saturation,age_at_admission,CHF,Arrhy,VALVE,PULMCIRC,PERIVASC,HTN,PARA,NEURO,CHRNLUNG,DM,HYPOTHY,RENLFAIL,LIVER,ULCER,AIDS,LYMPH,METS,TUMOR,ARTH,COAG,OBESE,WGHTLOSS,LYTES,BLDLOSS,ANEMDEF,ALCOHOL,DRUG,PSYCH,DEPRESS,Gender,Race2,Private Insurance,Public Insurance,LOT"
onlybranden="BRANDEN_SCORE,GCS,HR,RR,TEMPERATURE,SBP,DBP,MBP,SaO2,SpO2,Lactate,Oxygen Saturation,pCO2,pH,pO2,Albumin,Bicarbonate,Total Bilirubin,Creatinine,Glucose,Potassium,Sodium,Troponin I,Troponin T,Urea Nitrogen,Hematocrit,Hemoglobin,INR(PT),Neutrophils,Platelet Count,White Blood Cells,Position Change,Pressure Reducing Device,age_at_admission,CHF,Arrhy,VALVE,PULMCIRC,PERIVASC,HTN,PARA,NEURO,CHRNLUNG,DM,HYPOTHY,RENLFAIL,LIVER,ULCER,AIDS,LYMPH,METS,TUMOR,ARTH,COAG,OBESE,WGHTLOSS,LYTES,BLDLOSS,ANEMDEF,ALCOHOL,DRUG,PSYCH,DEPRESS,LOT,Gender,Race2,Private Insurance,Public Insurance"


echo --------  $seed seed
#python src/new_preprocessing/07_make_train_test.py $seed
#CUDA_VISIBLE_DEVICES=$gpu python src/model/pytorch-lstm.py --seed $seed \
#    --dataset-root datasets/seed_${seed} \
#    --exclude-feature "$full" \
#    --feature-importance 1 \
#    --model-type $model_type --max-epoch 8

#python src/model/xgb.py --seed $seed --exclude-feature "$onlybranden" \
#    --model-type $model_type --dataset-root datasets/seed_${seed}

python src/model/get_folded_results.py $model_type


# debugging
#python src/preprocessing/07_make_train_test.py 0
#CUDA_VISIBLE_DEVICES=0 python src/model/pytorch-lstm.py \
#    --seed 0 --model-type MLP --max-epoch 5 --trajectory 1 

# feature importance
#python src/preprocessing/07_make_train_test.py 5
#python src/model/pytorch-lstm.py --seed 0 --model-type MLP --max-epoch 1 --exclude-feature $ex
