

gpu=0
#ex=SBP,DBP,SaO2,BRANDEN_SCORE,sensory_perception,moisture,activity,mobility,nutrition,friction_shear
ex=SBP,DBP,SaO2,BRANDEN_SCORE
end=9
for ((i=0;i<=${end};i=i+1)); do
    echo --------  $i th iteration
    python src/preprocessing/07_make_train_test.py $i
#    python src/model/pytorch-lstm.py --seed $i --exclude-feature $ex --model-type LSTM --max-epoch 15
    CUDA_VISIBLE_DEVICES=$gpu python src/model/pytorch-lstm.py --seed $i --exclude-feature $ex --model-type MLP --max-epoch 15  
#    python src/model/pytorch-lstm.py --seed $i --exclude-feature $ex --model-type VTonly --max-epoch 10 
#    python src/model/xgb.py --seed $i --exclude-feature $ex --model-type XGB
done

#python src/model/get_folded_results.py XGB
#python src/model/get_folded_results.py VTonly
python src/model/get_folded_results.py MLP
#python src/model/get_folded_results.py LSTM


# debugging
#python src/preprocessing/07_make_train_test.py 0
#python src/model/pytorch-lstm.py --seed 0 --model-type LSTM --max-epoch 15

# feature importance
#python src/preprocessing/07_make_train_test.py 5
#python src/model/pytorch-lstm.py --seed 0 --model-type MLP --max-epoch 1 --exclude-feature $ex
