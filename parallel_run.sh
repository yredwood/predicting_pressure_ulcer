gpu=$1
seed=$(( $2+0 ))

#ex=SBP,DBP,SaO2,BRANDEN_SCORE,sensory_perception,moisture,activity,mobility,nutrition,friction_shear
ex=SBP,DBP,SaO2,BRANDEN_SCORE
end=9
    echo --------  $seed seed
#    python src/model/pytorch-lstm.py --seed $i --exclude-feature $ex --model-type LSTM --max-epoch 15
#python src/new_preprocessing/07_make_train_test.py $seed
CUDA_VISIBLE_DEVICES=$gpu python src/model/pytorch-lstm.py --seed $seed \
    --dataset-root datasets/seed_${seed} \
    --exclude-feature $ex --model-type VTonly --max-epoch 10
#        --feature-importance 1
#
#    CUDA_VISIBLE_DEVICES=$gpu python src/model/pytorch-lstm.py --seed $i \
#        --exclude-feature $ex --model-type LSTM --max-epoch 10  \
#        --feature-importance 1

#    python src/model/pytorch-lstm.py --seed $i --exclude-feature $ex --model-type VTonly --max-epoch 10 
#    python src/model/xgb.py --seed $i --exclude-feature $ex --model-type XGB


#python src/model/get_folded_results.py XGB
#python src/model/get_folded_results.py VTonly
#python src/model/get_folded_results.py MLP
python src/model/get_folded_results.py LSTM


# debugging
#python src/preprocessing/07_make_train_test.py 0
#CUDA_VISIBLE_DEVICES=0 python src/model/pytorch-lstm.py \
#    --seed 0 --model-type MLP --max-epoch 5 --trajectory 1 

# feature importance
#python src/preprocessing/07_make_train_test.py 5
#python src/model/pytorch-lstm.py --seed 0 --model-type MLP --max-epoch 1 --exclude-feature $ex
