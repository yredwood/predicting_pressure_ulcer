

end=5
for ((i=0;i<=${end};i=i+1)); do
    python src/preprocessing/07_make_train_test.py $i
#    python src/model/pytorch-lstm.py --seed $i --model-type LSTM --max-epoch 35
    python src/model/pytorch-lstm.py --seed $i --model-type MLP --max-epoch 15
done

#python src/model/get_folded_results.py 1
