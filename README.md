# How to run

# preprocessing
```
# root dir is kohi/
# run following codes sequentially

python src/preprocessing/00_stats_generator.py
python src/preprocessing/01_outlier_removal.py
python src/preprocessing/02_imputation.py
python src/preprocessing/03_averaging.py
python src/preprocessing/04_add_features.py
python src/preprocessing/05_converter.py
python src/preprocessing/06_feature_eng.py
```


# run models
```
python src/model/xgb.py
CUDA_VISIBLE_DEVICES=0 python src/model/lstm.py

```

