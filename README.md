# kaggle-commonlit-36th-place

## Features
### Overview
- Competition Overview
  - Build algorithms to rate the complexity of reading passages.
- Metrics
  - RMSE

### FrameWorks and Libraries
- Pytorch
- Huggingface Transformer
- Hydra
- Optuna

### EDA
- [EDA notebook]()

### Model
- Architectures
  1. [Apply attention to last output layers of BERT](https://github.com/fyk7/kaggle-commonlit-36th-place/blob/main/src/models.py#L12)
  2. [Average class tokens (last n layers) of BERT hidden layers](https://github.com/fyk7/kaggle-commonlit-36th-place/blob/main/src/models.py#L12)
- Base Models
  1. Roberta
  2. Deberta
  3. Electra

### Validation Strategy
1. [StratifiedKFold by readability bins](https://github.com/fyk7/kaggle-commonlit-36th-place/blob/main/src/make_folds.py#L236)
2. [StratifiedKFold by reference urls](https://github.com/fyk7/kaggle-commonlit-36th-place/blob/main/src/make_folds.py#L204) <- Main Validation Strategy 
<br>(Hight correlation between local cv and public cv)
3. [StratifiedKFold by readability bins and group by reference urls](https://github.com/fyk7/kaggle-commonlit-36th-place/blob/main/src/make_folds.py#L261) ([StratifiedGroupKFold]())

### Parameter Management
- Hydra
  - [Config files](https://github.com/fyk7/kaggle-commonlit-36th-place/blob/main/conf/config.yaml)
  - [Run scripts example](https://github.com/fyk7/kaggle-commonlit-36th-place/blob/main/scripts/roberta_large.sh)
  - [Inject config dict to entry point](https://github.com/fyk7/kaggle-commonlit-36th-place/blob/main/src/main.py#L35)

### Ensemble
- Optuna
  - [Weighted Average](https://github.com/fyk7/kaggle-commonlit-36th-place/blob/main/src/blending.py)

Tree architecture
<pre>
├── README.md
├── conf
│   ├── config.yaml
│   └── optimizer
│       ├── base.yaml
│       └── large.yaml
├── input
│   ├── fold.csv
│   ├── sample_submission.csv
│   ├── test.csv
│   ├── train.csv
│   └── train_toy.csv
├── notebook
│   └── CLRP_EDA.ipynb
├── output
│   ├── commonlit_blending_material_bs
│   ├── commonlit_blending_material_lg
│   ├── main
│   │   ├── batch_size=16,model_class=LitModel, ...
│   │   ├── ...
├── requirements.txt
├── scripts
│   ├── roberta_base.sh
│   ├── deberta_large.sh
│   ├── clstokenavg_roberta_large.sh
│   ├── ...
└── src
    ├── blending.py
    ├── dataset.py
    ├── final_submission.py
    ├── main.py     <----- Entry point
    ├── make_folds.py
    ├── models.py
    ├── predictor.py
    └── trainer.py
</pre>
