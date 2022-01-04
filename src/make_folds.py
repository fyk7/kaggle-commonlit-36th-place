import os
import re
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import pandas as pd
from omegaconf.dictconfig import DictConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def mk_fold_by_stratified_url(
    train_df: pd.DataFrame,
    cfg: DictConfig,
    fold_name: str = "Fold_stratified_url",
) -> pd.DataFrame:
    """
    文章の引用元となるurlに対してStratifiedKFold
    localのCVとpublicのCVの相関がよく取れていたので、
    validationにはこちらを採用
    """
    url_regex = re.compile(r'https?://[^/]+/')
    train_df['url_base'] = train_df['url_legal'].apply(
        lambda x: url_regex.search(x).group()
        if isinstance(x, str) else "NaN"
    )
    train_df.loc[train_df["url_base"] == "NaN", "url_base"] = \
        train_df.loc[train_df["url_base"] == "NaN", "id"]
    train_df["url_base_int"] = \
        LabelEncoder().fit_transform(train_df["url_base"])
    train_df[fold_name] = -1
    stratified_url_kfold = StratifiedKFold(
        n_splits=cfg.nfolds,
        shuffle=True,
        random_state=cfg.seed
    )
    for k, (train_idx, valid_idx) in \
        enumerate(stratified_url_kfold.split(X=train_df, y=train_df["url_base_int"])):
        train_df.loc[valid_idx, fold_name] = int(k)

    return train_df[["id", fold_name]]


def mk_fold_by_difficulty_bins(
    train_df: pd.DataFrame,
    cfg: DictConfig,
    fold_name: str = "Fold_diff_bin",
) -> pd.DataFrame:
    """
    文章の読みやすさ(readability)をbinに分割してから、
    そのbinに対してStratifiedKFoldする
    """
    num_bins = int(np.floor(1 + np.log2(len(train_df))))
    train_df['bins'] = pd.cut(
        train_df['target'], bins=num_bins, labels=False)
    bins = train_df['bins'].to_numpy()
    train_df['Fold'] = -1
    stratified_url_kfold = StratifiedKFold(
        n_splits=cfg.nfolds,
        shuffle=True,
        random_state=cfg.seed
    )
    for k , (train_idx, valid_idx) in \
        enumerate(stratified_url_kfold.split(X=train_df, y=bins)):
        train_df.loc[valid_idx, 'Fold'] = k

    return train_df[["id", "Fold"]]


@hydra.main(config_name="config", config_path="../conf")
def main(cfg: DictConfig):
    BASE_DIR = Path(cfg.base_dir)
    INPUT_DIR = BASE_DIR / "input"
    train_df_PATH = INPUT_DIR / "train.csv"
    train_df = pd.read_csv(train_df_PATH)
    fold_name="Fold_stratified_url"
    train_df_with_fold = mk_fold_by_stratified_url(train_df, cfg, fold_name)
    # train_df_with_fold = mk_fold_by_difficulty_bins(train_df, cfg, fold_name)
    train_df_with_fold.to_csv(
        os.path.join(os.getcwd(), f"{fold_name}.csv"), index=False
    )

if __name__ == "__main__":
    main()
