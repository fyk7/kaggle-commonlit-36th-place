import os
import gc
import logging
import random

import hydra
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import LitDataset
from models import LitModel, ClsTokenAvgModel, create_optimizer
from trainer import train


logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def set_random_seed(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True


@hydra.main(config_name="config", config_path="../conf")
def main(cfg: DictConfig) -> None:
    BASE_DIR = Path(cfg.base_dir)
    SEED = cfg.seed
    NUM_FOLDS = cfg.nfolds
    NUM_EPOCHS = cfg.epochs
    BATCH_SIZE = cfg.batch_size
    MAX_LEN = cfg.max_len
    MODEL_CLASS = cfg.model_class
    MODEL_PATH = cfg.model_path
    TOKENIZER_PATH = cfg.tokenizer_path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    INPUT_DIR = BASE_DIR / "input"
    OUTPUT_DIR = BASE_DIR / os.path.basename(os.getcwd())
    MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
    TRAIN_DATA_PATH = INPUT_DIR / "train.csv"
    FOLD_PATH = INPUT_DIR / "fold_88.csv"


    logger.info("Start to make train_df")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    train_df = train_df.drop(
        train_df[(train_df.target == 0) & (train_df.standard_error == 0)].index
    )
    train_df = train_df.reset_index(drop=True)
    fold_df = pd.read_csv(FOLD_PATH)
    train_df = train_df.merge(fold_df, on="id", how="left")

    logger.info("Start to download tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    logger.info("Start training folds")
    list_val_rmse = []
    for i in range(NUM_FOLDS):
        logger.info(f"\nFold {i + 1}/{NUM_FOLDS}")
        set_random_seed(SEED)

        train_dataset = LitDataset(
            train_df[train_df[cfg.fold_name] != i],
            tokenizer=tokenizer,
            max_length=MAX_LEN,
        )    
        val_dataset = LitDataset(
            train_df[train_df[cfg.fold_name] == i],
            tokenizer=tokenizer,
            max_length=MAX_LEN,
        )    
            
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            drop_last=True,
            shuffle=True,
            num_workers=2
        )    
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle=False,
            num_workers=2
        )    
            
        logger.info("Start to load BERT Model")
        try:
            model = eval(MODEL_CLASS)(
                huggingface_model_path=MODEL_PATH,
                is_lg_model="large" in MODEL_PATH
            ).to(DEVICE)
        except:
            raise(RuntimeError("Model {} not available!".format(MODEL_CLASS)))

        optimizer = create_optimizer(model, cfg)                        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=NUM_EPOCHS * len(train_loader),
            num_warmup_steps=50
        )    
        best_val_rmse = train(
            model=model,
            model_path=MODEL_OUTPUT_DIR / f"model_{i + 1}.pth",
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=DEVICE,
            scheduler=scheduler,
            logger=logger
        )
        list_val_rmse.append(best_val_rmse)
        logger.info(f"\nPerformance estimates:{list_val_rmse}")

        del model
        gc.collect()

if __name__ == "__main__":
    main()
