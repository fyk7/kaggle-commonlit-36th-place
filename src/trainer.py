import math
import time
import torch
import typing as t
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

# TODO configへ
EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]


def _eval_mse(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
):
    model.eval()            
    mse_sum = 0
    with torch.no_grad():
        for batch_num, (input_ids, attention_mask, target) in enumerate(data_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)                        
            target = target.to(device)           
            pred = model(input_ids, attention_mask)                       
            # mse_sumだからreduction="sum"
            mse_sum += nn.MSELoss(reduction="sum")(pred.flatten(), target).item()
    return mse_sum / len(data_loader.dataset)


def train(
    model: nn.Module,
    model_path: t.Union[Path, str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    device: str = "cpu",
    scheduler=None,
    num_epochs: int = 3,
    logger=None,
) -> float:    
    best_val_rmse = None
    best_epoch = 0
    eval_period = EVAL_SCHEDULE[0][1]    

    step = 0
    last_eval_step = 0
    start = time.time()

    for epoch in range(num_epochs):                           
        val_rmse = None         

        early_stop_counter = 0
        for batch_num, (input_ids, attention_mask, target) in enumerate(train_loader):

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)            
            target = target.to(device)                        

            optimizer.zero_grad()
            model.train()
            pred = model(input_ids, attention_mask)
            mse = nn.MSELoss(reduction="mean")(pred.flatten(), target)
            mse.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()


            # 学習が進むにつれてvalidationの頻度を高くする
            if step >= last_eval_step + eval_period:
                elapsed_seconds = time.time() - start
                num_steps = step - last_eval_step
                logger.info(f"\n{num_steps} steps took {elapsed_seconds:0.3} seconds")
                last_eval_step = step
                
                val_rmse = math.sqrt(_eval_mse(model, val_loader))                            
                logger.info(
                    f"Epoch: {epoch} batch_num: {batch_num}", 
                    f"val_rmse: {val_rmse:0.4}"
                )

                for rmse, period in EVAL_SCHEDULE:
                    if val_rmse >= rmse:
                        eval_period = period
                        break                               
                
                # val_rmseがbest_val_rmseを超えた場合のみmodelを保存
                if (not best_val_rmse) or (val_rmse < best_val_rmse):                    
                    best_val_rmse = val_rmse
                    best_epoch = epoch
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"New best_val_rmse: {best_val_rmse:0.4}")
                    early_stop_counter = 0
                else:       
                    logger.info(
                        f"Still best_val_rmse: {best_val_rmse:0.4}",
                        f"(from epoch {best_epoch})"
                    )                                    
                    early_stop_counter += 1
                start = time.time()

            step += 1

            if epoch >= 2 and early_stop_counter >= 20:
                logger.info("early_stop_counterが所定の回数を超えました!")
                break
                        
    return best_val_rmse
