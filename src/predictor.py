import numpy as np
import torch


def predict(model, data_loader, device):
    model.eval()
    result = np.zeros(len(data_loader.dataset))    
    index = 0
    with torch.no_grad():
        for batch_num, (input_ids, attention_mask) in enumerate(data_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)           
            # predは2次元であるからflattenする。[[], [], []] -> [, , ]
            pred = model(input_ids, attention_mask)                        
            result[index : index + pred.shape[0]] = pred.flatten().to("cpu")
            index += pred.shape[0]
    return result
