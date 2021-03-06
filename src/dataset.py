import pandas as pd
import torch
from torch.utils.data import Dataset

class LitDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 256,
        inference_only: bool = False,
    ):
        super().__init__()
        self.df = df        
        self.inference_only = inference_only
        self.text = df.excerpt.tolist()
        # self.text = [text.replace("\n", " ") for text in self.text]
        
        if not self.inference_only:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)        
    
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = max_length,
            truncation = True,
            return_attention_mask=True
        )        
 
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        attention_mask = torch.tensor(self.encoded['attention_mask'][index])
        
        if self.inference_only:
            return (input_ids, attention_mask)            
        else:
            target = self.target[index]
            return (input_ids, attention_mask, target)
