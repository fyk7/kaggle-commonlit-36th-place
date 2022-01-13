import gc
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModel, AutoTokenizer, AutoConfig)

train_df = pd.read_csv("../input/commonlitreadabilityprize/train.csv")
test_df = pd.read_csv("../input/commonlitreadabilityprize/test.csv")
sub_df = pd.read_csv("../input/commonlitreadabilityprize/sample_submission.csv")

NUM_FOLDS = 5
BATCH_SIZE = 8
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict(model, data_loader):
    model.eval()
    result = np.zeros(len(data_loader.dataset))    
    index = 0
    with torch.no_grad():
        for batch_num, (input_ids, attention_mask) in enumerate(data_loader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)           
            pred = model(input_ids, attention_mask)                        
            result[index : index + pred.shape[0]] = pred.flatten().to("cpu")
            index += pred.shape[0]
    return result



BATCH_SIZE=16
BASE_MODEL_PATH = "../input/roberta-base"
FT_MODEL_PATH = "../input/commonlit-roberta-0467"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

class LitDataset(Dataset):
    def __init__(self, df, infer=False):
        super().__init__()
        self.df = df        
        self.infer = infer
        self.text = df.excerpt.tolist()
        if not self.infer:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)        
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
 
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        atten_mask = torch.tensor(self.encoded['attention_mask'][index])
        
        if self.infer:
            return (input_ids, atten_mask)            
        else:
            target = self.target[index]
            return (input_ids, atten_mask, target)

class LitModelLarge1(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       
        
        self.roberta = AutoModel.from_pretrained(BASE_MODEL_PATH, config=config)
        self.attention = nn.Sequential(            
            nn.Linear(768, 512),               
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        ) 
        self.regressor = nn.Sequential(                        
            nn.Linear(768, 1)                     
        )
        
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask) 
        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)
        return self.regressor(context_vector)
    
test_dataset = LitDataset(test_df, infer=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=2)
preds = np.zeros((NUM_FOLDS, len(test_dataset)))

for idx in range(NUM_FOLDS):
    # アンダースコアに注意!!!!!!!!
    model_path = f"{FT_MODEL_PATH}/model_{idx+1}.pth"
    print(model_path)
    model = LitModelLarge1()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    preds[idx] = predict(model, test_loader)
    del model;gc.collect()

del test_dataset, test_loader, tokenizer;gc.collect()
rbt_bs_prtr_pred = preds.mean(axis=0)



BATCH_SIZE=16
BASE_MODEL_PATH = "../input/roberta-base"
FT_MODEL_PATH = "../input/rbt-bs-strat-url-16-256"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

class LitDataset(Dataset):
    def __init__(self, df, infer=False):
        super().__init__()
        self.df = df        
        self.infer = infer
        self.text = df.excerpt.tolist()
        if not self.infer:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)        
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        atten_mask = torch.tensor(self.encoded['attention_mask'][index])
        
        if self.infer:
            return (input_ids, atten_mask)            
        else:
            target = self.target[index]
            return (input_ids, atten_mask, target)

class LitModelLarge1(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       
        
        self.roberta = AutoModel.from_pretrained(BASE_MODEL_PATH, config=config)
        self.attention = nn.Sequential(            
            nn.Linear(768, 512),               
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        ) 
        self.regressor = nn.Sequential(                        
            nn.Linear(768, 1)                     
        )
        
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask) 
        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)
        return self.regressor(context_vector)
    
test_dataset = LitDataset(test_df, infer=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=2)
preds = np.zeros((NUM_FOLDS, len(test_dataset)))

for idx in range(NUM_FOLDS):
    model_path = f"{FT_MODEL_PATH}/model-{idx+1}.pth"
    print(model_path)
    model = LitModelLarge1()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    preds[idx] = predict(model, test_loader)
    del model;gc.collect()

del test_dataset, test_loader, tokenizer;gc.collect()
rbt_bs_strat_url_pred = preds.mean(axis=0)



BATCH_SIZE=16
BASE_MODEL_PATH = "../input/microsoft-deberta-bas"
FT_MODEL_PATH = "../input/dbt-bs-strat-url-16-256-try2"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

class LitDataset(Dataset):
    def __init__(self, df, infer=False):
        super().__init__()
        self.df = df        
        self.infer = infer
        self.text = df.excerpt.tolist()
        if not self.infer:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)        
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        atten_mask = torch.tensor(self.encoded['attention_mask'][index])
        
        if self.infer:
            return (input_ids, atten_mask)            
        else:
            target = self.target[index]
            return (input_ids, atten_mask, target)

class LitModelLarge1(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       
        
        self.roberta = AutoModel.from_pretrained(BASE_MODEL_PATH, config=config)
        self.attention = nn.Sequential(            
            nn.Linear(768, 512),               
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        ) 
        self.regressor = nn.Sequential(                        
            nn.Linear(768, 1)                     
        )
        
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask) 
        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)
        return self.regressor(context_vector)
    
test_dataset = LitDataset(test_df, infer=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=2)
preds = np.zeros((NUM_FOLDS, len(test_dataset)))

for idx in range(NUM_FOLDS):
    model_path = f"{FT_MODEL_PATH}/model-{idx+1}.pth"
    print(model_path)
    model = LitModelLarge1()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    preds[idx] = predict(model, test_loader)
    del model;gc.collect()

del test_dataset, test_loader, tokenizer;gc.collect()
dbt_bs_strat_url_pred = preds.mean(axis=0)



BATCH_SIZE=16
BASE_MODEL_PATH = "../input/google-electra-base-discriminator"
FT_MODEL_PATH = "../input/eltr-bs-strat-url-16-256"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

class LitDataset(Dataset):
    def __init__(self, df, infer=False):
        super().__init__()
        self.df = df        
        self.infer = infer
        self.text = df.excerpt.tolist()
        if not self.infer:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)        
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        atten_mask = torch.tensor(self.encoded['attention_mask'][index])
        
        if self.infer:
            return (input_ids, atten_mask)            
        else:
            target = self.target[index]
            return (input_ids, atten_mask, target)

class LitModelLarge1(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       
        
        self.roberta = AutoModel.from_pretrained(BASE_MODEL_PATH, config=config)
        self.attention = nn.Sequential(            
            nn.Linear(768, 512),               
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        ) 
        self.regressor = nn.Sequential(                        
            nn.Linear(768, 1)                     
        )
        
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask) 
        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)
        return self.regressor(context_vector)
    
test_dataset = LitDataset(test_df, infer=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=2)
preds = np.zeros((NUM_FOLDS, len(test_dataset)))

for idx in range(NUM_FOLDS):
    model_path = f"{FT_MODEL_PATH}/model-{idx+1}.pth"
    print(model_path)
    model = LitModelLarge1()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    preds[idx] = predict(model, test_loader)
    del model;gc.collect()

del test_dataset, test_loader, tokenizer;gc.collect()
eltr_bs_strat_url_pred = preds.mean(axis=0)



BATCH_SIZE=16
BASE_MODEL_PATH = "../input/roberta-base"
FT_MODEL_PATH = "../input/rbt-bs-cta-all-url-16-256"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

class LitDataset(Dataset):
    def __init__(self, df, infer=False):
        super().__init__()
        self.df = df        
        self.infer = infer
        self.text = df.excerpt.tolist()
        if not self.infer:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)        
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        atten_mask = torch.tensor(self.encoded['attention_mask'][index])
        
        if self.infer:
            return (input_ids, atten_mask)            
        else:
            target = self.target[index]
            return (input_ids, atten_mask, target)

class ClsTokenAvgBaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       

        self.dropout = nn.Dropout(p=0.20)
        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        self.roberta = AutoModel.from_pretrained(BASE_MODEL_PATH, config=config)  
        self.regressor = nn.Sequential(                        
            nn.Linear(768, 1)                        
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask) 
        hidden_layers = outputs.hidden_states
        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2
        )
        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)
        outputs = self.regressor(cls_output)
        return outputs

    
test_dataset = LitDataset(test_df, infer=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=2)
preds = np.zeros((NUM_FOLDS, len(test_dataset)))

for idx in range(NUM_FOLDS):
    model_path = f"{FT_MODEL_PATH}/model-{idx+1}.pth"
    print(model_path)
    model = ClsTokenAvgBaseModel()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    preds[idx] = predict(model, test_loader)
    del model;gc.collect()

del test_dataset, test_loader, tokenizer;gc.collect()
rbt_bs_cta_strat_url_pred = preds.mean(axis=0)



#########  注意  dbt-bsはeltrだった

BATCH_SIZE=16
BASE_MODEL_PATH = "../input/google-electra-base-discriminator"
FT_MODEL_PATH = "../input/dbt-bs-cta-all-strat-url-16-25"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

class LitDataset(Dataset):
    def __init__(self, df, infer=False):
        super().__init__()
        self.df = df        
        self.infer = infer
        self.text = df.excerpt.tolist()
        if not self.infer:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)        
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        atten_mask = torch.tensor(self.encoded['attention_mask'][index])
        
        if self.infer:
            return (input_ids, atten_mask)            
        else:
            target = self.target[index]
            return (input_ids, atten_mask, target)

class ClsTokenAvgBaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       

        self.dropout = nn.Dropout(p=0.20)
        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        self.roberta = AutoModel.from_pretrained(BASE_MODEL_PATH, config=config)  
        self.regressor = nn.Sequential(                        
            nn.Linear(768, 1)                        
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask) 
        hidden_layers = outputs.hidden_states
        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2
        )
        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)
        outputs = self.regressor(cls_output)
        return outputs

    
test_dataset = LitDataset(test_df, infer=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=2)
preds = np.zeros((NUM_FOLDS, len(test_dataset)))

for idx in range(NUM_FOLDS):
    model_path = f"{FT_MODEL_PATH}/model-{idx+1}.pth"
    print(model_path)
    model = ClsTokenAvgBaseModel()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    preds[idx] = predict(model, test_loader)
    del model;gc.collect()

del test_dataset, test_loader, tokenizer;gc.collect()
eltr_bs_cta_strat_url_pred = preds.mean(axis=0)



BASE_MODEL_PATH = "../input/robertalarge"
FT_MODEL_PATH = "../input/rbt-lg-lit-strati-url-b8-l256"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

class LitDataset(Dataset):
    def __init__(self, df, infer=False):
        super().__init__()
        self.df = df        
        self.infer = infer
        self.text = df.excerpt.tolist()
        
        if not self.infer:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)        
    
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
 
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        atten_mask = torch.tensor(self.encoded['attention_mask'][index])
        
        if self.infer:
            return (input_ids, atten_mask)            
        else:
            target = self.target[index]
            return (input_ids, atten_mask, target)

class LitModelLarge1(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       
        
        self.roberta = AutoModel.from_pretrained(BASE_MODEL_PATH, config=config)
        self.attention = nn.Sequential(            
            nn.Linear(1024, 512),               
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        ) 
        self.regressor = nn.Sequential(                        
            nn.Linear(1024, 1)                     
        )
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask) 
        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)   
        # context_vector = self.dropout(context_vector)
        return self.regressor(context_vector)
    
BATCH_SIZE=8
test_dataset = LitDataset(test_df, infer=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=2)
preds = np.zeros((NUM_FOLDS, len(test_dataset)))

for idx in range(NUM_FOLDS):
    model_path = f"{FT_MODEL_PATH}/model-{idx+1}.pth"
    print(model_path)
    model = LitModelLarge1()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    preds[idx] = predict(model, test_loader)
    del model;gc.collect()

del test_dataset, test_loader, tokenizer;gc.collect()
rbt_lg_strat_url_pred = preds.mean(axis=0)



BASE_MODEL_PATH = "../input/robertalarge"
FT_MODEL_PATH = "../input/rbt-lg-strat-url-v2-8-256"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

class LitDataset(Dataset):
    def __init__(self, df, infer=False):
        super().__init__()
        self.df = df        
        self.infer = infer
        self.text = df.excerpt.tolist()
        
        if not self.infer:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)        
    
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
 
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        atten_mask = torch.tensor(self.encoded['attention_mask'][index])
        
        if self.infer:
            return (input_ids, atten_mask)            
        else:
            target = self.target[index]
            return (input_ids, atten_mask, target)

class LitModelLarge1(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       
        
        self.roberta = AutoModel.from_pretrained(BASE_MODEL_PATH, config=config)
        self.attention = nn.Sequential(            
            nn.Linear(1024, 512),               
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        ) 
        self.regressor = nn.Sequential(                        
            nn.Linear(1024, 1)                     
        )
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask) 
        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)   
        # context_vector = self.dropout(context_vector)
        return self.regressor(context_vector)
    
BATCH_SIZE=8
test_dataset = LitDataset(test_df, infer=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=2)
preds = np.zeros((NUM_FOLDS, len(test_dataset)))

for idx in range(NUM_FOLDS):
    ####################   注意　-バーが二つ必要
    model_path = f"{FT_MODEL_PATH}/model--{idx+1}.pth"
    print(model_path)
    model = LitModelLarge1()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    preds[idx] = predict(model, test_loader)
    del model;gc.collect()

del test_dataset, test_loader, tokenizer;gc.collect()
rbt_lg_strat_url_v2_pred = preds.mean(axis=0)




BASE_MODEL_PATH = "../input/electra-large-discriminator"
FT_MODEL_PATH = "../input/eltr-lg-atten-512-8-256"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

class LitDataset(Dataset):
    def __init__(self, df, infer=False):
        super().__init__()
        self.df = df        
        self.infer = infer
        self.text = df.excerpt.tolist()  
        if not self.infer:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
 
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        atten_mask = torch.tensor(self.encoded['attention_mask'][index])
        
        if self.infer:
            return (input_ids, atten_mask)            
        else:
            target = self.target[index]
            return (input_ids, atten_mask, target)
        
class ElectraLarge(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       
        
        self.roberta = AutoModel.from_pretrained(BASE_MODEL_PATH, config=config)
        self.attention = nn.Sequential(            
            nn.Linear(1024, 512),               
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )        
        self.regressor = nn.Sequential(                        
            nn.Linear(1024, 1)                     
        )
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask) 
        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)  
        return self.regressor(context_vector)
    
BATCH_SIZE=8
test_dataset = LitDataset(test_df, infer=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=2)
preds = np.zeros((NUM_FOLDS, len(test_dataset)))

for idx in range(NUM_FOLDS):
    model_path = f"{FT_MODEL_PATH}/model-{idx+1}.pth"
    print(model_path)
    model = ElectraLarge()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    preds[idx] = predict(model, test_loader)
    del model;gc.collect()
del test_dataset, test_loader, tokenizer;gc.collect()
eltr_lg_strat_url_pred = preds.mean(axis=0)

BASE_MODEL_PATH = "../input/microsoft-deberta-large-mnli"
FT_MODEL_PATH = "../input/dbt-lg-true-strat-url-lr122-4-256"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

class LitDataset(Dataset):
    def __init__(self, df, infer=False):
        super().__init__()
        self.df = df        
        self.infer = infer
        self.text = df.excerpt.tolist()
        if not self.infer:
            self.target = torch.tensor(df.target.values, dtype=torch.float32) 
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
 
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        atten_mask = torch.tensor(self.encoded['attention_mask'][index])
        if self.infer:
            return (input_ids, atten_mask)            
        else:
            target = self.target[index]
            return (input_ids, atten_mask, target)

class DebertaModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})   
        self.roberta = AutoModel.from_pretrained(BASE_MODEL_PATH, config=config)
        self.attention = nn.Sequential(          
            nn.Linear(1024, 512),            
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )  
        self.regressor = nn.Sequential(                        
            nn.Linear(1024, 1)                        
        )
        
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)        

        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)       
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)
        return self.regressor(context_vector)
    
########## バッチサイズに要注意!!!!!
BATCH_SIZE=4
test_dataset = LitDataset(test_df, infer=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=2)
preds = np.zeros((NUM_FOLDS, len(test_dataset)))

for idx in range(NUM_FOLDS):
    model_path = f"{FT_MODEL_PATH}/model-{idx+1}.pth"
    print(model_path)
    model = DebertaModel()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    preds[idx] = predict(model, test_loader)
    del model;gc.collect()

del test_dataset, test_loader, tokenizer;gc.collect()
dbt_lg_strat_url_pred = preds.mean(axis=0)

BASE_MODEL_PATH = "../input/microsoft-deberta-large-mnli"
# FT_MODEL_PATH = "../input/dbt-lg-true-strat-url-lr122-4-256"
FT_MODEL_PATH = "../input/dbt-lg-straturl-v2-4-256"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

class LitDataset(Dataset):
    def __init__(self, df, infer=False):
        super().__init__()
        self.df = df        
        self.infer = infer
        self.text = df.excerpt.tolist()
        if not self.infer:
            self.target = torch.tensor(df.target.values, dtype=torch.float32) 
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
 
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        atten_mask = torch.tensor(self.encoded['attention_mask'][index])
        if self.infer:
            return (input_ids, atten_mask)            
        else:
            target = self.target[index]
            return (input_ids, atten_mask, target)

class DebertaModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})   
        self.roberta = AutoModel.from_pretrained(BASE_MODEL_PATH, config=config)
        self.attention = nn.Sequential(          
            nn.Linear(1024, 512),            
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )  
        self.regressor = nn.Sequential(                        
            nn.Linear(1024, 1)                        
        )
        
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)        

        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)       
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)
        return self.regressor(context_vector)
    
########## バッチサイズに要注意!!!!!
BATCH_SIZE=4
test_dataset = LitDataset(test_df, infer=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=2)
preds = np.zeros((NUM_FOLDS, len(test_dataset)))

for idx in range(NUM_FOLDS):
    model_path = f"{FT_MODEL_PATH}/model-{idx+1}.pth"
    print(model_path)
    model = DebertaModel()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    preds[idx] = predict(model, test_loader)
    del model;gc.collect()

del test_dataset, test_loader, tokenizer;gc.collect()
dbt_lg_strat_url_v2_pred = preds.mean(axis=0)

BASE_MODEL_PATH = "../input/robertalarge"
FT_MODEL_PATH = "../input/cta-lg-after18-8-256"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

class LitDataset(Dataset):
    def __init__(self, df, infer=False):
        super().__init__()
        self.df = df        
        self.infer = infer
        self.text = df.excerpt.tolist()
        if not self.infer:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)        
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        ) 
 
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        atten_mask = torch.tensor(self.encoded['attention_mask'][index])
        if self.infer:
            return (input_ids, atten_mask)            
        else:
            target = self.target[index]
            return (input_ids, atten_mask, target)

class ClsTokenAvgModel2(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       

        self.dropout = nn.Dropout(p=0.20)

        n_weights = config.num_hidden_layers + 1 - 18
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        # self.init_weights()
        self.roberta = AutoModel.from_pretrained(BASE_MODEL_PATH, config=config)  
        self.regressor = nn.Sequential(                        
            nn.Linear(1024, 1)                        
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)        
        hidden_layers = outputs[2][18:]
        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2
        )
        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)
        logits = self.regressor(cls_output)
        outputs = logits
        return outputs
    
BATCH_SIZE=8
test_dataset = LitDataset(test_df, infer=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=2)
preds = np.zeros((NUM_FOLDS, len(test_dataset)))

for idx in range(NUM_FOLDS):
    model_path = f"{FT_MODEL_PATH}/model-{idx+1}.pth"
    print(model_path)
    model = ClsTokenAvgModel2()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    preds[idx] = predict(model, test_loader)
    del model;gc.collect()

del test_dataset, test_loader, tokenizer;gc.collect()
cta_lg_strat_url_pred = preds.mean(axis=0)

# 0.23922778 0.40651899 0.2204791  0.09009821 0.04367593

sub_df["target"] = \
(
    (rbt_bs_prtr_pred * 0.9 + rbt_bs_strat_url_pred * 0.1) * 0.8 +
    dbt_bs_strat_url_pred * 0.08 +
    eltr_bs_strat_url_pred * 0.08 +
    rbt_bs_cta_strat_url_pred * 0.02 +
    eltr_bs_cta_strat_url_pred * 0.02
) * 0.16 + \
(
    (rbt_lg_strat_url_pred * 0.9 + rbt_lg_strat_url_v2_pred * 0.1) * 0.25 +
    eltr_lg_strat_url_pred * 0.25 +
    (dbt_lg_strat_url_pred * 0.1 + dbt_lg_strat_url_v2_pred * 0.9) * 0.40 +
    cta_lg_strat_url_pred * 0.1
) * 0.84

sub_df.to_csv("submission.csv", index=False)
