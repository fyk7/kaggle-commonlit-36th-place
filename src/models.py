import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoConfig,
    AdamW,
)
from omegaconf import DictConfig


# TODO base系とlarge系で分岐させる
class LitModel(nn.Module):
    def __init__(
        self,
        huggingface_model_path: str,
        is_lg_model: bool = False
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(huggingface_model_path)
        config.update({
            "output_hidden_states":True, 
            "hidden_dropout_prob": 0.0,
            "layer_norm_eps": 1e-7,
        })                       
        self.roberta = AutoModel.from_pretrained(
            huggingface_model_path,
            config=config
        )  
        if is_lg_model:
            self.attention = nn.Sequential(            
                nn.Linear(1024, 512),            
                nn.Tanh(),                       
                nn.Linear(512, 1),
                nn.Softmax(dim=1)
            )        
            self.regressor = nn.Sequential(                        
                nn.Linear(1024, 1)                        
            )
        else:
            self.attention = nn.Sequential(            
                nn.Linear(768, 512),            
                nn.Tanh(),                       
                nn.Linear(512, 1),
                nn.Softmax(dim=1)
            )        
            self.regressor = nn.Sequential(                        
                nn.Linear(768, 1)                        
            )
        # self.dropout = nn.Dropout(0.05)
        
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )        
        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)
        context_vector = \
            torch.sum(weights * last_layer_hidden_states, dim=1)        
        # context_vector = self.dropout(context_vector)
        return self.regressor(context_vector)


class ClsTokenAvgModel(nn.Module):
    def __init__(
        self,
        huggingface_model_path: str,
        start_layer: int = 0,
        is_lg_model: bool = False
    ):
        super().__init__()
        self.start_layer = start_layer
        config = AutoConfig.from_pretrained(huggingface_model_path)
        config.update({
            "output_hidden_states":True, 
            "hidden_dropout_prob": 0.0,
            "layer_norm_eps": 1e-7
        })                       

        self.dropout = nn.Dropout(p=0.20)
        n_weights = config.num_hidden_layers + 1 - self.start_layer
        # n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        # self.init_weights()
        self.roberta = \
            AutoModel.from_pretrained(huggingface_model_path, config=config)  
        self.regressor = nn.Sequential(                        
            nn.Linear(768, 1)                        
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)        
        # outputs[2] and outputs.hidden_states is same.
        hidden_layers = outputs.hidden_states[self.start_layer:]
        # hidden_layers = outputs.hidden_states
        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :])
            for layer in hidden_layers], dim=2
        )
        cls_output = (
            torch.softmax(self.layer_weights, dim=0) * cls_outputs
        ).sum(-1)
        outputs = self.regressor(cls_output)
        return outputs


def create_optimizer(model: nn.Module, cfg: DictConfig):
    named_parameters = list(model.named_parameters())    
    roberta_parameters = named_parameters[:cfg.optimizer.atten_param_start]    
    attention_parameters = named_parameters[
        cfg.optimizer.atten_param_start: cfg.optimizer.regressor_param_start
    ]
    regressor_parameters = named_parameters[cfg.optimizer.regressor_param_start:]
        
    # attentionのparameters
    attention_group = [params for (_, params) in attention_parameters]
    # regressorのparameters
    regressor_group = [params for (_, params) in regressor_parameters]
    parameters = []
    # 以下のパラメタいじる
    parameters.append({"params": attention_group, "lr": eval(cfg.optimizer.atten_lr)})
    parameters.append({"params": regressor_group, "lr": eval(cfg.optimizer.reg_lr)})

    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01
        lr = eval(cfg.optimizer.bert_low_layer_lr)

        # bertの中間layerの学習率を調整する
        if layer_num >= cfg.optimizer.bert_mid_level_layer_start:        
            lr = eval(cfg.optimizer.bert_mid_layer_lr)

        # bertの後半layerの学習率を調整する
        if layer_num >= cfg.optimizer.bert_high_level_layer_start:
            lr = eval(cfg.optimizer.bert_high_layer_lr)

        parameters.append({
            "params": params,
            "weight_decay": weight_decay,
            "lr": lr
        })

    return AdamW(parameters)
