import torch
import sys
sys.path.append("../")
import torch
import torch.nn as nn
import numpy as np
import copy
from torch.optim import SGD, AdamW
from transformers import get_linear_schedule_with_warmup
import os
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict


class FedAlg():
    def __init__(self, dls, client_weights, lrs, max_epoches, aggregation_freq, device, saved_dir, model_name, **args):
        self.saved_dir = saved_dir
        self.dls = dls
        self.client_weights = client_weights
        self.lrs = lrs
        self.max_epoches = max_epoches
        self.aggregation_freq = aggregation_freq
        self.device = device
        self.client_num = len(client_weights)
        self.defining_metric = "f1-score"
        self.eval_metrics = ['loss', 'precision', 'recall', 'f1-score']
        self.model_name=model_name
        self.num_labels = num_labels
        self.args = args
        
        
        ## setup models and training configs
        self.server_model = self.generate_models().to(device)
        self.client_models = [copy.deepcopy(self.server_model).to(device) for i in range(self.client_num)]
        self.optimizers = [AdamW(params=self.client_models[idx].parameters(), lr=self.lrs[idx]) for idx in range(self.client_num)]
        self.schedulers = [get_linear_schedule_with_warmup(self.optimizers[idx], 
                                            num_warmup_steps = 0,
                                            num_training_steps = self.max_epoches*len(dls[idx]['train'])) for idx in range(self.client_num)]
        # self.optimizers = [AdamW(params=client_models[idx].parameters(), lr=lrs[i], weight_decay=1e-5) for idx in range(client_num)]
    
    def generate_models(self):
        pass
    
    def local_train(self, idx):
        ## access trainloader self.dls[idx]['train']
        pass
    
    def local_validate(self, idx):
        ## access trainloader self.dls[idx]['validation']
        pass
    
    def global_validate(self):
        ## access trainloader self.dls[idx]['validation']
        pass
    
    def save_models(self, file_name="best.pt"):
        for client_idx in range(self.client_num):
            torch.save(self.client_models[client_idx].state_dict(), f"./{self.saved_dir}/site-{client_idx+1}/{file_name}")
        torch.save(self.server_model.state_dict(), f"./{self.saved_dir}/global/{file_name}")
    
    def communication(self, server_model, models, not_update_client=False, bn_exclude=False):
        with torch.no_grad():
            # aggregate params
            if bn_exclude:
                for key in server_model.state_dict().keys():
                    if 'norm' not in key:
                        temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                        # temp = torch.zeros_like(server_model.state_dict()[key], dtype=type(server_model.state_dict()[key]))
                        for client_idx in range(len(self.client_weights)):
                            temp += self.client_weights[client_idx] * models[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                        # if not not_update_client:
                        for client_idx in range(len(self.client_weights)):
                            models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
            else:
                for key in server_model.state_dict().keys():
                    # num_batches_tracked is a non trainable LongTensor and
                    # num_batches_tracked are the same for all clients for the given datasets
                    #if 'num_batches_tracked' in key or (args.conv_only and 'classifier' in key):
                    if 'num_batches_tracked' in key:
                        server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                    else:
                        temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                        # temp = torch.zeros_like(server_model.state_dict()[key], dtype=type(server_model.state_dict()[key]))
                        for client_idx in range(len(self.client_weights)):
                            temp += self.client_weights[client_idx] * models[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                        if not not_update_client:
                            for client_idx in range(len(self.client_weights)):
                                models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        return server_model, models

    
    def fit(self):
        os.makedirs(self.saved_dir, exist_ok=True)
        for i in range(self.client_num):
            os.makedirs(os.path.join(self.saved_dir, f'site-{i+1}'), exist_ok=True)
            os.makedirs(os.path.join(self.saved_dir, f'global'), exist_ok=True)
            
        writer = SummaryWriter(log_dir=f"{self.saved_dir}/tb_events/")
        best_score = 0
        for epoch in range(self.max_epoches):
            # local update
            for client_idx in range(self.client_num):
                # local_train(model, dls[client_idx], optimizers[client_idx], device)
                self.local_train(client_idx)
                
            # aggregation & save best model
            with torch.no_grad():

                # validation
                val_metrics = defaultdict(lambda: {})
                
                for client_idx in range(self.client_num):
                    print(f"{epoch}/{self.max_epoches}:========================== local validation client {client_idx} ========================")
                    val_metrics[f"site-{client_idx+1}"] = self.local_validate(client_idx)
                print(f"{epoch}/{self.max_epoches}:========================== global validation ========================")
                val_metrics['global'] = self.global_validate()
                
                for metric in self.eval_metrics:
                    tmp_dict_train, tmp_dict_validation = {}, {}
                    for client_idx in range(self.client_num):
                        tmp_dict_train[f"site-{client_idx+1}"] = val_metrics[f"site-{client_idx+1}"]['train']['macro avg'][metric]
                        tmp_dict_validation[f"site-{client_idx+1}"] = val_metrics[f"site-{client_idx+1}"]['val']['macro avg'][metric]
                
                    writer.add_scalars(f'{metric}/train', tmp_dict_train, epoch)
                    writer.add_scalars(f'{metric}/val', tmp_dict_validation, epoch)
                    
                if not (epoch % self.aggregation_freq):
                    self.server_model, self.client_models = self.communication(self.server_model, self.client_models)
                
                
            ## checkpoint the best performance based on macro avg f1-score
            if val_metrics['global']['val']['macro avg'][self.defining_metric] > best_score:
                best_score = val_metrics['global']['val']['macro avg'][self.defining_metric]
                print("update best model")
                self.save_models(file_name="best.pt")
    
        ## save final model
        self.save_models(file_name="final.pt")