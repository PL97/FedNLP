from torch.utils.data import Dataset, DataLoader
import torch

import copy
import os



import sys

from fed_algo.fedalg import FedAlg
from models.BERT import BertModel
from collections import defaultdict


class trainer_base:
    def __init__(self):
        self.model = None
        self.trainloader = None
        self.valloader = None
        self.device = None
        self.idx_to_label = None
        self.epochs = None
        self.lr = None
    
        ## define solver
        self.optimizer = None
        self.scheduler = None

        self.writer = None
        self.saved_dir = None

        self.model = self.model.to(self.device)
        os.makedirs(self.saved_dir, exist_ok=True)
    
    def fit(self):
        best_f1 = 0

        for epoch_num in range(self.epochs):
            
            ## train the model and validate the performance
            self.train_step()
            print(f"{epoch_num}/{self.epochs}:========================== Train ========================")
            train_metrics = self.validate(self.trainloader, prefix='train')
            print(f"{epoch_num}/{self.epochs}:========================== Validation ========================")
            val_metrics = self.validate(self.valloader, prefix='val')
            
            ## write metric to tensorboard
            for metric in ['loss', 'precision', 'recall', 'f1-score']:
                self.writer.add_scalars(f'{metric}', {
                    "train": train_metrics['macro avg'][metric],
                    "validation":  val_metrics['macro avg'][metric],
                    }, epoch_num)
                
                
            ## checkpoint the best performance based on macro avg f1-score
            if val_metrics['macro avg']['f1-score'] > best_f1:
                best_f1 = val_metrics['macro avg']['f1-score']
                print("update best model")
                torch.save(self.model.state_dict(), f"./{self.saved_dir}/best.pt")
        
        ## save final model
        torch.save(self.model.state_dict(), f"./{self.saved_dir}/final.pt")
        return self.model


####################################### federated learning ##########################################

class NER_FedAvg_base(FedAlg):
    def generate_models(self):
        return BertModel(num_labels = 19, model_name=self.model_name)
    
    def train_by_epoch(self):
        pass
    
    def validate(self):
        pass
    
    def local_train(self, idx):
        ## access trainloader self.dls[idx]['train']
        ## access model self.client_models[idx]
        self.train_by_epoch(idx)
    
    def local_validate(self, idx):
        ## access trainloader self.dls[idx]['validation']
        ## access local model self.client_models[idx]
        ## access global model self.server_model[idx]
        return self.validate(idx)
    
    def global_validate(self):
        ## access trainloader self.dls[idx]['validation']
        label_map = self.dls[0]['train'].dataset.ids_to_labels
        ret_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        for client_idx in range(self.client_num):
            tmp_train = self.validate(self.server_model, self.dls[client_idx]['train'], self.device, label_map)
            tmp_validation = self.validate(self.server_model, self.dls[client_idx]['validation'], self.device, label_map)
            for k, v in tmp_train.items():
                for kk, vv in v.items():
                    ret_dict['train'][k][kk] += tmp_train[k][kk]
                    ret_dict['validation'][k][kk] += tmp_validation[k][kk]
                    

        ## aggregate results
        ### can use key "validation" also
        for k, v, in ret_dict['train'].items():
            for kk, vv in v.items():
                ret_dict['train'][k][kk] /= self.client_num
                ret_dict['validation'][k][kk] /= self.client_num
                
        return ret_dict
    

    def communication(self, server_model, models, not_update_client=False, bn_exclude=False):
        with torch.no_grad():
            # aggregate params
            for key in server_model.state_dict().keys():
                if 'norm' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(self.client_num):
                        temp += self.client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    # if not not_update_client:
                    for client_idx in range(len(self.client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        return server_model, models
        
        