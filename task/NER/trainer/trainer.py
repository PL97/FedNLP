from torch.optim import SGD, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import copy
import os
from torch.utils.tensorboard import SummaryWriter

from seqeval.metrics import classification_report, f1_score
import sys
from utils.parse_metric_summary import parse_summary
from fed_algo.fedalg import FedAlg
from models.BERT import BertModel
from collections import defaultdict

def train_by_epoch(model, dl, optimizer, device, scheduler):
    model = model.to(device)
    model.train()

    for train_data, train_label in tqdm(dl):
        
        train_label = train_label.to(device)
        mask = train_data['attention_mask'].squeeze(1).to(device)
        input_id = train_data['input_ids'].squeeze(1).to(device)

        optimizer.zero_grad()
        loss, _ = model(input_id, mask, train_label)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) ## optional

        loss.backward()
        optimizer.step()
        scheduler.step()
    
    return model

def validate(model, dataloader, device, label_map):
    model = model.to(device)
    model.eval()

    total_acc_val, total_loss_val, val_total = 0, 0, 0
    val_y_pred, val_y_true = [], []
    
    for val_data, val_label in dataloader:

        val_label = val_label.to(device)
        val_total += val_label.shape[0]
        mask = val_data['attention_mask'].squeeze(1).to(device)
        input_id = val_data['input_ids'].squeeze(1).to(device)

        loss, logits = model(input_id, mask, val_label)
        
        for i in range(logits.shape[0]):
            ## remove pad tokens
            logits_clean = logits[i][val_label[i] != -100]
            label_clean = val_label[i][val_label[i] != -100]

            ## calcluate acc and store prediciton and true labels
            predictions = logits_clean.argmax(dim=1)
            acc = (predictions == label_clean).float().mean()
            total_acc_val += acc.item()
            total_loss_val += loss.item()
            val_y_pred.append([label_map[x.item()] for x in predictions])
            val_y_true.append([label_map[x.item()] for x in label_clean])
    
    ## prepare metric summary   
    summary = classification_report(val_y_true, val_y_pred)
    print("validation: ", summary)
    metric_dict = parse_summary(summary)
    metric_dict['macro avg']['loss'] = total_loss_val/val_total
    metric_dict['macro avg']['acc'] = total_acc_val/val_total
    return metric_dict
    

def train_loop(model, train_dataloader, val_dataloader, saved_dir, **args):
    
    os.makedirs(saved_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=f"{saved_dir}/tb_events/")

    LEARNING_RATE = args['LEARNING_RATE']
    EPOCHS = args['EPOCHS']
    device = args['device']

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = EPOCHS*len(train_dataloader))

    model = model.to(device)

    best_f1 = 0

    for epoch_num in range(EPOCHS):
        label_map = train_dataloader.dataset.ids_to_labels
        
        ## train the model and validate the performance
        train_by_epoch(model, train_dataloader, optimizer, device, scheduler)
        train_metrics = validate(model, train_dataloader, device, label_map)
        val_metrics = validate(model, val_dataloader, device, label_map)
        
        ## write metric to tensorboard
        for metric in ['loss', 'acc', 'f1-score']:
            writer.add_scalars(f'{metric}', {
                "train": train_metrics['macro avg'][metric],
                "validation":  val_metrics['macro avg'][metric],
                }, epoch_num)
            
            
        ## checkpoint the best performance based on macro avg f1-score
        if val_metrics['macro avg']['f1-score'] > best_f1:
            best_f1 = val_metrics['macro avg']['f1-score']
            print("update best model")
            torch.save(model.state_dict(), f"./{saved_dir}/best.pt")
    
    ## save final model
    torch.save(model.state_dict(), f"./{saved_dir}/final.pt")
    return model


####################################### federated learning ##########################################

class NER_FedAvg(FedAlg):
    def generate_models(self):
        return BertModel(num_labels = 19)
    
    def local_train(self, idx):
        ## access trainloader self.dls[idx]['train']
        ## access model self.client_models[idx]
        train_by_epoch(self.client_models[idx], self.dls[idx]['train'], self.optimizers[idx], self.device, self.scheduler)
    
    def local_validate(self, idx):
        ## access trainloader self.dls[idx]['validation']
        ## access local model self.client_models[idx]
        ## access global model self.server_model[idx]
        label_map = self.dls[idx]['train'].dataset.ids_to_labels
        ret_dict = {}
        ret_dict['train'] = validate(self.client_models[idx], self.dls[idx]['train'], self.device, label_map)
        ret_dict['validation'] = validate(self.client_models[idx], self.dls[idx]['validation'], self.device, label_map)
        return ret_dict
    
    def global_validate(self):
        ## access trainloader self.dls[idx]['validation']
        label_map = self.dls[0]['train'].dataset.ids_to_labels
        ret_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        for client_idx in range(self.client_num):
            tmp_train = validate(self.server_model, self.dls[client_idx]['train'], self.device, label_map)
            tmp_validation = validate(self.server_model, self.dls[client_idx]['validation'], self.device, label_map)
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
        
        