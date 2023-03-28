from trainer.trainer_base import trainer_base, NER_FedAvg_base
from transformers import get_linear_schedule_with_warmup
from torch.optim import SGD, AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
from seqeval.metrics import classification_report, f1_score
from utils.parse_metric_summary import parse_summary
import os
import torch

from models.BERT import BertModel
from utils.nereval import classifcation_report as ner_classificaiton_report

def _shared_train_step(model, trainloader, optimizer, device, scheduler, scaler):
    model.train()
    model.to(device)
    for train_data, train_label in tqdm(trainloader):
        train_label = train_label.to(device)
        mask = train_data['attention_mask'].squeeze(1).to(device)
        input_id = train_data['input_ids'].squeeze(1).to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, _ = model(input_id, mask, train_label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0) ## optional
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, _ = model(input_id, mask, train_label)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0) ## optional
            optimizer.step()
        scheduler.step()
        

@torch.no_grad()
def _shared_validate(model, dataloader, device, ids_to_labels, prefix, scaler, return_meta=False):
    model.eval()
    model.to(device)

    total_acc_val, total_loss_val, val_total = 0, 0, 0
    val_y_pred, val_y_true = [], []
    
    for val_data, val_label in dataloader:

        val_label = val_label.to(device)
        val_total += val_label.shape[0]
        mask = val_data['attention_mask'].squeeze(1).to(device)
        input_id = val_data['input_ids'].squeeze(1).to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, logits = model(input_id, mask, val_label)
        else:
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
            val_y_pred.append([ids_to_labels[x.item()] for x in predictions])
            val_y_true.append([ids_to_labels[x.item()] for x in label_clean])
    
    ## prepare metric summary   
    summary = classification_report(y_true=val_y_true, y_pred=val_y_pred, zero_division=0)
    print(f"{prefix}: ", summary)
    metric_dict = parse_summary(summary)
    metric_dict['macro avg']['loss'] = total_loss_val/val_total
    metric_dict['macro avg']['acc'] = total_acc_val/val_total
    
    ## flatten the list
    val_y_true = [item for sublist in val_y_true for item in sublist]
    val_y_pred = [item for sublist in val_y_pred for item in sublist]
    lenient_metric = ner_classificaiton_report(tags_true=val_y_true, tags_pred=val_y_pred, mode='lenient')
    strict_metric = ner_classificaiton_report(tags_true=val_y_true, tags_pred=val_y_pred, mode='strict')
    
    if return_meta:
        metric_dict['meta'] = {
            "true": val_y_true,
            "pred": val_y_pred
        }
    
    metric_dict['strict'] = strict_metric
    metric_dict['lenient'] = lenient_metric
    return metric_dict


class trainer_bert(trainer_base):

    def train_step(self):
        return _shared_train_step(model=self.model, \
                                  trainloader=self.trainloader, \
                                  optimizer=self.optimizer, \
                                  device=self.device, \
                                  scheduler=self.scheduler, \
                                  scaler=self.scaler)

    def validate(self, dataloader, prefix):
        return _shared_validate(model=self.model, \
                                dataloader=dataloader, \
                                device=self.device, \
                                prefix=prefix, \
                                ids_to_labels=self.ids_to_labels, \
                                scaler=self.scaler)
    
    def inference(self, dataloader, prefix):
        return _shared_validate(model=self.model, \
                                dataloader=dataloader, \
                                device=self.device, \
                                prefix=prefix, \
                                ids_to_labels=self.ids_to_labels, \
                                scaler=self.scaler, \
                                return_meta=True)
    

class NER_FedAvg_bert(NER_FedAvg_base):
        
    def generate_models(self):
        return BertModel(num_labels = self.num_labels, model_name=self.model_name)
    
    def train_by_epoch(self, model, train_dl, optimizer, scheduler):        
        _shared_train_step(model=model, \
                           trainloader=train_dl, \
                           optimizer=optimizer, \
                           scheduler=scheduler, \
                           device=self.device, 
                           scaler=self.scaler)
        
    def validate(self, model, client_idx):
        ret_dict = {}
        ret_dict['train'] =_shared_validate(model=model, \
                                            dataloader=self.dls[client_idx]['train'], \
                                            ids_to_labels=self.ids_to_labels, \
                                            prefix='train', \
                                            device=self.device, \
                                            scaler=self.scaler)
        
        ret_dict['val'] =_shared_validate(model=model, \
                                            dataloader=self.dls[client_idx]['val'], \
                                            ids_to_labels=self.ids_to_labels, \
                                            prefix='val', \
                                            device=self.device, \
                                            scaler=self.scaler)
        return ret_dict
    
    def inference(self, dataloader, prefix):
        return _shared_validate(self.server_model, \
                                dataloader, \
                                ids_to_labels=self.ids_to_labels, \
                                prefix=prefix, \
                                device=self.device, \
                                scaler=self.scaler, \
                                return_meta=True)
        
        