from trainer.trainer_base import trainer_base, RE_FedAvg_base
from transformers import get_linear_schedule_with_warmup
from torch.optim import SGD, AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
from utils.parse_metric_summary import parse_summary
import os
from sklearn.metrics import classification_report
import torch

from models.BERT import BertModel

def _shared_train_step(model, trainloader, optimizer, device, scheduler, scaler):
    model.train()
    for train_data, train_label in tqdm(trainloader):
        train_label = train_label.to(device)
        mask = train_data['attention_mask'].squeeze(1).to(device)
        input_id = train_data['input_ids'].squeeze(1).to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, _ = model(input_id, mask, train_label)[:2]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, _ = model(input_id, mask, train_label)[:2]
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0) ## optional
            optimizer.step()
        scheduler.step()
        

@torch.no_grad()
def _shared_validate(model, dataloader, device, ids_to_labels, prefix, scaler, return_meta=False):
    model.eval()

    total_acc_val, total_loss_val, val_total = 0, 0, 0
    val_y_pred, val_y_true = [], []
    
    for val_data, val_label in dataloader:

        val_label = val_label.to(device)
        val_total += val_label.shape[0]
        mask = val_data['attention_mask'].squeeze(1).to(device)
        input_id = val_data['input_ids'].squeeze(1).to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, logits = model(input_id, mask, val_label)[:2]
        else:
            loss, logits = model(input_id, mask, val_label)[:2]
        
        val_y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
        val_y_true.extend(val_label.detach().cpu().tolist())
        
        
    ## prepare metric summary   
    target_names = [str(x) for x in ids_to_labels.values()]
    try:
        summary = classification_report(y_true=val_y_true, y_pred=val_y_pred, \
                    target_names=target_names, zero_division=0)
        print(f"{prefix}: ", summary)
        metric_dict = parse_summary(summary)
    except:
        from collections import defaultdict
        metric_dict = defaultdict(lambda: defaultdict(lambda: 0.))
    
    metric_dict['macro avg']['loss'] = total_loss_val/val_total
    metric_dict['macro avg']['acc'] = total_acc_val/val_total
    
    if return_meta:
        metric_dict['meta'] = {
            "true": val_y_true,
            "pred": val_y_pred
        }
    
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
        
    

class RE_FedAvg_bert(RE_FedAvg_base):
    def generate_models(self):
        return BertModel(num_labels = self.num_labels, model_name=self.model_name)
    
    def train_by_epoch(self, client_idx):
        model = self.client_models[client_idx]
        trainloader = self.dls[client_idx]['train']
        optimizer = self.optimizers[client_idx]
        scheduler = self.schedulers[client_idx]
        
        _shared_train_step(model=model, \
                           trainloader=trainloader, \
                           optimizer=optimizer, \
                           scheduler=scheduler, \
                           device=self.device, \
                           scaler=self.scaler)
        
    def validate(self, model, client_idx):
        trainloader = self.dls[client_idx]['train']
        valloader = self.dls[client_idx]['val']
        ret_dict = {}
        ret_dict['train'] =_shared_validate(model=model, \
                                            dataloader=trainloader, \
                                            ids_to_labels=self.ids_to_labels, \
                                            prefix='train', \
                                            device=self.device, \
                                            scaler=self.scaler)
        
        ret_dict['val'] =_shared_validate(model=model, \
                                                 dataloader=valloader, \
                                                 ids_to_labels=self.ids_to_labels, \
                                                 prefix='val', \
                                                 device=self.device, \
                                                 scaler=self.scaler)
        return ret_dict

    def inference(self, dataloader, ids_to_labels, prefix):
        return _shared_validate(self.server_model, \
                                dataloader, \
                                ids_to_labels=ids_to_labels, \
                                prefix=prefix, \
                                device=self.device, \
                                scaler=self.scaler, \
                                return_meta=True)
        
        
        