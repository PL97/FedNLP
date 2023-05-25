from trainer.trainer_base import trainer_base, NER_FedAvg_base, NER_FedProx_base

from tqdm import tqdm
import torch.nn as nn
from seqeval.metrics import classification_report, f1_score
from utils.parse_metric_summary import parse_summary
import os
import torch
from models.BILSTM_CRF import BIRNN_CRF
from utils.nereval import classifcation_report as ner_classificaiton_report

def _shared_train_step(model, trainloader, optimizer, device, scheduler, scaler):
    model.train()
    model.to(device)
    total_loss = 0
    for X, y in tqdm(trainloader):
        X, y = X.to(device), y.to(device)
        y = y.long()
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = model.loss(X, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0) ## optional
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model.loss(X, y)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0) ## optional
            optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss
        
@torch.no_grad()
def _shared_validate(model, dataloader, device, ids_to_labels, prefix, scaler, return_meta=False):
    model.eval()
    model.to(device)
    preds, targets, pred_orig, target_orig = [], [], [], []
    total_loss_val, val_total = 0, 0
    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
        y = y.long()
        val_total += y.shape[0]

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = model.loss(X, y)
                score, pred = model(X)
        else:
            loss = model.loss(X, y)
            score, pred = model(X)
            
        total_loss_val += loss.item()
        preds.extend(pred)
        targets.extend(y.detach().cpu().tolist())
        
    preds = [item for sublist in preds for item in sublist]
    targets = [item for sublist in targets for item in sublist]
    # update and log
    for p, t in zip(preds, targets):
        if ids_to_labels[t] not in ['<START>', '<STOP>', '<PAD>']:
            if ids_to_labels[p] == '<PAD>':
                pred_orig.append('O')
            else:
                pred_orig.append(ids_to_labels[p])
            target_orig.append(ids_to_labels[t])   
        
    ## prepare metric summary   
    summary = classification_report(y_true=[target_orig], y_pred=[pred_orig], zero_division=0, digits=3)
    print(f"{prefix}: ", summary)
    metric_dict = parse_summary(summary)
    metric_dict['macro avg']['loss'] = total_loss_val/val_total
    
    lenient_metric = ner_classificaiton_report(tags_true=target_orig, tags_pred=pred_orig, mode='lenient')
    strict_metric = ner_classificaiton_report(tags_true=target_orig, tags_pred=pred_orig, mode='strict')
    
    if return_meta:
        metric_dict['meta'] = {
            "true": target_orig,
            "pred": pred_orig
        }
    
    metric_dict['strict'] = strict_metric
    metric_dict['lenient'] = lenient_metric
    
    return metric_dict


class trainer_bilstm_crf(trainer_base):

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

    def inference(self, model, dataloader, prefix):
        return _shared_validate(model=model, \
                                dataloader=dataloader, \
                                device=self.device, \
                                prefix=prefix, \
                                ids_to_labels=self.ids_to_labels, \
                                scaler=self.scaler, \
                                return_meta=True)
    

class NER_FedAvg_bilstm_crf(NER_FedAvg_base):
        
    def generate_models(self):
        return BIRNN_CRF(vocab_size=self.args['vocab_size'], \
                          tagset_size = len(self.ids_to_labels)-2, \
                          embedding_dim=200, \
                          num_rnn_layers=1, \
                          hidden_dim=256, device=self.device)
    
    
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
        
        
class NER_FedProx_bilstm_crf(NER_FedProx_base):
        
    def generate_models(self):
        return BIRNN_CRF(vocab_size=self.args['vocab_size'], \
                          tagset_size = len(self.ids_to_labels)-2, \
                          embedding_dim=200, \
                          num_rnn_layers=1, \
                          hidden_dim=256, device=self.device)
    
    def train_by_epoch(self, server_model, model, train_dl, optimizer, scheduler):   
        mu = 0.1
        def fedprox_train_step(server_model, model, trainloader, optimizer, device, scheduler, scaler):
            model.train()
            model.to(device)
            total_loss = 0
            for X, y in tqdm(trainloader):
                X, y = X.to(device), y.to(device)
                y = y.long()
                optimizer.zero_grad()
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss = model.loss(X, y)
                        
                        ################## calculate the proximal term ###################
                        w_diff = torch.tensor(0., device=device)
                        for w, w_t in zip(server_model.parameters(), model.parameters()):
                            w, w_t = w.to(device), w_t.to(device)
                            w_diff += torch.pow(torch.norm(w-w_t), 2)
                        loss += mu / 2. * w_diff
                        ##################################################################
                        
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    # nn.utils.clip_grad_norm_(model.parameters(), 1.0) ## optional
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = model.loss(X, y)
                    loss.backward()
                    # nn.utils.clip_grad_norm_(model.parameters(), 1.0) ## optional
                    optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            return total_loss
                
                
                
        return fedprox_train_step(server_model=server_model, \
                        model=model, \
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