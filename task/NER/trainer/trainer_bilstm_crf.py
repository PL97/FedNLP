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
from models.BILSTM_CRF import BIRNN_CRF
from utils.nereval import classifcation_report as ner_classificaiton_report

def _shared_train_step(model, trainloader, optimizer, device, scheduler):
    model.train()
    for X, y in tqdm(trainloader):
        X, y = X.to(device), y.to(device)
        y = y.long()
        optimizer.zero_grad()
        loss = model.loss(X, y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0) ## optional
        optimizer.step()
        scheduler.step()
        
@torch.no_grad()
def _shared_validate(model, dataloader, device, ids_to_labels, prefix, return_meta=False):
    model.eval()
    preds, targets, pred_orig, target_orig = [], [], [], []
    total_loss_val, val_total = 0, 0
    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
        y = y.long()
        val_total += y.shape[0]

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
    summary = classification_report(y_true=[target_orig], y_pred=[pred_orig], zero_division=0)
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
    def __init__(self, model, dls, device, ids_to_labels, lr, epochs, saved_dir):
        self.model = model
        self.trainloader = dls['train']
        self.valloader = dls['val']
        self.device = device
        self.ids_to_labels = ids_to_labels
        self.epochs = epochs
        self.lr = lr
        self.saved_dir = saved_dir
    
        ## define solver
        self.optimizer = AdamW(model.parameters(), lr=self.lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                num_warmup_steps = 0,
                                num_training_steps = self.epochs*len(self.trainloader))

        self.model = self.model.to(self.device)
        self.writer = SummaryWriter(log_dir=f"{self.saved_dir}/tb_events/")
        os.makedirs(self.saved_dir, exist_ok=True)

    def train_step(self):
        return _shared_train_step(model=self.model, \
                                  trainloader=self.trainloader, \
                                  optimizer=self.optimizer, \
                                  device=self.device, \
                                  scheduler=self.scheduler)


    def validate(self, dataloader, prefix):
        return _shared_validate(model=self.model, \
                                dataloader=dataloader, \
                                device=self.device, \
                                prefix=prefix, \
                                ids_to_labels=self.ids_to_labels)

    def inference(self, dataloader, prefix):
        return _shared_validate(model=self.model, \
                                dataloader=dataloader, \
                                device=self.device, \
                                prefix=prefix, \
                                ids_to_labels=self.ids_to_labels, \
                                return_meta=True)
    

class NER_FedAvg_bilstm_crf(NER_FedAvg_base):
        
    def generate_models(self):
        return BIRNN_CRF(vocab_size=self.args['vocab_size'], \
                          tagset_size = len(self.args['ids_to_labels'])-2, \
                          embedding_dim=200, \
                          num_rnn_layers=1, \
                          hidden_dim=256, device=self.device)
    
    
    def train_by_epoch(self, client_idx):
        model = self.client_models[client_idx]
        trainloader = self.dls[client_idx]['train']
        optimizer = self.optimizers[client_idx]
        scheduler = self.schedulers[client_idx]
        
        _shared_train_step(model=model, \
                           trainloader=trainloader, \
                           optimizer=optimizer, \
                           scheduler=scheduler, \
                           device=self.device)
        
    def validate(self, model, client_idx):
        trainloader = self.dls[client_idx]['train']
        ids_to_labels = self.args['ids_to_labels']
        valloader = self.dls[client_idx]['val']
        ret_dict = {}
        ret_dict['train'] =_shared_validate(model=model, \
                                            dataloader=trainloader, \
                                            ids_to_labels=ids_to_labels, \
                                            prefix='train', \
                                            device=self.device)
        
        ret_dict['val'] =_shared_validate(model=model, \
                                                 dataloader=valloader, \
                                                 ids_to_labels=ids_to_labels, \
                                                 prefix='val', \
                                                 device=self.device)
        return ret_dict
    
    def inference(self, dataloader, ids_to_labels, prefix):
        return _shared_validate(model=self.server_model, \
                                dataloader=dataloader, \
                                device=self.device, \
                                prefix=prefix, \
                                ids_to_labels=ids_to_labels, \
                                return_meta=True)
        
        