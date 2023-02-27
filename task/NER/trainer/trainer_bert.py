from trainer.trainer_base import trainer_base, NER_FedAvg_base
from transformers import get_linear_schedule_with_warmup
from torch.optim import SGD, AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
from seqeval.metrics import classification_report, f1_score
from utils.parse_metric_summary import parse_summary
import os

from models.BERT import BertModel
from utils.nereval import classifcation_report as ner_classificaiton_report

def _shared_train_step(model, trainloader, optimizer, device, scheduler):
    model.train()
    for train_data, train_label in tqdm(trainloader):
        train_label = train_label.to(device)
        mask = train_data['attention_mask'].squeeze(1).to(device)
        input_id = train_data['input_ids'].squeeze(1).to(device)

        optimizer.zero_grad()
        loss, _ = model(input_id, mask, train_label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0) ## optional
        optimizer.step()
        scheduler.step()
        

def _shared_validate(model, dataloader, device, ids_to_labels, prefix):
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
    
    ## save meta data
    metric_dict['meta'] = {
        "true": val_y_true,
        "pred": val_y_pred
    }
    
    metric_dict['strict'] = strict_metric
    metric_dict['lenient'] = lenient_metric
    return metric_dict


class trainer_bert(trainer_base):
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
    

class NER_FedAvg_bert(NER_FedAvg_base):
    def generate_models(self):
        return BertModel(num_labels = 19, model_name=self.model_name)
    
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
        ids_to_labels = self.dls[client_idx]['train'].dataset.ids_to_labels
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
        
        