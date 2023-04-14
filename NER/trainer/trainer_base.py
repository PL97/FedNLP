import torch
import os
from collections import defaultdict
import numbers
from transformers import get_linear_schedule_with_warmup
from torch.optim import SGD, AdamW
from torch.utils.tensorboard import SummaryWriter

from fed_algo.fedalg import FedAlg



class trainer_base:
    def __init__(self, model, dls, device, ids_to_labels, lr, epochs, saved_dir, amp=True):
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
        
        ## automatic mixed precision (AMP)
        self.scaler = torch.cuda.amp.GradScaler() if amp else None
    
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
                    "val":  val_metrics['macro avg'][metric],
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
    def __init__(self, dls, client_weights, lrs, max_epoches, aggregation_freq, device, saved_dir, model_name, num_labels, amp, **args):
        self.ids_to_labels = args['ids_to_labels'] #! parent's __init__ calls generate_model, it is where it calls this attribute
        super().__init__(dls, client_weights, lrs, max_epoches, aggregation_freq, device, saved_dir, model_name, num_labels, amp, **args)
       
    
    def generate_models(self):
        pass
    
    def train_by_epoch(self):
        pass
    
    def validate(self, model, idx):
        pass
    
    def local_train(self, idx):
        client_model = self.generate_models().to(self.device)
        client_model.load_state_dict(self.client_state_dict[idx])
        client_optimizer = AdamW(params=client_model.parameters(), lr=self.lrs[idx])
        client_scheduler = get_linear_schedule_with_warmup(client_optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = len(self.dls[idx]['train']))
        self.train_by_epoch(client_model, self.dls[idx]['train'], client_optimizer, client_scheduler)
        self.client_state_dict[idx] = client_model.cpu().state_dict()
    
    def local_validate(self, idx):
        client_model = self.generate_models().to(self.device)
        client_model.load_state_dict(self.client_state_dict[idx])
        return self.validate(client_model, idx)
    
    def global_validate(self):
        ## access trainloader self.dls[idx]['validation']
        ret_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        for client_idx in range(self.client_num):
            global_metrics = self.validate(self.server_model, client_idx)
            for k, v in global_metrics['train'].items():
                for kk, vv in v.items():
                    if isinstance(global_metrics['train'][k][kk], numbers.Number): 
                        ret_dict['train'][k][kk] += global_metrics['train'][k][kk] if k in global_metrics['train'] else 0  # special case: client got 0 samples for token k
                        ret_dict['val'][k][kk] += global_metrics['val'][k][kk] if k in global_metrics['val'] else 0 # special case: client got 0 samples for token k
                    

        ## aggregate results
        ### can use key "validation" also
        for k, v, in ret_dict['train'].items():
            for kk, vv in v.items():
                ret_dict['train'][k][kk] /= self.client_num
                ret_dict['val'][k][kk] /= self.client_num
                
        return ret_dict
    

    def communication(self, server_model, not_update_client=False, bn_exclude=False):
        with torch.no_grad():
            server_state_dict = server_model.cpu().state_dict()
            # aggregate params
            for key in server_state_dict.keys():
                temp = torch.zeros_like(server_state_dict[key], dtype=torch.float32)
                for client_idx in range(self.client_num):
                    temp += self.client_weights[client_idx] * self.client_state_dict[client_idx][key]
                server_state_dict[key].data.copy_(temp)
                # if not not_update_client:
                for client_idx in range(len(self.client_weights)):
                    self.client_state_dict[client_idx][key].data.copy_(server_state_dict[key])
            server_model.load_state_dict(server_state_dict)
        return server_model
    



class NER_FedProx_base(NER_FedAvg_base):
    def local_train(self, idx):
        client_model = self.generate_models().to(self.device)
        client_model.load_state_dict(self.client_state_dict[idx])
        client_optimizer = AdamW(params=client_model.parameters(), lr=self.lrs[idx])
        client_scheduler = get_linear_schedule_with_warmup(client_optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = len(self.dls[idx]['train']))
        self.train_by_epoch(self.server_model, client_model, self.dls[idx]['train'], client_optimizer, client_scheduler)
        self.client_state_dict[idx] = client_model.cpu().state_dict()
    
    
        
        