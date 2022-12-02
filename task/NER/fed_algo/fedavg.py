import torch
import sys
sys.path.append("../")
import torch
import torch.nn as nn
import numpy as np
import copy
from fed_algorithms.fed_utils import train, test, plot_fed, generate_models, write_log_head, write_log_body, communication
from torch.optim import SGD, AdamW
import os

def fed_train(client_weights, args, logfile, generate_models):
    
    # client_num = len(args.datasets)
    # client_weights = [1/client_num for i in range(client_num)]
    client_num = len(client_weights)

    server_model = generate_models().to(args[0].device)
    
    server_model = nn.DataParallel(server_model)
    client_models = [copy.deepcopy(server_model).to(args[i].device) for i in range(client_num)]


    # save by mean
    best_mean_acc = 0
    max_mean_auprc = 0
    min_mean_loss = 1e9
    losses, accs, auprcs, aurocs = {}, {}, {}, {}
    losses['train'] = np.zeros((client_num, args[0].max_epoch))
    losses['val'] = np.zeros((client_num, args[0].max_epoch))
    accs['train'] = np.zeros((client_num, args[0].max_epoch))
    accs['val'] = np.zeros((client_num, args[0].max_epoch))
    auprcs['train'] = np.zeros((client_num, args[0].max_epoch))
    auprcs['val'] = np.zeros((client_num, args[0].max_epoch))
    aurocs['train'] = np.zeros((client_num, args[0].max_epoch))
    aurocs['val'] = np.zeros((client_num, args[0].max_epoch))

    for i in range(args[0].max_epoch):
        #optimizers = [SGD(params=client_models[idx].parameters(), lr=args[idx].lr) for idx in range(client_num)]
        optimizers = [AdamW(params=client_models[idx].parameters(), lr=args[idx].lr, weight_decay=1e-5) for idx in range(client_num)]
        log_str_head = "============ Train epoch {} ============".format(i)
        print(log_str_head)
        # local update
        for client_idx, model in enumerate(client_models):
            train_loss, train_acc = train(model, \
                    args[client_idx].train_dl, \
                    args[client_idx].wk_iter, \
                    optimizers[client_idx], \
                    args[client_idx].criterion,\
                    args[client_idx].device)

        # aggregation & save best model
        with torch.no_grad():
            log_str = ""
            server_model, client_models = communication(args[0], server_model, client_models, client_weights)

            # validation
            for client_index, model in enumerate(client_models):
                train_metrics = test(model, args[client_index].train_dl, args[client_index].criterion, args[client_index].device, format_head='train_')
                val_metrics = test(model, args[client_idx].val_dl, args[client_index].criterion, args[client_index].device, format_head='val_')


                tmp_str = ' Site-{:<10s}| \
                            Train Loss: {train_loss:.4f} | \
                            Train Acc: {train_acc:.4f} |\
                            Train ROC: {train_roc:.4f}|\
                            Train PRC: {train_prc:.4f}|\
                            Val Loss: {val_loss:.4f} | \
                            Val Acc: {val_acc:.4f} |\
                            Val ROC: {val_roc:.4f}|\
                            Val PRC: {val_prc:.4f}'.\
                            format(args[client_index].dataset ,\
                            **train_metrics,
                            **val_metrics)

                log_str += tmp_str + "\n"
                print(tmp_str)

                accs['train'][client_index, i] = train_metrics['train_acc']
                accs['val'][client_index, i] = val_metrics['val_acc']
                losses['train'][client_index, i] = train_metrics['train_loss']
                losses['val'][client_index, i] = val_metrics['val_loss']
                auprcs['train'][client_index, i] = train_metrics['train_prc']
                auprcs['val'][client_index, i] = val_metrics['val_prc']
                aurocs['train'][client_index, i] = train_metrics['train_roc']
                aurocs['val'][client_index, i] = val_metrics['val_roc']
                    
                print("----"*15)


                # save best model
                #if best_mean_acc < np.mean(accs['val'][:, i]):
            #if min_mean_loss > np.mean(losses['val'][:, i]):
            if max_mean_auprc < np.mean(auprcs['val'][:, i]):
                best_mean_acc = np.mean(accs['val'][:, i])
                min_mean_loss = np.mean(losses['val'][:, i])
                max_mean_auprc = np.mean(auprcs['val'][:, i])
                tmp_str = "***update best model, save to {}***\n".format(args[client_index].saved_path)
                print(tmp_str, min_mean_loss)

                saved_models = {}
                for tmp_idx, model in enumerate(client_models):
                    saved_models[args[tmp_idx].dataset] = model.state_dict()
                saved_models['server_model'] = server_model.state_dict()
                torch.save(saved_models, os.path.join(args[0].saved_path, "best.pt"))
    
    saved_models = {}
    for tmp_idx, model in enumerate(client_models):
        saved_models[args[tmp_idx].dataset] = model.state_dict()
    saved_models['server_model'] = server_model.state_dict()
    torch.save(saved_models, os.path.join(args[0].saved_path, "final.pt"))
    # plot
    plot_fed(args, losses, "losses")
    plot_fed(args, accs, "accuracy")
    plot_fed(args, auprcs, "auprc")
    plot_fed(args, aurocs, "auroc")