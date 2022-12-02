from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import copy
import os
from torch.utils.tensorboard import SummaryWriter

from seqeval.metrics import classification_report, f1_score



def train_loop(model, train_dataset, val_dataset, saved_dir, **args):
    
    os.makedirs(saved_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=f"{saved_dir}/tb_events/")

    BATCH_SIZE = args['BATCH_SIZE']
    LEARNING_RATE = args['LEARNING_RATE']
    EPOCHS = args['EPOCHS']
    
    # sampler_train = DistributedSampler(train_dataset, num_replicas=4, rank=0, shuffle=False, drop_last=False)
    # train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, sampler=sampler_train)
    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # model = torch.nn.DataParallel(model)

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0
        train_total = 0
        y_pred = []
        y_true = []
        label_map = train_dataloader.dataset.ids_to_labels
        print(label_map)
        

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            train_total += train_label.shape[0]
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][train_label[i] != -100]
              label_clean = train_label[i][train_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              y_pred.append([label_map[x.item()] for x in predictions])
              y_true.append([label_map[x.item()] for x in label_clean])
              
              
              acc = (predictions == label_clean).float().mean()
              
              total_acc_train += acc
              total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0
        val_total = 0
        val_y_pred = []
        val_y_true = []
        
        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            val_total += val_label.shape[0]
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, val_label)
            

            for i in range(logits.shape[0]):

              logits_clean = logits[i][val_label[i] != -100]
              label_clean = val_label[i][val_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
             
              total_acc_val += acc
              total_loss_val += loss.item()
              val_y_pred.append([label_map[x.item()] for x in predictions])
              val_y_true.append([label_map[x.item()] for x in label_clean])

        val_accuracy = total_acc_val / val_total
        val_loss = total_loss_val / val_total
        
        
        writer.add_scalar('Loss/val', val_loss, epoch_num)
        writer.add_scalar('acc/val', total_acc_val / val_total, epoch_num)
        writer.add_scalar('Loss/train', total_loss_train / train_total, epoch_num)
        writer.add_scalar('acc/train', total_acc_train / train_total, epoch_num)
        
        print("training: ", classification_report(y_true, y_pred))
        print(f1_score(val_y_true, val_y_pred))
        print("_"*10)
        print("validation: ", classification_report(val_y_true, val_y_pred))
        
        
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            print("update best model")
            torch.save(model.state_dict(), f"./{saved_dir}/best.pt")

        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / train_total: .3f} | Accuracy: {total_acc_train / train_total: .3f} | Val_Loss: {total_loss_val / val_total: .3f} | Accuracy: {total_acc_val / val_total: .3f}')

    torch.save(model.state_dict(), f"./{saved_dir}/final.pt")
    return model