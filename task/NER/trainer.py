from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

def train_loop(model, train_dataset, val_dataset, **args):

    BATCH_SIZE = args['BATCH_SIZE']
    LEARNING_RATE = args['LEARNING_RATE']
    EPOCHS = args['EPOCHS']
    
    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0
        train_total = 0

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
              acc = (predictions == label_clean).float().mean()
              
              total_acc_train += acc
              total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0
        val_total = 0

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

        val_accuracy = total_acc_val / train_total
        val_loss = total_loss_val / val_total

        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / train_total: .3f} | Accuracy: {total_acc_train / train_total: .3f} | Val_Loss: {total_loss_val / val_total: .3f} | Accuracy: {total_acc_val / val_total: .3f}')