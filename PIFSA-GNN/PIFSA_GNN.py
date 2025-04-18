import numpy as np
from model import PIFSA_GNN
from data_loader import DataLoader
import torch
import torch.optim as optim
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import random
from datetime import datetime

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = DataLoader(args.dataset)
    kg = data_loader.load_kg()
    df_dataset = data_loader.load_dataset()

    def train_test_val_split(data, ratio_train, ratio_test, ratio_val):
        train, middle = train_test_split(data, train_size=ratio_train, test_size=ratio_test + ratio_val, random_state=2023)
        ratio = ratio_val/(1-ratio_train)
        test, validation = train_test_split(middle, test_size=ratio)
        return train, test, validation

    x_train, x_test, x_validation = train_test_val_split(df_dataset, 0.6, 0.2, 0.2)

    UAndI = dict()
    df_len = len(x_train['userID'])
    for index, row in tqdm(x_train.iterrows(), total=df_len, desc='Obtain items') :
        userID = row['userID']
        itemID = row['itemID']
        label = row['label']
        if(label == 1):
            if(userID) in UAndI:
                UAndI[userID].append(itemID)
            else:
                UAndI[userID] = [itemID]

    UAndI_len = len(UAndI)
    for k, v in tqdm(UAndI.items(), total=UAndI_len, desc='Organize items'):
        v_len = len(v)
        if (v_len < args.neighbor_sample_size):
            v.extend(random.choices(v, k=args.neighbor_sample_size - v_len))
            UAndI[k] = v
        elif (v_len > args.neighbor_sample_size):
            v = random.sample(v, args.neighbor_sample_size)
            UAndI[k] = v

    class PIFSA_GNN_Dataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.df = df
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            user_id = np.array(self.df.iloc[idx]['userID'])
            item_id = np.array(self.df.iloc[idx]['itemID'])
            label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
            return user_id, item_id, label

    train_dataset = PIFSA_GNN_Dataset(x_train)
    validation_dataset = PIFSA_GNN_Dataset(x_validation)
    test_dataset = PIFSA_GNN_Dataset(x_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    num_user, num_entity, num_relation = data_loader.get_num()
    net = PIFSA_GNN(num_user, num_entity, num_relation, kg, args, device, UAndI).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    print('device: ', device)

    # train
    auc_score_list = []
    f1_score_list = []
    precision_score_list = []
    recall_score_list = []

    auc_score_t_list = []
    f1_score_t_list = []
    precision_score_t_list = []
    recall_t_score_list = []

    train_loader_len = len(train_loader)
    validation_loader_len = len(validation_loader)
    test_loader_len = len(test_loader)

    for epoch in range(args.n_epochs):
        net.train()
        for i, (user_ids, item_ids, labels) in tqdm(enumerate(train_loader),total=train_loader_len,desc='Train'):
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(user_ids, item_ids).to(device) # [batch_size]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # evaluate
        with torch.no_grad():
            total_roc = 0
            total_f1 = 0
            total_precision = 0
            total_recall = 0
            net.eval()
            for user_ids, item_ids, labels in tqdm(validation_loader, total=validation_loader_len,desc='Validation'):
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                outputs = net(user_ids, item_ids).to(device)
                pre_label = [1 if o.item() >= 0.5 else 0 for o in outputs]
                pre_label = torch.Tensor(pre_label)
                try:
                    total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                    total_f1 += f1_score(labels.cpu().detach().numpy(),pre_label.cpu().detach().numpy(), average='binary')
                    total_precision += precision_score(labels.cpu().detach().numpy(),pre_label.cpu().detach().numpy(), average='binary')
                    total_recall += recall_score(labels.cpu().detach().numpy(),pre_label.cpu().detach().numpy(), average='binary')
                except ValueError:
                    pass
            auc_score_list.append(total_roc / len(validation_loader))
            f1_score_list.append(total_f1/ len(validation_loader))
            precision_score_list.append(total_precision/len(validation_loader))
            recall_score_list.append(total_recall/len(validation_loader))
        # test
        with torch.no_grad():
            total_t_roc = 0
            total_t_f1 = 0
            total_t_precision = 0
            total_t_recall = 0
            net.eval()
            for user_ids, item_ids, labels in tqdm(test_loader, total=test_loader_len,desc='Test'):
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                outputs = net(user_ids, item_ids).to(device)
                pre_label = [1 if o.item() >= 0.5 else 0 for o in outputs]
                pre_label = torch.Tensor(pre_label)
                try:
                    total_t_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                    total_t_f1 += f1_score(labels.cpu().detach().numpy(), pre_label.cpu().detach().numpy(), average='binary')
                    total_t_precision += precision_score(labels.cpu().detach().numpy(), pre_label.cpu().detach().numpy(), average='binary')
                    total_t_recall += recall_score(labels.cpu().detach().numpy(), pre_label.cpu().detach().numpy(), average='binary')
                except ValueError:
                    pass
            auc_score_t_list.append(total_t_roc / len(test_loader))
            f1_score_t_list.append(total_t_f1 / len(test_loader))
            precision_score_t_list.append(total_t_precision / len(test_loader))
            recall_t_score_list.append(total_t_recall / len(test_loader))
        print('\n' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              '[Epoch {0}] '.format(epoch + 1),
              'Val auc:{:.4f}'.format(total_roc / len(validation_loader)),
              'f1:{:.4f}'.format(total_f1 / len(validation_loader)),
              'pre:{:.4f}'.format(total_precision / len(validation_loader)),
              'recall:{:.4f}'.format(total_recall / len(validation_loader)),
              'Test auc:{:.4f}'.format(total_t_roc / len(test_loader)),
              'f1:{:.4f}'.format(total_t_f1 / len(test_loader)),
              'pre:{:.4f}'.format(total_t_precision / len(test_loader)),
              'recall:{:.4f}'.format(total_t_recall / len(test_loader))
              )




