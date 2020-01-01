import torch 
import torch.nn as nn
import numpy as np
import os
import argparse
import pickle

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import CustomDataset
from utils import collate_fn
from utils import get_auroc
from utils import get_ap
from utils import plot_roc_curve
from utils import plot_pr_curve

import pdb

# ================================

def parse_args():
    parser = argparse.ArgumentParser(description='hi')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--max-epoch', default=30, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--model-type', default='MLP')
    parser.add_argument('--exclude-feature', default='')
    parser.add_argument('--output-root', default='results')
    args = parser.parse_args()
    return args


# =================================




class VTonly(nn.Module): # vital only
    def __init__(self, name, seed, dynamic_dim):
        super().__init__()
        self.name = name
        self.seed = seed
        self.lstm = nn.LSTM(dynamic_dim, 32, num_layers=1, batch_first=True)
        self.dense = nn.Linear(32, 1)
        self.act = nn.Sigmoid()

        for name, param in self.named_parameters():
            if 'weight' in name:
                param.data.uniform_(-0.05, 0.05)
            else:
                param.data.uniform_(-0.1, 0.1)
        
    def forward(self, xd, xs, xlen=None):
        xd = nn.utils.rnn.pack_padded_sequence(xd, xlen, batch_first=True)
        x, h = self.lstm(xd)
        x, xlen = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # get the final status
        final = torch.stack([x[i,xlen[i]-1] for i in range(x.shape[0])])
        x = self.act(self.dense(final))
        return x
        


class MLP(nn.Module):
    def __init__(self, name='MLP', seed=0, static_dim=0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(static_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.name = name
        self.seed = seed

        for name, param in self.named_parameters():
            if 'weight' in name:
                param.data.uniform_(-0.05, 0.05)
            else:
                param.data.uniform_(-0.1, 0.1)

    def forward(self, xd, xs, xlen=None):
        return self.net(xs)

class LSTM(nn.Module):

    def __init__(self, name, seed, dynamic_dim, static_dim):
        super().__init__()
        self.name = name
        self.seed = seed
        
        lstm_hdim = 32
        mlp_hdim = 128

        self.lstm = nn.LSTM(dynamic_dim, lstm_hdim, batch_first=True)
        self.mlp = nn.Sequential(
                nn.Linear(static_dim, mlp_hdim),
                nn.ReLU(),
                nn.Dropout(0.5),
        )
        self.dense = nn.Linear(lstm_hdim+mlp_hdim, 1)
        self.act = nn.Sigmoid()

        for name, param in self.named_parameters():
            if 'weight' in name:
                param.data.uniform_(-0.05, 0.05)
            else:
                param.data.uniform_(-0.1, 0.1)

    def forward(self, xd, xs, xlen=None):
        xd = nn.utils.rnn.pack_padded_sequence(xd, xlen, batch_first=True)
        x, h = self.lstm(xd)
        x, xlen = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # get the final status
        lstm_out = torch.stack([x[i,xlen[i]-1] for i in range(x.shape[0])])
        mlp_out = self.mlp(xs)
        out = torch.cat((lstm_out, mlp_out), axis=1)
        out = self.act(self.dense(out))
        return out



def train_epoch(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0.
    for i, batch in enumerate(iterator):

        xd = batch[0].cuda()
        xs = batch[1].cuda()
        y_true = batch[2].cuda()
        xlen = batch[3]

        optimizer.zero_grad()
        y_pred = model(xd, xs, xlen)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def test_epoch(model, iterator, criterion, logging=False):

    model.eval()
    eval_loss = []
    eval_pred = []
    eval_true = []
    for i, batch in enumerate(iterator):
        
        xd = batch[0].cuda()
        xs = batch[1].cuda()
        y_true = batch[2].cuda()
        xlen = batch[3]
        
        y_pred = model(xd, xs, xlen)
        loss = criterion(y_pred, y_true)
        eval_loss.append(loss.item())
        eval_pred.append(y_pred)
        eval_true.append(y_true)

    eval_pred = torch.cat(eval_pred, axis=0).data.cpu().numpy()
    eval_true = torch.cat(eval_true, axis=0).data.cpu().numpy()

    auc = get_auroc(eval_true, eval_pred)
    pr = get_ap(eval_true, eval_pred)


    if logging:
        if not os.path.exists(model.name):
            os.makedirs(model.name)
#        plot_args = {'lw': 1, 'alpha': 0.5, 'color': 'gray', 'ls': '-'}
#        roc_fname = os.path.join(model.name, 'roc_curve_{}'.format(model.seed))
#        plot_roc_curve(0, eval_true, eval_pred, **plot_args)
#        plt.savefig(roc_fname)
#
#        pr_fname = os.path.join(model.name, 'pr_curve_{}'.format(model.seed))
#        plot_pr_curve(1, eval_true, eval_pred, **plot_args)
#        plt.savefig(pr_fname)

        vals_fname = os.path.join(model.name, 'preds_labels_{}.pkl'.format(model.seed))
        with open(vals_fname, 'wb') as f:
            pickle.dump((eval_pred, eval_true), f)
        

    return np.mean(eval_loss), auc, pr




if __name__=='__main__':
    
    args = parse_args()

    model_type = args.model_type
    max_epoch = args.max_epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    name = os.path.join(args.output_root,
            '{}'.format(model_type))
    exclude = args.exclude_feature.split(',')

    train_dataset = CustomDataset('./datasets', 'train_data.pkl', exclude)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=10, drop_last=True, collate_fn=collate_fn)

    test_dataset = CustomDataset('./datasets', 'test_data.pkl', exclude)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn)

    valid_dataset = CustomDataset('./datasets', 'valid_data.pkl', exclude)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn)

    dynamic_dim = train_dataset.dynamic_dim
    static_dim = train_dataset.static_dim


    if model_type == 'MLP':
        model = MLP(name, args.seed, static_dim).cuda()
    elif model_type == 'VTonly':
        model = VTonly(name, args.seed, dynamic_dim).cuda()
    elif model_type == 'LSTM':
        model = LSTM(name, args.seed, dynamic_dim, static_dim).cuda()


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    best_loss = 10000.
    for epoch in range(max_epoch):

        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        valid_loss, valid_auc, valid_ap = test_epoch(model, valid_loader, criterion)

        print ('Epoch {:4d} | train_loss: {:.4f}, valid_loss: {:.4f},  '
                'AUC: {:.4f},  AP: {:.4f}'.format(
                    epoch, train_loss, valid_loss, valid_auc, valid_ap)
        )

        if best_loss > valid_loss:
            test_loss, test_auc, test_ap = test_epoch(model, test_loader, criterion, logging=True)
            best_loss = valid_loss
        else:
            #print ('best model is NOT updated')
            pass
    
    print ('============================================')
    print ('Best model AUC: {:.4f}, AP: {:.4f}'.format(test_auc, test_ap))
















        #
