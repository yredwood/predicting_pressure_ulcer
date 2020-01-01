import numpy as np
import os
import pickle

from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pdb

FONT_SIZE = 20
FONT_SIZE_TICK = 17


class CustomDataset(Dataset):
    def __init__(self, dataset_root, dset_type, exclude_feature_list=None):

        # load train or test dataset
        self.path = os.path.join(dataset_root, dset_type)
        with open(self.path, 'rb') as f:
            data = pickle.load(f)

        self.dynamic = data['xd']
        self.static = data['xs']
        self.label = data['y']

        self.dynamic_dim = self.dynamic[0].shape[-1]
        self.static_dim = self.static[0].shape[-1]

        
        # load meta-data
        path = os.path.join(dataset_root, 'meta_data.pkl')
        with open(path, 'rb') as f:
            self.meta_data = pickle.load(f)

        # TODO feature excluding


    def __len__(self):
        return len(self.dynamic)

    def __getitem__(self, idx):
        return self.dynamic[idx], self.static[idx], self.label[idx]


def collate_fn(batch):
#    dynamic_max_len = max([len(d[0]) for d in batch])
    dynamic = pad_sequence([torch.tensor(d[0].astype(np.float32)) for d in batch]).transpose(0,1) # batch-first
    static = torch.tensor([d[1].astype(np.float32) for d in batch])
    label = torch.tensor([float(d[2]) for d in batch]).unsqueeze(1)
    return dynamic, static, label


def plot_roc_curve(fignum, label, prediction, legend=None, **kwargs):
    lw = kwargs['lw']
    alpha = kwargs['alpha']
    color = kwargs['color']
    ls = kwargs['ls']
    
    fpr, tpr, roc_thr = metrics.roc_curve(label, prediction)
    auc = metrics.auc(fpr, tpr)
    fig = plt.figure(fignum, figsize=(7,6))
    ax = fig.add_subplot(111)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')

    if legend is not None:
        ax.plot(fpr, tpr, lw=lw, alpha=alpha, color=color, linestyle=ls,
                label='{} (AUC={:.3f})'.format(legend, auc),
                linewidth=3)
    else:
        ax.plot(fpr, tpr, lw=lw, alpha=alpha, color=color, linewidth=3, linestyle=ls)

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('1 - Specificity', fontsize=FONT_SIZE)
    ax.set_ylabel('Sensitivity', fontsize=FONT_SIZE)
    ax.set_title('ROC curve', fontsize=FONT_SIZE)
    ax.legend(loc='lower right', fontsize=FONT_SIZE-2)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()

    return fpr, tpr, roc_thr, auc

def get_auroc(label, prediction):
    fpr, tpr, roc_thr = metrics.roc_curve(label, prediction)
    auc = metrics.auc(fpr, tpr)
    return auc

def get_ap(label, prediction):
    return metrics.average_precision_score(label, prediction)

def plot_pr_curve(fignum, label, prediction, legend=None, extra_legend='', clf=False, **kwargs):
    lw = kwargs['lw']
    alpha = kwargs['alpha']
    color = kwargs['color']
    ls = kwargs['ls']

    ap = get_ap(label, prediction)
    precision, recall, ap_thr = metrics.precision_recall_curve(label, prediction)
    #  auc = metrics.auc(recall, precision)
    fig = plt.figure(fignum, figsize=(7, 6))
    ax = fig.add_subplot(111)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')

    if clf:
        ax.clf()
    if legend is not None:
        ax.step(recall, precision, lw=lw, alpha=alpha, color=color, linestyle=ls,
                 label='{} (AP={:.3f}{})'.format(legend, ap, extra_legend),
                 linewidth=3)
    else:
        ax.step(recall, precision, lw=lw, alpha=alpha, color=color, linestyle=ls,
                 linewidth=3)

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=FONT_SIZE)
    ax.set_ylabel('Precision', fontsize=FONT_SIZE)
    ax.set_title('PR curve', fontsize=FONT_SIZE)
    ax.legend(loc='upper right', fontsize=FONT_SIZE-2)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()

    return recall, precision, ap_thr, ap


if __name__=='__main__':
    dataset = CustomDataset('./datasets', 'train_data.pkl')
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=10,
            drop_last=True, collate_fn=collate_fn)

    for i, batches in enumerate(data_loader):
        pdb.set_trace()
