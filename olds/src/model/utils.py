import numpy as np
import os
import pickle

from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib
import matplotlib.gridspec as gridspec
from scipy.stats.stats import pearsonr
import seaborn as sns


import pdb

FONT_SIZE = 20
FONT_SIZE_TICK = 17
matplotlib.use('Agg')
#sns.set_context("seaborn")
sns.set(style='darkgrid')


class CustomDataset(Dataset):
    def __init__(self, dataset_root, dset_type, exclude_feature_list=[], exclude_all_type=None):

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

        self.print_stats()

        #exclude_feature_list = ['DBP', 'SBP']
        self.exd, self.exs = [], []
        for head in exclude_feature_list:
            if head in self.meta_data['dh2ind'].keys():
                self.exd.extend(self.meta_data['dh2ind'][head])
            else:
                pass
                #print ('wrong excluding param: {}'.format(head))

            if head in self.meta_data['sh2ind'].keys():
                self.exs.extend(self.meta_data['sh2ind'][head])
            else:
                pass
                #print ('wrong excluding param: {}'.format(head))
        
        for i in range(len(self)):
            self.dynamic[i][:,self.exd] = 0.
            self.static[i][self.exs] = 0.

        
        #exclude_all_type = 'static'
        sth = ','.join([head for head in self.meta_data['static_header']])
        dyh = ','.join([head for head in self.meta_data['dynamic_header']])

        if exclude_all_type is not None:
            if exclude_all_type == 'static':
                self.exs = []
                for head in self.meta_data['static_header']:
                    if head in self.meta_data['sh2ind'].keys():
                        self.exs.extend(self.meta_data['sh2ind'][head])
                
                for i in range(len(self)):
                    self.static[i][self.exs] = 0.
            if exclude_all_type == 'dynamic':
                self.exd = []
                self.exs = []
                for head in self.meta_data['dynamic_header']:
                    if head in self.meta_data['sh2ind'].keys():
                        self.exs.extend(self.meta_data['sh2ind'][head])

                    if head in self.meta_data['dh2ind'].keys():
                        self.exd.extend(self.meta_data['dh2ind'][head])

                for i in range(len(self)):
                    self.static[i][self.exs] = 0.
                    self.dynamic[i][:,self.exd] = 0.



    def __len__(self):
        return len(self.dynamic)

    def __getitem__(self, idx):
        return self.dynamic[idx], self.static[idx], self.label[idx]

    def print_stats(self):
        
        # print dynamic params
        import pdb

        d_header = self.meta_data['dynamic_header']
        s_header = self.meta_data['static_header']

        d_avg, d_std = self.meta_data['dynamic_avg'], self.meta_data['dynamic_std']
        s_avg, s_std = self.meta_data['static_avg'], self.meta_data['static_std']

        index = 0 
        print ('==== Dynamic input statistics ====')
        for key, value in self.meta_data['dh2ind'].items():
            print ('{:5d}. {:20s}: {:10.4f}, {:10.4f}'.format(
                index, key, d_avg[value[0]], d_std[value[0]]))
            index += 1


        print ('==== Static input statistics ====')
        print ('min, max, mean, std')
        for key, value in self.meta_data['sh2ind'].items():
            
            if len(value) > 1:
                pass
            else:
                print ('{:5d}. {:20s}: {:10.4f}, {:10.4f}'.format(
                    index, key, s_avg[value[0]], s_std[value[0]]))
                index += 1
                    



        pdb.set_trace()
        
        pass



def collate_fn(batch):
#    dynamic_max_len = max([len(d[0]) for d in batch])
    # sort by length
    dynamic_length = [len(d[0]) for d in batch]
    sorted_idx = np.argsort(dynamic_length)
    samples = [batch[i] for i in reversed(sorted_idx)]

    dynamic = pad_sequence([torch.tensor(d[0].astype(np.float32)) for d in samples]).transpose(0,1) # batch-first
    static = torch.tensor([d[1].astype(np.float32) for d in samples])
    label = torch.tensor([float(d[2]) for d in samples]).unsqueeze(1)

    lengths = [len(d[0]) for d in samples]

    return dynamic, static, label, lengths


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
                label=legend)
                #linewidth=3)
                #label='{} (AUC={:.3f})'.format(legend, auc),
    else:
        ax.plot(fpr, tpr, lw=lw, alpha=alpha, color=color, linestyle=ls)

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
                label=legend)
                #linewidth=3)
                 #label='{} (AP={:.3f}{})'.format(legend, ap, extra_legend),
    else:
        ax.step(recall, precision, lw=lw, alpha=alpha, color=color, linestyle=ls)
                 #linewidth=3)

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=FONT_SIZE)
    ax.set_ylabel('Precision', fontsize=FONT_SIZE)
    ax.set_title('PR curve', fontsize=FONT_SIZE)
    ax.legend(loc='upper right', fontsize=FONT_SIZE-2)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()

    return recall, precision, ap_thr, ap



def plot_trajectory(pred, label, data, savename):

    color_list = sns.color_palette('dark')[:9]

    fig = plt.figure(0, figsize=(14,14))
    outer = gridspec.GridSpec(2,1, height_ratios=[1,1])
    board1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
    board2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])

    icu_id = savename.split('/')[-1].split('-')[0]

    b1_axes = []
    b1_axes.append(plt.subplot(board1[0]))
    b1_axes[0].plot(pred, color='r', marker='o', markerfacecolor='None', markersize=4, label=label)
    b1_axes[0].set_ylim(0,1)
    b1_axes[0].legend(loc='upper left')
    title = 'Prediction Trajectory: {}'.format(icu_id)
    b1_axes[0].set_title(title)
    b1_axes[0].set_ylabel('Prediction score')
    b1_axes[0].set_xlabel('Time (hour)')

    b2_axes = []
    b2_axes.append(plt.subplot(board2[0]))
    color_idx = 0
    for head, seq in data.items():
        b2_axes[0].plot(seq, color=color_list[color_idx], marker='o',
                markerfacecolor='None', markersize=4, label=head)
        color_idx += 1
    b2_axes[0].legend(loc='upper left')
    b2_axes[0].set_ylim(-2.5,2.5)
    b2_axes[0].set_title('Signal values (standardized)')
    b2_axes[0].set_xlabel('Time (hour)')

    plt.tight_layout()
    plt.savefig(savename)


#    for cell in board1:
#        b1_axes.append(plt.subplot(cell))
#        [score_axes[-1].axvline(x=points[key], color='k', linewidth=2) for key in points.keys()]
    





if __name__=='__main__':
    dataset = CustomDataset('./datasets', 'train_data.pkl')
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=10,
            drop_last=True, collate_fn=collate_fn)

    for i, batches in enumerate(data_loader):
        pdb.set_trace()
