import os
import sys
import numpy as np 
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import get_auroc
from utils import get_ap
from utils import plot_roc_curve
from utils import plot_pr_curve

import pdb



if __name__=='__main__':
    # get argument as model type
    try:
        model = sys.argv[1]
    except:
        model = 'MLP'


    root_dir = 'results/{}'.format(model)
    flist = [os.path.join(root_dir, _dir) for _dir in os.listdir(root_dir) \
            if 'preds_labels' in _dir]

    aucs, aps = [], []
    preds, trues = [], []
    for fname in flist:
        with open(fname, 'rb') as f:
            data = pickle.load(f) # (pred, label) tuple
        
        aucs.append( get_auroc(data[1], data[0]) )
        aps.append( get_ap(data[1], data[0]) )
        preds.append(data[0])
        trues.append(data[1])
    
        plot_args = {'lw': 1, 'alpha': 0.5, 'color': 'gray', 'ls': '-'}
        plot_roc_curve(0, data[1], data[0], **plot_args)
        plot_pr_curve(1, data[1], data[0], **plot_args)
    
    plot_args = {'lw': 1, 'alpha': 0.9, 'color': 'black', 'ls': '-'}
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    aucstr = '{} AUC: {:.3f} ({:.3f})'.format(model, np.mean(aucs), np.std(aucs))
    apstr = '{} AP: {:.3f} ({:.3f})'.format(model, np.mean(aps), np.std(aucs))
    plot_roc_curve(0, trues, preds, legend=aucstr, **plot_args)
    plt.savefig( os.path.join(root_dir, 'auc') )
    plot_pr_curve(1, trues, preds, legend=apstr, **plot_args)
    plt.savefig( os.path.join(root_dir, 'pr') )
        
    print ('Results for {} model:: {} repetition'.format(model, len(flist)))
    print ('AUC: {:.3f} ({:.3f})'.format(np.mean(aucs), np.std(aucs)))
    print ('AP: {:.3f} ({:.3f})'.format(np.mean(aps), np.std(aps)))


    # feature importance ::
    flist = [os.path.join(root_dir, _dir) for _dir in os.listdir(root_dir) \
            if 'feature_importance' in _dir]
    if len(flist) == 0:
        exit()

    feature_importance = []
    for fname in flist:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        
        header= data['header']
        feature_importance.append(data['feature_importance'])

    mean = np.mean(feature_importance, axis=0)
    std = np.std(feature_importance, axis=0)
    sorted_idx = np.argsort(mean)
    output_string = []
    for i in reversed(sorted_idx):
        output = '{:20s} : {:.3f} ({:.3f})'.format(header[i], mean[i], std[i])
        print (output)
        output_string.append(output)
        
    fname = os.path.join(root_dir, 'fi.txt')
    with open(fname, 'w') as f:
        f.writelines( '\n'.join(output_string) )









    #
