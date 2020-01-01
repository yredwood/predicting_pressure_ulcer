import os
import sys
import numpy as np 
import pickle
import matplotlib.pyplot as plt

from utils import get_auroc
from utils import get_ap
from utils import plot_roc_curve
from utils import plot_pr_curve

import pdb



if __name__=='__main__':
    # get argument as model type
    try:
        model_type = int(sys.argv[1])
    except:
        model_type = 0

    model = ['MLP', 'LSTM'][model_type] 

    root_dir = 'results/{}'.format(model)
    flist = [os.path.join(root_dir, _dir) for _dir in os.listdir(root_dir) \
            if _dir.endswith('.pkl')]

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
    plot_roc_curve(0, trues, preds, **plot_args)
    plt.savefig( os.path.join(root_dir, 'auc') )
    plot_pr_curve(1, trues, preds, **plot_args)
    plt.savefig( os.path.join(root_dir, 'pr') )
        

    print ('Results for {} model:: {} repetition'.format(model, len(flist)))
    print ('AUC: {:.3f} ({:.3f})'.format(np.mean(aucs), np.std(aucs)))
    print ('AP: {:.3f} ({:.3f})'.format(np.mean(aps), np.std(aps)))









    #