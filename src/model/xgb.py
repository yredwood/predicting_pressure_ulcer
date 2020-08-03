import pickle
import argparse
import matplotlib
matplotlib.use('Agg')

import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from matplotlib.ticker import FormatStrFormatter

from utils import CustomDataset
from utils import get_auroc
from utils import get_ap

plt.style.use('grayscale')


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
    parser.add_argument('--dataset-root', default='datasets/')
    args = parser.parse_args()
    return args


# =================================


if __name__ == '__main__':

    args = parse_args()

    param = {
        'max_depth': 4, 'eta': 1, 'silent': 1,
        'objective': 'binary:logistic'
    }
    param['nthread']     = 12
    param['eval_metric'] = 'auc'

    name = os.path.join(
        args.output_root,
        '{}'.format(args.model_type)
    )
    exclude = args.exclude_feature.split(',')

    train_dataset = CustomDataset(args.dataset_root, 'train_data.pkl', exclude)
    valid_dataset = CustomDataset(args.dataset_root, 'valid_data.pkl', exclude)
    test_dataset = CustomDataset(args.dataset_root, 'test_data.pkl', exclude)

    dtrain = xgb.DMatrix(np.array(train_dataset.static), label=train_dataset.label)
    dvalid = xgb.DMatrix(np.array(valid_dataset.static), label=valid_dataset.label)
    dtest  = xgb.DMatrix(np.array(test_dataset.static), label=test_dataset.label)

    evallist = [(dvalid, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, 10, evallist)

    pred = bst.predict(dtest)
    
    auc = get_auroc(test_dataset.label, pred)
    ap = get_ap(test_dataset.label, pred)
    
    save_root = os.path.join(args.output_root,
            '{}'.format(args.model_type))
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    fname = os.path.join(save_root, 'preds_labels_{}.pkl'.format(args.seed))
    with open(fname, 'wb') as f:
        pickle.dump((pred, test_dataset.label), f)

    print ('AUC: {:.3f}, AP: {:.3f}'.format(auc, ap))



#FONT_SIZE = 20
#FONT_SIZE_TICK = 17
#
#def read_pickle(fname):
#    with open(fname, 'rb') as fp:
#        records = pickle.load(fp)
#    return records
#
#def get_average_precision(label, prediction):
#    return metrics.average_precision_score(label, prediction)
#
#def plot_pr_curve(fignum, label, prediction, legend=None, extra_legend='', clf=False, **kwargs):
#    lw = kwargs['lw']
#    alpha = kwargs['alpha']
#    color = kwargs['color']
#    ls = kwargs['ls']
#
#    ap = get_average_precision(label, prediction)
#    precision, recall, ap_thr = metrics.precision_recall_curve(label, prediction)
#    #  auc = metrics.auc(recall, precision)
#    fig = plt.figure(fignum, figsize=(7, 6))
#    ax = fig.add_subplot(111)
#
#    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#    ax.set_axisbelow(True)
#    ax.grid(color='gray', linestyle='dashed')
#
#    if clf:
#        ax.clf()
#    if legend is not None:
#        ax.step(recall, precision, lw=lw, alpha=alpha, color=color, linestyle=ls,
#                 label='{} (AP={:.3f}{})'.format(legend, ap, extra_legend),
#                 linewidth=3)
#    else:
#        ax.step(recall, precision, lw=lw, alpha=alpha, color=color, linestyle=ls,
#                 linewidth=3)
#
#    ax.set_xlim([-0.05, 1.05])
#    ax.set_ylim([-0.05, 1.05])
#    ax.set_xlabel('Recall (Sensitivity)', fontsize=FONT_SIZE)
#    ax.set_ylabel('Precision', fontsize=FONT_SIZE)
#    ax.set_title('PR curve', fontsize=FONT_SIZE)
#    ax.legend(loc='upper right', fontsize=FONT_SIZE-2)
#    ax.tick_params(labelsize=FONT_SIZE_TICK)
#    plt.tight_layout()
#
#    return recall, precision, ap_thr, ap
#
#def plot_roc_curve(fignum, label, prediction, legend=None, **kwargs):
#    lw = kwargs['lw']
#    alpha = kwargs['alpha']
#    color = kwargs['color']
#    ls = kwargs['ls']
#
#    fpr, tpr, roc_thr = metrics.roc_curve(label, prediction)
#    auc = metrics.auc(fpr, tpr)
#    fig = plt.figure(fignum, figsize=(7,6))
#    ax = fig.add_subplot(111)
#
#    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#    ax.set_axisbelow(True)
#    ax.grid(color='gray', linestyle='dashed')
#
#    if legend:
#        ax.plot(fpr, tpr, lw=lw, alpha=alpha, color=color, linestyle=ls,
#                label='XGB AUC={:.3f})'.format(auc),
#                linewidth=3)
#    else:
#        ax.plot(fpr, tpr, lw=lw, alpha=alpha, color=color, linewidth=3, linestyle=ls)
#
#    ax.set_xlim([-0.05, 1.05])
#    ax.set_ylim([-0.05, 1.05])
#    ax.set_xlabel('1 - Specificity', fontsize=FONT_SIZE)
#    ax.set_ylabel('Sensitivity', fontsize=FONT_SIZE)
#    ax.set_title('ROC curve', fontsize=FONT_SIZE)
#    ax.legend(loc='lower right', fontsize=FONT_SIZE-2)
#    ax.tick_params(labelsize=FONT_SIZE_TICK)
#    plt.tight_layout()
#
#    return fpr, tpr, roc_thr, auc
#
#fname = 'datasets/06_feature_eng.pkl'
#data = read_pickle(fname)
#
##np.random.seed(6150)
#
#param = {
#    'max_depth': 4, 'eta': 1, 'silent': 1,
#    'objective': 'binary:logistic'
#}
#param['nthread']     = 12
#param['eval_metric'] = 'auc'
#
#num_round = 10
#
#os.environ['CUDA_VISIBLE_DEVICES'] = '6'
#
#labels = []
#preds = []
#
#xs = data['xs']
#ys = data['ys']
#
#xs = xs.astype(np.float32)
#ys = ys.astype(np.float32)
#
#r_idx = np.random.permutation(ys.shape[0])
#xs = xs[r_idx]
#ys = ys[r_idx]
#
#chunk_size = int(ys.shape[0]/5)
#
#for i in range(5):
#    print('Experiments %d'%i)
#
#    sp = i*chunk_size
#    ep = (i+1)*chunk_size
#
#    test_xs = xs[sp:ep]
#    test_ys = ys[sp:ep]
#
#    train_xs = np.concatenate((xs[:sp], xs[ep:]), axis=0)
#    train_ys = np.concatenate((ys[:sp], ys[ep:]), axis=0)
#
#    print('TRAIN POS RATE: %.4f'%(np.sum(train_ys)/len(train_ys)))
#    print('TEST POS RATE: %.4f'%(np.sum(test_ys)/len(test_ys)))
#
#    print(train_xs.shape, train_ys.shape)
#    print(test_xs.shape, test_ys.shape)
#
#    dtrain = xgb.DMatrix(train_xs, label=train_ys)
#    dtest  = xgb.DMatrix(test_xs, label=test_ys)
#
#    evallist = [(dtest, 'eval'), (dtrain, 'train')]
#    bst = xgb.train(param, dtrain, num_round, evallist)
#
#    test_ys_pred = bst.predict(dtest)
#
#    plot_roc_curve(0, test_ys, test_ys_pred, lw=1, alpha=0.8, \
#                   color='0.9', ls='-')
#
#    labels.extend(test_ys)
#    preds.extend(test_ys_pred)
#
#print(len(labels), len(preds))
#
#plot_roc_curve(0, labels, preds, lw=2, alpha=1.0, \
#               color='0.2', ls='-', legend='XGB')
#plt.savefig(os.path.join('./results/', 'initial_result.png'))
