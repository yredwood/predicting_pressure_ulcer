import pickle
import matplotlib
matplotlib.use('Agg')

import numpy as np
import xgboost as xgb
import os
import time
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
import csv

import tensorflow as tf
from tensorflow.keras import layers


import pdb


# ======================================
tf.random.set_seed(100)
np.random.seed(100)

model_type = ['LSTM', 'MLP', 'VT'][0]
use_branden = True
positive_upsample = True
FONT_SIZE = 20
FONT_SIZE_TICK = 17
plt.style.use('grayscale')
fold_idx = 0

model_name = '{}_{}'.format(model_type,
        'Br' if use_branden else 'NoBr')
# ======================================


VITAL_SIGNS_HEADER = [
    'BRANDEN_SCORE', 'GCS', 'HR', 'RR', 'TEMPERATURE',
    'SBP', 'DBP', 'MBP', 'SaO2', 'SpO2'
]

LAB_HEADER = [
    'Lactate', 'Oxygen Saturation', 'pCO2', 'pH', 'pO2',
    'Albumin', 'Bicarbonate', 'Total Bilirubin', 'Creatinine',
    'Glucose', 'Potassium', 'Sodium', 'Troponin I', 'Troponin T',
    'Urea Nitrogen', 'Hematocrit', 'Hemoglobin', 'INR(PT)',
    'Neutrophils', 'Platelet Count', 'White Blood Cells',
    'Position Change', 'Pressure Reducing Device',
]
HEADER = VITAL_SIGNS_HEADER + LAB_HEADER

def read_stats(fname):
    avgs = np.zeros(len(HEADER))
    stds = np.zeros(len(HEADER))
    with open(fname, 'r') as csvfile:
        spm_reader = csv.reader(csvfile)
        for i, row in enumerate(spm_reader):
            if i==0:
                continue
            head = row[0]
            avg = row[2]
            std = row[3]
            hidx = HEADER.index(head)
            avgs[hidx] = float(avg)
            stds[hidx] = float(std)
    return avgs, stds

def read_csv(fname):
    with open(fname, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        dict_list = [row for row in reader]
    return dict_list

def hashing(records, key):
    records_dict = {}
    for record in records:
        r_key = record[key]
        records_dict[r_key] = records_dict.get(r_key, list())
        records_dict[r_key].append(record)
    return records_dict

def read_data(data_root, fold_num=0, random_seed=0):
    dynamic_fname = os.path.join(data_root, '05_converter.pkl')
    with open(dynamic_fname, 'rb') as fp:
        dynamic_data = pickle.load(fp)

    static_fname = os.path.join(data_root, '06_feature_eng.pkl')
    with open(static_fname, 'rb') as fp:
        static_data = pickle.load(fp)

    dynamic = []; label = []; static = []
    for ids in dynamic_data['mats'].keys():

        dynamic.append(dynamic_data['mats'][ids])
        sid = static_data['icustay_id'].index(ids)
        static.append([float(_s) for _s in static_data['xs'][sid]])

        if dynamic_data['ys'][ids] == 'case':
            label.append(1)
        elif dynamic_data['ys'][ids] == 'control':
            label.append(0)
        else:
            raise ValueError()
        assert label[-1] == static_data['ys'][sid]
    
    np.random.seed(random_seed)
    r_idx = np.random.permutation(len(label))
    x = [dynamic[_r] for _r in r_idx]
    y = [label[_r] for _r in r_idx]
    xs = np.array(static)[r_idx] # (B,Ds)

    xall = np.concatenate(x, axis=0)
    avg = np.mean(xall, 0)
    std = np.std(xall, 0)

    x = [(_x - avg) / std for _x in x]

    avg = np.mean(xs, axis=0)
    std = np.std(xs, axis=0)
    # only standardize non-categorical features
    for i in range(len(xs[0])):
        if not static_data['cs'][i]:
            xs[:,i] = (xs[:,i] - avg[i]) / (std[i] + 1e-8)
    xs = [_x for _x in xs]

    if not use_branden:
        dynamic_branden_index = HEADER.index('BRANDEN_SCORE')
        static_branden_index = [dynamic_branden_index + \
                x[0].shape[-1]*i for i in range(4)] # avg, std, max_val, min_val
        for i in range(len(x)):
            x[i][:,dynamic_branden_index] = 0
            xs[i][static_branden_index] = 0

    # add case samples for data imbalance: 3 times more case data
    # split 1:4 test:train
    split_idx = int(len(x)*0.2)
    sp = fold_num * split_idx
    ep = (1+fold_num) * split_idx
    
    test_x = (x[sp:ep], xs[sp:ep])
    test_y = y[sp:ep]
    train_x = (
        x[:sp] + x[ep:],
        xs[:sp] + xs[ep:]
    )
    train_y = y[:sp] + y[ep:]

    if positive_upsample:
        add_x = []; add_xs = []
        for i in range(len(train_y)):
            if train_y[i] == 1:
                for _ in range(4):
                    add_x.append(train_x[0][i])
                    add_xs.append(train_x[1][i])
        
        train_x[0].extend(add_x)
        train_x[1].extend(add_xs)
        train_y.extend([1 for _ in range(len(add_x))])

    return train_x, train_y, test_x, test_y

def pad_sequences(x, value=-5):
    return tf.keras.preprocessing.sequence.pad_sequences(x, padding='post', dtype='float32', value=value)

def bin_accuracy(y, y_hat):
    return tf.reduce_mean(tf.dtypes.cast(
        tf.dtypes.cast(y_hat > 0.5, tf.int32) == tf.dtypes.cast(y, tf.int32), tf.float32))

def compute_mask(x):
    # x.shape (bsz, timestep, dim)
    zeros = tf.dtypes.cast(x==-5, tf.float32)
    mask = tf.reduce_mean(zeros, axis=2) != 1
    return mask


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
                label='{} AUC={:.3f})'.format(legend, auc),
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

def get_average_precision(label, prediction):
    return metrics.average_precision_score(label, prediction)

def plot_pr_curve(fignum, label, prediction, legend=None, extra_legend='', clf=False, **kwargs):
    lw = kwargs['lw']
    alpha = kwargs['alpha']
    color = kwargs['color']
    ls = kwargs['ls']

    ap = get_average_precision(label, prediction)
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


class LSTM(tf.keras.Model):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
            layers.Dropout(0.5),
        ])

        self.net = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
        ])
        self.dense = layers.Dense(1, activation='sigmoid')

    def call(self, xd, xs, training, mask=None):
        # x: (xd, xs)
        yd = self.lstm(xd, training=training, mask=mask)
        ys = self.net(xs, training=training)
        out = self.dense(tf.concat([yd, ys], axis=1))

        return out


class VTonly(tf.keras.Model):
    def __init__(self):
        super(VTonly, self).__init__()
        self.lstm = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid'),
        ])

    def call(self, xd, xs, training, mask=None):
        return self.lstm(xd, training=training, mask=mask)

class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, xd, xs, training, mask=None):
        return self.net(xs, training=training, mask=mask)


def batch_fn(xd, xs, y):
    # x: list of batch data
    # ex) [[1,2,3], [2,3,4,5,6], ...]
    input_xd = tf.convert_to_tensor(pad_sequences(xd))
    input_xs = tf.convert_to_tensor(xs)
    input_y = tf.expand_dims(y, 1)
    return input_xd, input_xs, input_y



micro_preds = []; micro_targets = []
aucs = []; aps = []
for n_exec in range(1):

    # training code
    pred_all = []; label_all = []
    # =========PARAMS==============
    lr = 1e-3
    n_epochs = 50
    batch_size = 32
    optimizer = tf.optimizers.Adam(lr)
    loss_fn = tf.losses.BinaryCrossentropy()
    metric_fn = bin_accuracy
    if model_type=='LSTM':
        model = LSTM()
    elif model_type=='VT':
        model = VTonly()
    elif model_type=='MLP':
        model = MLP()


    model.compile()
    # =============================

    data_root = './datasets'

    train_x, train_y, test_x, test_y = read_data(data_root, fold_idx, random_seed=n_exec)
    for epoch in range(n_epochs):
        t0 = time.time()
        
        # train epoch
        tr_losses = []; tr_accs = []
        idx = np.arange(len(train_y))
        np.random.shuffle(idx)
        
        for start_idx in tqdm(range(0, len(train_y), batch_size)):
            
            end_idx = min(start_idx + batch_size, len(train_y))
            input_xd, input_xs, input_y = batch_fn(
                    [train_x[0][_i] for _i in idx[start_idx:end_idx]],
                    [train_x[1][_i] for _i in idx[start_idx:end_idx]],
                    [train_y[_i] for _i in idx[start_idx:end_idx]],
            )

            with tf.GradientTape() as tape:
                mask = compute_mask(input_xd)
               # y_hat = model(input_xs, tf.constant(True))
                y_hat = model(input_xd, input_xs, tf.constant(True), mask)
                loss = loss_fn(input_y[:,0], y_hat[:,0])
                acc = metric_fn(input_y, y_hat)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                tr_losses.append(tf.reduce_mean(loss))
                tr_accs.append(tf.reduce_mean(acc))


        # validate epoch
        te_losses = []; te_accs = []; te_preds = []
        idx = np.arange(len(test_y))
        for start_idx in tqdm(range(0, len(test_y), batch_size)):

            end_idx = min(start_idx + batch_size, len(test_y))
            input_xd, input_xs, input_y = batch_fn(
                    [test_x[0][_i] for _i in idx[start_idx:end_idx]],
                    [test_x[1][_i] for _i in idx[start_idx:end_idx]],
                    [test_y[_i] for _i in idx[start_idx:end_idx]],
            )
            
            mask = compute_mask(input_xd)
            #y_hat = model(input_xs, tf.constant(False))
            y_hat = model(input_xd, input_xs, tf.constant(False), mask)
            loss = loss_fn(input_y, y_hat)
            acc = metric_fn(input_y, y_hat)

            te_losses.append(tf.reduce_mean(loss))
            te_accs.append(tf.reduce_mean(acc))
            te_preds.append(y_hat)

        te_preds = tf.squeeze(tf.concat(te_preds, axis=0)).numpy()
        te_labels = np.array(test_y)

        print ('Epoch {:4d}| Training Stats: (loss: {:.4f}, acc: {:.4f}) | '
                'Test Stats: (loss: {:.4f}, acc: {:.4f} in {:.2f} sec)'.format(
                epoch,
                tf.reduce_mean(tr_losses),
                tf.reduce_mean(tr_accs),
                tf.reduce_mean(te_losses),
                tf.reduce_mean(te_accs),
                time.time() - t0,
        ))

        debug=False
        if debug:
            _, _, _, auc = plot_roc_curve(0, te_labels, te_preds, lw=1, alpha=0.5,
                    color='gray', ls='-', legend='epoch {}'.format(epoch))
            plt.savefig(os.path.join('./results/', '{}.png'.format(model_type)))

    _, _, _, auc = plot_roc_curve(0, te_labels, te_preds, lw=1, alpha=0.5,
            color='gray', ls='-', legend=None)
    plt.savefig(os.path.join('./results/', '{}_roc.png'.format(model_name)))
    _, _, _, ap = plot_pr_curve(1, te_labels, te_preds, lw=1, alpha=0.5,
            color='gray', ls='-', legend=None)

    plt.savefig(os.path.join('./results/', '{}_pr.png'.format(model_name)))
    aucs.append(auc)
    aps.append(ap)
    
    # sample lengths
    nsamples = int(len(te_labels)*1.0)
    rind = np.random.randint(len(te_labels), size=nsamples)
    micro_preds.append(te_preds[rind])
    micro_targets.append(te_labels[rind])
    print ('=================={}===================='.format(n_exec))


print ('macro auc : {:.3f}, std: {:.3f}'.format(np.mean(aucs), np.std(aucs)))
print ('macro ap : {:.3f}, std: {:.3f}'.format(np.mean(aps), np.std(aps)))
labels = np.concatenate(micro_targets, axis=0)
preds = np.concatenate(micro_preds, axis=0)
plot_roc_curve(0, labels, preds, lw=1, alpha=1.0,
        color='black', ls='-', legend='{}: '.format(model_type))
plt.savefig(os.path.join('./results/', '{}_roc.png'.format(model_name)))
plot_pr_curve(1, labels, preds, lw=1, alpha=1.0,
        color='black', ls='-', legend='{}: '.format(model_type))
plt.savefig(os.path.join('./results/', '{}_pr.png'.format(model_name)))
