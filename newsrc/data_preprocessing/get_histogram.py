import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import BayesianRidge

from timeslot_and_imputation import multiproc_read_records
from utils import strptime

def plot_histogram(output_fname, dist_a, dist_b, num_bins=20, range=None):
    n, bins, patches = plt.hist(dist_a, num_bins, range=range,
            facecolor='blue', density=True, alpha=0.5, label='case')
    n, bins, patches = plt.hist(dist_b, num_bins, range=range,
            facecolor='red', density=True, alpha=0.5, label='control')

    plt.xlabel('data length (hour)')
    plt.ylabel('Probability')
    plt.title('Case/control histrogram')
    plt.legend()
    plt.savefig(output_fname)

def time_to_sore(records, header):
    soretime = records[0][header.index('SORETIME')]
    if not isinstance(soretime, str):
        return None
    else:
        if soretime == 'N/A':
            return None, None

    t0 = strptime(records[0][header.index('EVENTTIME')])
    td_sore = (strptime(soretime) - t0).total_seconds() / 3600.
    td_all = (strptime(records[-1][header.index('EVENTTIME')]) - t0).total_seconds() / 3600.
    return td_all, td_sore


def get_length(records, header):
    t0 = strptime(records[0][header.index('EVENTTIME')])
    t1 = strptime(records[-1][header.index('EVENTTIME')])

    length = (t1 - t0).total_seconds() / 3600.
    soretime = records[0][header.index('SORETIME')]
    if not isinstance(soretime, str):
        case = False
    else:
        if soretime == 'N/A':
            case = False
        else:
            case = True

    return length, case


if __name__ == '__main__':

    input_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/02_remove_after_event.csv'
    #input_fname = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/01_outlier.csv'
    output_fname = '/home/mike/codes/predicting_pressure_ulcer/visutalization/length_histogram.png'
    records, header = multiproc_read_records(input_fname, 'EVENTTIME', debug=False)
    
    dist_case = []
    dist_control = []
    sore_x, sore_y = [], []
    for key, record in records.items():
        length, case = get_length(record, header)
        if case:
            dist_case.append(length)
        else:
            dist_control.append(length)

        sore = time_to_sore(record, header)
        if sore is not None:
            sore_x.append(sore[0])
            sore_y.append(sore[1])
    
    print ('case avg : {:.4f} | std: {:.4f}'.format(np.mean(dist_case), np.std(dist_case)))
    print ('control avg : {:.4f} | std: {:.4f}'.format(np.mean(dist_control), np.std(dist_control)))

    plot_histogram(output_fname, dist_case, dist_control, range=(0, 720), num_bins=15)

    
    # bayesian regression
    clf = BayesianRidge()
    clf.fit(np.reshape(sore_x, [-1,1]), sore_y)

    x = np.reshape(np.linspace(0, 2000, 200), [-1,1])
    y_mean, y_std = clf.predict(x, return_std=True)

    plt.figure(figsize=(10,10))
    plt.title('Bayesian LR')
    #plt.errorbar(x, y_mean, y_std, color='navy', linewidth=2)
    plt.plot(x, y_mean, 'k-', linewidth=2)
    plt.plot(sore_x, sore_y, 'b.')

    plt.xlabel('total icustay length')
    plt.ylabel('prediction on event time')
    plt.savefig('/home/mike/codes/predicting_pressure_ulcer/visutalization/blr.png')
    import pdb
    pdb.set_trace()







