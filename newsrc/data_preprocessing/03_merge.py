import os
import pandas as pd

from constants import DTYPES
from 02_timeslot import get_stats





if __name__ == '__main__':
    input_records = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/02_timeslot.csv'
    input_prd = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/pressure_reducing_device.csv'
    input_pc = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/position_change.csv'
    input_demo = '/home/mike/codes/predicting_pressure_ulcer/preprocessed/demographics.csv'

    # 1. load records
    records = pd.read_csv(input_records, nrows=1000000)

