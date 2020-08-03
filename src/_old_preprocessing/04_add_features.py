import os
from datetime import datetime, timedelta
import pdb

# merge position change, pressure reducing device features
output_root = './datasets'
if not os.path.exists(output_root):
    os.makedirs(output_root)

data_dir = './datasets'
input_fname = os.path.join(data_dir, '03_averaging.csv')
output_fname = os.path.join(data_dir, '04_add_features.csv')


input_dir = './input_datasets'

strptime_m = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M')
strptime_s = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

def read_pc(fname):
    print ('reading position change data..')
    with open(fname, 'r') as f:
        lines = f.readlines()

    out_dict = {}
    for line in lines[1:]:
        icu_stay, time = line.strip().split(',')
        if icu_stay in out_dict.keys():
            out_dict[icu_stay].append(time)
        else:
            out_dict[icu_stay] = [time]

    return out_dict

def read_prd(fname):
    print ('reading pressure reducing device..')
    with open(fname, 'r') as f:
        lines = f.readlines()

    out_dict = {}
    for line in lines[1:]:
        icu_stay, t0, t1 = line.strip().split(',')
        if icu_stay in out_dict.keys():
            out_dict[icu_stay].append([t0, t1])
        else:
            out_dict[icu_stay] = [[t0, t1]]
    return out_dict

pc_dict = read_pc(os.path.join(input_dir,
    'position_change_events_548,224066,227952.csv'))

prd_dict = read_prd(os.path.join(input_dir,
    'pressure_reducing_device_579,224088.csv'))


with open(input_fname, 'r') as f:
    lines = f.readlines()
    output_lines = []
    for num_line, line in enumerate(lines):
        if num_line==0:
            # header
            headers = line.strip() + ',Position Change,Pressure Reducing Device\n'
            h = headers.strip().split(',')
            output_lines.append(headers)
        else:
            # hourly data 
            line = line.strip().split(',')
            icu_stay = line[h.index('ICUSTAY_ID')]
            ctime = line[h.index('TIMESTAMP')]
            
            # add position change feature
            pc_flag = 0
            if icu_stay in pc_dict.keys():
                for _p in pc_dict[icu_stay]:
                    td = (strptime_s(ctime) - strptime_m(_p)).total_seconds()

                    if td >=0 and td < 60*60:
                        pc_flag = 1

            prd_flag = 0
            if icu_stay in prd_dict.keys():
                for _p in prd_dict[icu_stay]:

                    td0 = (strptime_s(ctime) - strptime_s(_p[0])).total_seconds()
                    td1 = (strptime_s(ctime) - strptime_s(_p[1])).total_seconds()
                    if td0 >= 0 and td1 < 0:
                        prd_flag = 1
                
            out = line + [str(pc_flag), str(prd_flag) + '\n']
            output_lines.append(','.join(out))
                
#        debug=True
#        if debug and num_line == 3000:
#            break

        if num_line % 10000 == 0:
            print ('{:10d} / {:10d} processed'.format(num_line, len(lines)))

with open(output_fname, 'w') as f:
    f.writelines(output_lines)
