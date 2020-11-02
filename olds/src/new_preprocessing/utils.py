import csv

def read_csv(fname, verbose=True):
    count = 0
    with open(fname, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        dict_list = [row for row in reader]
    if verbose:
        print("Reading %s done..."%fname)
    return dict_list

def write_csv(fname, header, dict_list, verbose=True):
    with open(fname, 'w', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, header)
        writer.writeheader()
        writer.writerows(dict_list)
    if verbose:
        print("Writing %s done..."%fname)

def hashing(records, key):
    records_dict = {}
    for record in records:
        r_key = record[key]
        records_dict[r_key] = records_dict.get(r_key, list())
        records_dict[r_key].append(record)
    return records_dict
