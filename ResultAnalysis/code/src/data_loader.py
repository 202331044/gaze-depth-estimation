import numpy as np

keys = {"sbj", "y", "distance", "diameter", "x"}

def read_file(filename):
    with open(filename, 'r') as file:
        return [row.strip().split(",") for row in file]

def make_train_test_data_set(input_file, train_idx):
    train = {k : [] for k in keys}
    test = {k : [] for k in keys}

    for row in input_file[1:]:
        sbj_idx, depth, x1, x2 = row

        sbj = int(sbj_idx)
        y, frame_idx = int(depth[0:3]), int(depth[-1])
        distance, diameter = float(x1), float(x2)

        is_train = frame_idx in train_idx
        target = train if is_train else test

        target['sbj'].append([sbj])
        target['y'].append([y])
        target['distance'].append([distance])
        target['diameter'].append([diameter])
        target['x'].append([distance, diameter])
    
    return  train, test

def make_all_data_set(input_file, train_idx):
    all_data = {k : [] for k in keys}

    for row in input_file[1:]:
        sbj_idx, depth, x1, x2 = row

        sbj = int(sbj_idx)
        y = int(depth[0:3])
        distance, diameter = float(x1), float(x2)
        
        all_data['sbj'].append([sbj])
        all_data['y'].append([y])
        all_data['distance'].append([distance])
        all_data['diameter'].append([diameter])
        all_data['x'].append([distance, diameter])
    
    return all_data