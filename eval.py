import re
import json, time, random, torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# from src.helper import *
from sklearn.metrics import f1_score

def preprocess_data(data_name):
    print('Preprocessing Data {} ...'.format(data_name))
    with open(data_name, 'r') as f:
        data_file = json.load(f)
    all_label = []
    for row in tqdm(data_file):
        levels_dict = get_key()
        all_label.append(convert_one_hot(row["pred_labels"], levels_dict, 1553))
    return all_label

def preprocess_data_label(data_name):
    print('Preprocessing Data {} ...'.format(data_name))
    with open(data_name, 'r') as f:
        data_file = json.load(f)
    all_label = []
    for row in tqdm(data_file):
        levels_dict = get_key()
        all_label.append(convert_one_hot(row["levels"], levels_dict, 1553))
    return all_label

def get_key():
    path = "data/label/levels.json"
    with open(path, 'r+') as file:
        content=file.read()   
    key = json.loads(content)
    return key

def convert_one_hot(data, one_hot, length):
    data_ = [0 for i in range(length)]
    for i in data:
        data_[one_hot[i]] = 1
    return data_

if __name__ == '__main__': 
    pred_path = "test.json"
    label_path = "data/verification set.json"
    all_pred = preprocess_data(pred_path)
    all_label = preprocess_data_label(label_path)
    levels_f1 = f1_score(all_pred, all_label, average='micro')   
    print(levels_f1) 
