import torch
import json
import os
import numpy as np
from collections import OrderedDict
from gensim.models import KeyedVectors

# 2022.4.21 设置阈值
def get_threshold(pred, threshold=0.5, top_indices=None):
    threshold = torch.tensor([threshold])
    results = (pred > threshold).float()*1
    if top_indices:
        for i in range(results.shape[0]):
            for j in range(results.shape[1]):
                if j not in top_indices[i]:
                    results[i][j] = 0            
    return results

# 2022.4.21 利用上下级标签的映射，根据上级预测结果生成下一级的标签掩码
def get_mask(path, preds, next_level_len):
    with open(path, 'r+') as file:
        content=file.read()   
    reflect = json.loads(content)#将json格式文件转化为python的字典文件
    mask = torch.zeros(preds.shape[0], next_level_len)
    no_mask = torch.ones(preds.shape[0], next_level_len)
    preds_true = []  # 存储预测标签为1的索引

    row = 0
    for pred in preds:
        pred_true = []  
        for i in range(preds.shape[1]):
            if pred[i] == 1 and str(i) in reflect:
                pred_true.extend(reflect[str(i)])
        preds_true.append(pred_true)
        index = (torch.LongTensor([row for i in range(len(pred_true))]),torch.LongTensor(pred_true))#生成索引
        value = torch.Tensor([1 for i in range(len(pred_true))]) #生成要填充的值
        mask.index_put_(index, value)    
        row += 1
    return mask

# 2022.4.22 读取各个label的数量，为损失函数赋予权值
def read_json(path):
    with open(path, 'r+') as file:
        content=file.read()   
    key = json.loads(content)
    return key

def get_label_weight(label_index_path, label_num_path, length):
    label_weight = [0 for i in range(length)]
    label_index = read_json(label_index_path)
    label_num = read_json(label_num_path)
    for key in label_index:
        label_weight[label_index[key]] = label_num[key]
    return label_weight

# 2022.4.27 ready for glove/word2vec embedding
# most code refer to the following link:
# https://github.com/RandolphVI/Hierarchical-Multi-Label-Text-Classification/blob/c0594efdbd00638925d36bb2b75d330c85788209/utils/data_helpers.py#L15

def load_word2vec_matrix(word2vec_file):
    """
    Get the word2idx dict and embedding matrix.
    Args:
        word2vec_file: The word2vec file.
    Returns:
        word2idx: [dict] The word2idx dict.
        embedding_matrix: [array] The word2vec model matrix.
    Raises:
        IOError: If word2vec model file doesn't exist.
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist. ")

    print("Load pre-train embedding...")
    if word2vec_file[-4:] == ".bin": 
        wv = KeyedVectors.load(word2vec_file, mmap='r')
    elif word2vec_file[-4:] == ".txt": 
        wv = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
    else:
        raise IOError("[Error] The word2vec file is not .txt or .bin. ")
    # wv = api.load("glove-wiki-gigaword-100")

    word2idx = OrderedDict({"_UNK": 0})
    embedding_size = wv.vector_size
    for k, v in wv.vocab.items():
        word2idx[k] = v.index + 1
    vocab_size = len(word2idx)

    embedding_matrix = np.zeros([vocab_size, embedding_size])
    for key, value in word2idx.items():
        if key == "_UNK":
            embedding_matrix[value] = [0. for _ in range(embedding_size)]
        else:
            embedding_matrix[value] = wv[key]
    return word2idx, embedding_matrix

def token_to_index(x, word2idx):
    result = []
    for item in x:
        if item not in word2idx.keys():
            result.append(word2idx['_UNK'])
        else:
            word_idx = word2idx[item]
            result.append(word_idx)
    return result

def pad_token_seq(x, length):
    if len(x) < length:
        x = x + [0 for i in range(length - len(x))]
    else:
        x = x[:length]
    return x

# 2022.5.1 最大概率标签
def get_label_topk(scores, top_num=5):
    """
    Get the predicted labels based on the topK number.
    Args:
        scores: [tensor] The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        The predicted labels
    """
    predicted_labels = []
    predicted_scores = []
    scores = scores.detach().numpy()
    scores = np.ndarray.tolist(scores)
    for score in scores:
        score_list = []
        index_list = np.argsort(score)[-top_num:]
        index_list = index_list[::-1]
        for index in index_list:
            score_list.append(score[index])
        predicted_labels.append(np.ndarray.tolist(index_list))
        predicted_scores.append(score_list)
    return predicted_labels


# 2022.5.10 输出结果函数
def get_key(index):
    path = "data/label/level_" + str(index) + '.json'
    with open(path, 'r+') as file:
        content=file.read()   
    key = json.loads(content)
    return key

def index_to_char(pred, dict, start, end):
    ture_pred = []
    pred = pred[start:end]
    for index, i in enumerate(pred):
        if i == 1.0:
            ture_pred.append(dict[index]) 
    return ture_pred

def get_results(pred, ori_test_data):
    """
    Get results file
    """
    title_list = []
    abstract_list = []
    pred_list = []
    data_new = []

    level1_dict = get_key(1)
    level1_dict = dict(zip(level1_dict.values(),level1_dict.keys()))
    level2_dict = get_key(2)
    level2_dict = dict(zip(level2_dict.values(),level2_dict.keys()))
    level3_dict = get_key(3)
    level3_dict = dict(zip(level3_dict.values(),level3_dict.keys()))

    for p in pred:
        ture_pred1 = index_to_char(p, level1_dict, 0, 21)
        ture_pred2 = index_to_char(p, level2_dict, 21, 281)
        ture_pred3 = index_to_char(p, level3_dict, 281, 1553)
        ture_pred = ture_pred1 + ture_pred2 + ture_pred3
        pred_list.append(ture_pred)
        
    with open(ori_test_data, 'r') as f:
        data = json.load(f)

    for d in data:
        title_list.append(d.get("title"))
        abstract_list.append(d.get("abstract"))

    for i in range(0, len(data)):
        data_new.append(dict(title=title_list[i], abstract=abstract_list[i], pred_labels=pred_list[i]))
    with open("results/test_4.json", 'w') as f:
        json.dump(data_new, f)

