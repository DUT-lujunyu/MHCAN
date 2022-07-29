import re
import json, time, random, torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from gensim.models import KeyedVectors

import nltk
from nltk.corpus import stopwords
from src.helper import *

class Datasets(Dataset):
    '''
    The dataset based on Bert.
    '''
    def __init__(self, kwargs, data_name, add_special_tokens=True, not_test=True):
        self.not_test = not_test
        self.data_name = data_name
        # self.max_text_len = max_text_len
        # self.max_title_len = max_title_len
        self.max_tok_len = kwargs["max_tok_len"]
        self.add_special_tokens = add_special_tokens  
        # self.tokenizer = BertTokenizer.from_pretrained('resources/scibert_scivocab_cased')
        self.tokenizer = BertTokenizer.from_pretrained(kwargs["plm_path"])

        self.word2vec_path = kwargs["word2vec_path"]
        with open(data_name, 'r') as f:
            self.data_file = json.load(f)
        self.preprocess_data()

        # 2022.4.29
        # self.word2idx, self.embedding_matrix = self.load_word2vec_matrix('glove/glove_vec.6B.100d.bin')

    def preprocess_data(self):
        print('Preprocessing Data {} ...'.format(self.data_name))
        word2idx, embedding_matrix = load_word2vec_matrix(self.word2vec_path)
        data_time_start=time.time()
        for row in tqdm(self.data_file):
            # ori_title = row['title'].lower()
            # ori_text = row['abstract'].lower()
            ori_title = row['title']
            ori_text = row['abstract']
            # text = self.tokenizer(ori_text, add_special_tokens=self.add_special_tokens,
            #                       max_length=int(self.max_text_len), padding='max_length', truncation=True)
            # title = self.tokenizer(ori_title, add_special_tokens=self.add_special_tokens,
            #                       max_length=int(self.max_title_len), padding='max_length', truncation=True)
            # 2022.4.26 set truncation as "longest_first"
            text_title = self.tokenizer(ori_title, ori_text, max_length=int(self.max_tok_len), padding='max_length',
                                        truncation="longest_first", return_token_type_ids=True, return_attention_mask=True)
            # 2022.5.10 测试集没有标签信息，把level标签都设为0 增加了一个not_test参数
            if self.not_test:            
                level1_dict = self.get_key(1)
                level2_dict = self.get_key(2)
                level3_dict = self.get_key(3)
                row["convert_level1"] = self.convert_one_hot(row["level1"], level1_dict, 21)
                row["convert_level2"] = self.convert_one_hot(row["level2"], level2_dict, 260)
                row["convert_level3"] = self.convert_one_hot(row["level3"], level3_dict, 1272)
                row["convert_levels"] = row["convert_level1"]+row["convert_level2"]+row["convert_level3"]
            else:
                row["convert_level1"] = [0]*21
                row["convert_level2"] = [0]*260
                row["convert_level3"] = [0]*1272
                row["convert_levels"] = row["convert_level1"]+row["convert_level2"]+row["convert_level3"]
            
            # For BERT
            # row['text_idx'] = text['input_ids']
            # row['text_ids'] = text['token_type_ids']
            # row['text_mask'] = text['attention_mask']
            # row['title_idx'] = title['input_ids']
            # row['title_ids'] = title['token_type_ids']
            # row['title_mask'] = title['attention_mask']
            row['text_title_idx'] = text_title['input_ids']
            row['text_title_ids'] = text_title['token_type_ids']
            row['text_title_mask'] = text_title['attention_mask']

            # for glove
            sub_text = re.sub(u"([^\u0030-\u0039\u0041-\u005a\u0061-\u007a])"," ",ori_text)
            text_token = [w for w in sub_text.split() if w not in stopwords.words('english')]
            row["text_token_ids"] = token_to_index(text_token, word2idx)
            sub_title = re.sub(u"([^\u0030-\u0039\u0041-\u005a\u0061-\u007a])"," ",ori_title)
            title_token = [w for w in sub_title.split() if w not in stopwords.words('english')]
            row["title_token_ids"] = token_to_index(title_token, word2idx)
            row["title_text_token_ids"] = row["title_token_ids"] + row["text_token_ids"]
            # pad
            # row["text_token_len"] = len(row["text_token_ids"])
            # row["text_token_ids"] = pad_token_seq(row["text_token_ids"], self.max_text_len)
            # row["title_token_len"] = len(row["title_token_ids"])
            # row["title_token_ids"] = pad_token_seq(row["title_token_ids"], self.max_title_len)
            row["title_text_token_len"] = len(row["title_text_token_ids"])
            row["title_text_token_ids"] = pad_token_seq(row["title_text_token_ids"], self.max_tok_len)

        data_time_end = time.time()
        print("... finished preprocessing cost {} ".format(data_time_end-data_time_start))

    def get_key(self, index):
        path = "data/label/level_" + str(index) + '.json'
        with open(path, 'r+') as file:
            content=file.read()   
        key = json.loads(content)
        return key
    
    def convert_one_hot(self, data, one_hot, length):
        data_ = [0 for i in range(length)]
        for i in data:
            data_[one_hot[i]] = 1
        return data_

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx, corpus=None):
        row = self.data_file[idx]
        sample = {# 'text_idx': row['text_idx'], 'text_ids': row['text_ids'], 'text_mask': row['text_mask'],
                # 'title_idx': row['title_idx'], 'title_ids': row['title_ids'], 'title_mask': row['title_mask'],
                # For BERT
                'text_title_idx': row['text_title_idx'], 'text_title_ids': row['text_title_ids'], 'text_title_mask': row['text_title_mask'], 
                # For GloVe
                # 'text_token_len': row["text_token_len"], 'text_token_ids': row["text_token_ids"], 
                # 'title_token_len': row["title_token_len"], 'title_token_ids': row["title_token_ids"], 
                'title_text_token_len': row["title_text_token_len"], 'title_text_token_ids': row["title_text_token_ids"]
                }
        # For label
        sample['level1'] = row['convert_level1']
        sample['level2'] = row['convert_level2']
        sample['level3'] = row['convert_level3']
        sample['levels'] = row['convert_levels']

        return sample


class Dataloader(DataLoader):
    '''
    A batch sampler of a dataset. 
    '''
    def __init__(self, data, batch_size, shuffle=True, SEED=0):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle 
        self.SEED = SEED
        random.seed(self.SEED)

        self.indices = list(range(len(data))) 
        if shuffle:
            random.shuffle(self.indices) 
        self.batch_num = 0 

    def __len__(self):
        return int(len(self.data) / float(self.batch_size))

    def num_batches(self):
        return len(self.data) / float(self.batch_size)

    def __iter__(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.indices != []:
            idxs = self.indices[:self.batch_size]
            batch = [self.data.__getitem__(i) for i in idxs]
            self.indices = self.indices[self.batch_size:]
            return batch
        else:
            raise StopIteration

    def get(self):
        self.reset() 
        return self.__next__()

    def reset(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle: random.shuffle(self.indices)

def to_tensor(batch):
    '''
    Convert a batch data into tensor
    '''
    # For label
    level1 = torch.tensor([b['level1'] for b in batch])
    level2 = torch.tensor([b['level2'] for b in batch])
    level3 = torch.tensor([b['level3'] for b in batch])
    levels = torch.tensor([b['levels'] for b in batch])
    args = {'level1': level1,'level2': level2, 'level3': level3, 'levels': levels}

    # args['text_idx'] = torch.tensor([b['text_idx'] for b in batch])
    # args['text_ids'] = torch.tensor([b['text_ids'] for b in batch])
    # args['text_mask'] = torch.tensor([b['text_mask'] for b in batch])
    # args['title_idx'] = torch.tensor([b['title_idx'] for b in batch])
    # args['title_ids'] = torch.tensor([b['title_ids'] for b in batch])
    # args['title_mask'] = torch.tensor([b['title_mask'] for b in batch]) 

    # For BERT  
    args['text_title_idx'] = torch.tensor([b['text_title_idx'] for b in batch])
    args['text_title_ids'] = torch.tensor([b['text_title_ids'] for b in batch])
    args['text_title_mask'] = torch.tensor([b['text_title_mask'] for b in batch])

    # For GloVe
    # args['text_token_len'] = torch.tensor([b['text_token_len'] for b in batch])
    # args['text_token_ids'] = torch.tensor([b['text_token_ids'] for b in batch])
    # args['title_token_len'] = torch.tensor([b['title_token_len'] for b in batch])
    # args['title_token_ids'] = torch.tensor([b['title_token_ids'] for b in batch])
    args['title_text_token_len'] = torch.tensor([b['title_text_token_len'] for b in batch])
    args['title_text_token_ids'] = torch.tensor([b['title_text_token_ids'] for b in batch])

    return args