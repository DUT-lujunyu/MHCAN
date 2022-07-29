import torch
import numpy as np
import os
from os import path

dataset = 'NLPCC2022Task5Track1'  # 数据集

kwargs = {
    # Path
    "train_data" : 'data/training set_modified.json',                                # 训练集
    "dev_data" : 'data/verification set_modified.json',                                    # 验证集
    "train_data" : 'data/test_training data.json',                # Test training set (2000)
    "dev_data" : 'data/test_verification data.json',              # Test verification set (500)        
    "test_data" : 'data/Testing dataset_modified.json',
    "ori_test_data" : 'data/Testing dataset.json',
    "test_data" : 'data/test_v2_modified.json',                                 # 测试集
    "ori_test_data" : 'data/test_v2.json',   
    "result_path" : 'results/',
    "checkpoint_path" : 'checkpoints/',
    "filename" : "ronghe/label+train+0.6687ckp-HAM-ML-450_D-0.5_B-16_AD-256_HD-200_LD-1536_E-100_Lr-1e-05_HL-1_SEED-0_GA-[0.45, 0.5, 0.5]_MA-1_L-finallyfinally12_alpha-0.5_NoL1-1_Glo-False_topk-[7, 8, 7]-BEST.tar",
    
    "key1_2_path" : 'data/label/key1_reflect_key2.json',
    "key2_3_path" : 'data/label/key2_reflect_key3.json',
    "graph_adj_path" : 'data/graph/adj.npy',
    "graph_id_path" : 'data/graph/ID.npy',
    "label_index_path1" : "data/label/level_1.json",
    "label_index_path2" : "data/label/level_2.json",
    "label_index_path3" : "data/label/level_3_.json",
    "label_num_path1" : "data/label/num_key1.json",
    "label_num_path2" : "data/label/num_key2.json",
    "label_num_path3" : "data/label/num_key3.json",

    "plm_path" : 'resources/1scibert_scivocab_cased',
    "word2vec_path" : 'glove/glove.6B.300d.bin',

    "use_cuda" : True,
    "device" : 'cuda:1',   # 设备

    # Experiment
    "SEED" : 0,
    "dropout" : 0.5 ,                                            # 随机失活
    "epochs" : 20,                                             # epoch数 
    "batchsize" : 16,                                            # mini-batch大小
    "lr" : 1e-5,                                                # 学习率  transformer:5e-4 
    "scheduler" : False,                                          # 是否学习率衰减
    "num_warm" : 0,                                               # The epoch number at which to start saving scores
    "max_text_len" : 150,  
    "max_title_len" : 50,
    "max_tok_len" : 200,                                              # 每条样本处理成的长度(短填长切)
    "total_classes" :  1553,
    "class_num" :  [21, 260, 1272], 
    "vocab_dim" :  768,   # embedding with bert (sci-bert) 

    # Model parameter
    "attention_dim" : 256,
    "fc_hidden_dim" : 200,
    "lstm_hidden_dim" : 150,
    "local_input_dim" : 1536, # vocab_dim*2
    "out_dim" : 600,
    "highway_layers_num" : 1,
    "graph_hidden_layers_num" : 2,
    "if_global" : False, # 是否引入全局logits
    "if_graph" : True,  # 是否引入label embedding
    "if_grad" : True,  # 是否训练 embedding layer (BERT or glove) 

    # Loss and result
    "loss_func_name" : "CBloss-ntr",
    "alpha" : 0.5,  # hyp for rebalanced loss
    "train_num" : 90000,  # hyp for  rebalanced loss
    "if_label_mask" : True,                                       # 是否使用 label_mask
    "threshold" : [0.45, 0.5, 0.5],                               # 门限
    "top_k" : [7, 8, 7],                                          # 选择top-k个预测结果
    "score_key" : 'levels_F1_macro'                              # Score key for selecting the best model                
}
