from ast import arg
from importlib import import_module
from src.losses import *
from src.datasets import *
from src.utils import *
from src.Models import *
from src.helper import *
import torch.optim as optim
import torch.nn as nn
import transformers

import argparse
transformers.logging.set_verbosity_error()  # 不打印 transformers 的警告

from config import kwargs

# 命令行输入参数
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--embedding', default='bert', type=str, help='bert or glove')
parser.add_argument('--model', default='NN', type=str, help='choose a model: NN, HAM')
parser.add_argument('--start_train', default=True, type=str, help='training or evaluation')
args = parser.parse_args()

if args.embedding == "glove":
    kwargs["vocab_dim"] = 300

# 2022.4.22 loss weight
class_freq_1 = get_label_weight(kwargs["label_index_path1"], kwargs["label_num_path1"], kwargs["class_num"][0]) # 每个label的样本数列表
class_freq_2 = get_label_weight(kwargs["label_index_path2"], kwargs["label_num_path2"], kwargs["class_num"][1])
class_freq_3 = get_label_weight(kwargs["label_index_path3"], kwargs["label_num_path3"], kwargs["class_num"][2])



if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(kwargs["SEED"])
    torch.cuda.manual_seed_all(kwargs["SEED"])
    np.random.seed(kwargs["SEED"])
    torch.backends.cudnn.deterministic = True

    # 构建数据集
    if args.start_train:
        # 第一次训练
        # trn_data = Datasets(kwargs, kwargs["train_data"])
        # dev_data =  Datasets(kwargs, kwargs["dev_data"])
        # torch.save({
        #     'train_data': trn_data,
        #     }, 'checkpoints/train_data.tar')
        # torch.save({
        #     'dev_data': dev_data,
        #     }, 'checkpoints/dev_data.tar')
        
        # 后续训练，直接加载
        checkpoint = torch.load('checkpoints/train_data.tar')
        trn_data = checkpoint['train_data']  
        checkpoint = torch.load('checkpoints/dev_data.tar')
        dev_data = checkpoint['dev_data']         

        print('The size of the Training dataset: {}'.format(len(trn_data)))
        print('The size of the Validation dataset: {}'.format(len(dev_data)))
        trn_dataloader = Dataloader(trn_data,  batch_size=int(kwargs["batchsize"]), SEED=kwargs["SEED"])
        dev_dataloader = Dataloader(dev_data, batch_size=int(kwargs["batchsize"]), SEED=kwargs["SEED"])
    else:
        # 第一次测试，保存Datasets
        # test_data =  Datasets(kwargs, test_data, max_text_len=max_text_len, max_title_len=max_title_len, max_tok_len=max_tok_len, not_test=False)
        # torch.save({
        #     'test_data': test_data,
        #     }, 'checkpoints/data1.tar')

        # 之后测试时直接加载Dataset，提升速度
        checkpoint = torch.load('checkpoints/data1.tar')
        test_data = checkpoint['test_data']
        print('The size of the Test dataset: {}'.format(len(test_data)))
        test_dataloader = Dataloader(test_data, batch_size=int(kwargs["batchsize"]), SEED=kwargs["SEED"])

    #  初始化 embedding 和 model
    if args.embedding == "glove": 
        word2idx, embedding_matrix = load_word2vec_matrix(kwargs["word2vec_path"])
        embed_model = BiLSTM(kwargs, embedding_matrix)
    else:
        embed_model = Bert_Layer(**kwargs)

    if args.model == "NN":
        # model = TwoLayerFFNNLayer(**kwargs)
        model = LevelOutLayer(**kwargs)
        model_name = '{}-NN_ML-{}_D-{}_B-{}_HD-{}_E-{}_Lr-{}_SEED-{}_GA-{}_MA-{}_L-{}_alpha-{}_topk-{}'.format(args.embedding, kwargs["max_tok_len"], kwargs["dropout"], kwargs["batchsize"],
                                                                                        kwargs["fc_hidden_dim"], kwargs["epochs"], kwargs["lr"], kwargs["SEED"], kwargs["threshold"], 
                                                                                        kwargs["if_label_mask"], kwargs["loss_func_name"], kwargs["alpha"], kwargs["top_k"])
    else:
        model = HAMs(**kwargs)
        model_name = '{}-HAM-ML-{}_D-{}_B-{}_AD-{}_HD-{}_LD-{}_E-{}_Lr-{}_HL-{}_SEED-{}_GA-{}_MA-{}_L-{}_alpha-{}_topk-{}'.format(args.embedding, kwargs["max_tok_len"], kwargs["dropout"], kwargs["batchsize"], kwargs["attention_dim"],
                                                                                        kwargs["fc_hidden_dim"], kwargs["local_input_dim"], kwargs["epochs"], kwargs["lr"], kwargs["highway_layers_num"], kwargs["SEED"], kwargs["threshold"], 
                                                                                        kwargs["if_label_mask"], kwargs["loss_func_name"], kwargs["alpha"], kwargs["top_k"])

    if kwargs['use_cuda']:
        embed_model = embed_model.to(kwargs['device'])
        model = model.to(kwargs['device'])

    embed_optimizer = optim.AdamW(embed_model.parameters(), lr=kwargs["lr"])
    model_optimizer = optim.AdamW(model.parameters(), lr=kwargs["lr"])
    fgm = FGM(embed_model, epsilon=1, emb_name='word_embeddings.')

    # 2022.4.22 re-balanced loss
    loss_func1 = get_loss_func(kwargs["loss_func_name"], class_freq_1, kwargs["train_num"], kwargs["alpha"])
    loss_func2 = get_loss_func(kwargs["loss_func_name"], class_freq_2, kwargs["train_num"])
    loss_func3 = get_loss_func(kwargs["loss_func_name"], class_freq_3, kwargs["train_num"])
    loss_func4 = nn.BCEWithLogitsLoss()

    if args.start_train:
        max_lst = []
        start_time = time.time()
        print("Start training...")
        train(kwargs, max_lst, model_name, embed_model, model, loss_func1, loss_func2,
            loss_func3, loss_func4, embed_optimizer, model_optimizer, fgm, trn_dataloader, dev_dataloader, threshold=kwargs['threshold'])
        print("[{}] total runtime: {:.2f} minutes".format(model_name, (time.time() - start_time)/60.))
    else:
        start_time = time.time()
        print("Start evaluating...")
        checkpoint = torch.load(kwargs["filename"])
        embed_model.load_state_dict(checkpoint['embed_model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        _, pred = eval(kwargs, embed_model, model, loss_func1, loss_func2, loss_func3, loss_func4, test_dataloader, threshold=kwargs['threshold'], data_name='TEST')
        get_results(pred, kwargs["ori_test_data"])
        print("[{}] total runtime: {:.2f} minutes".format(model_name, (time.time() - start_time)/60.))