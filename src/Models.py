from src.ModelLayers import *
from transformers import BertModel
# from src.BERT import BertModel
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class HAM(nn.Module):
    def __init__(self, depth, **kwargs):
        super(HAM, self).__init__()
        self.depth = depth
        self.class_num = kwargs['class_num'][self.depth]
        self.atten_layer = AttentionLayer(depth, **kwargs)
        self.fc_layer = FcLayer(**kwargs, if_local=True)
        self.local_layer = LocalLayer(depth, **kwargs)
        # Graph
        self.if_graph = kwargs["if_graph"]
        self.graph = LabelGraph(**kwargs)
        self.device = kwargs["device"]
        self.adj_path = kwargs["graph_adj_path"]
        self.id_path = kwargs["graph_id_path"]

    def forward(self, att_input, pooled_emb, last_visual=None):
        '''
        input:
            att_input: [batch_size, max_len, vocab_dim]
            pooled_emb: [batch_size, vocab_dim]
            last_visual: [batch_size, max_len]
        
        output:
            logits: [batch_size, class_num]
            scores: [batch_size, class_num]
            visual: [batch_size, sequence_length]
        '''
        if self.depth != 0:
            att_input = torch.mul(att_input, last_visual.unsqueeze(-1))

        if self.if_graph:
            ID = torch.tensor(np.load(self.id_path)).to(torch.long).to(self.device)
            adj = torch.tensor(np.load(self.adj_path)).to(torch.float32).to(self.device)
            label_vectors, graph_vectors = self.graph(ID, adj)
            if self.depth == 0:
                label_vectors = label_vectors[:self.class_num]
            elif self.depth == 1:
                label_vectors = label_vectors[21:self.class_num+21]
            elif self.depth == 2:
                label_vectors = label_vectors[281:self.class_num+281]
            att_weight, att_out = self.atten_layer(att_input, label_vectors) # att_out: [batch_size, vocab_dim]
        else:
            att_weight, att_out = self.atten_layer(att_input) # att_out: [batch_size, vocab_dim]
        
        local_input = torch.cat((pooled_emb, att_out), dim=1)  # local_input: [batch_size, vocab_dim*2]`1`1
        local_fc_out = self.fc_layer(local_input)
        logits, scores, visual = self.local_layer(local_fc_out, att_weight)
        return local_fc_out, logits, scores, visual

class HAMs(nn.Module):
    def __init__(self, **kwargs):
        super(HAMs, self).__init__()
        self.ham_1 = HAM(0, **kwargs)
        self.ham_2 = HAM(1, **kwargs)
        self.ham_3 = HAM(2, **kwargs)

        self.hams_fc = FcLayer(**kwargs, if_out=True)
        self.dropout = nn.Dropout(kwargs["dropout"])
        self.highway_layer = HighwayLayer(**kwargs)
        self.out_fc = nn.Linear(kwargs["fc_hidden_dim"], kwargs["total_classes"])
    
    def forward(self, att_input, pooled_emb):
        att_input = self.dropout(att_input)
        pooled_emb = self.dropout(pooled_emb)
        first_local_fc_out, first_logits, first_scores, first_visual = self.ham_1(att_input, pooled_emb)
        second_local_fc_out, second_logits, second_scores, second_visual = self.ham_2(att_input, pooled_emb, first_visual)
        third_local_fc_out, third_logits, third_scores, third_visual = self.ham_3(att_input, pooled_emb, second_visual)
        
        hams_out = torch.cat([first_local_fc_out, second_local_fc_out, third_local_fc_out], dim=1)
        hams_fc_out = self.hams_fc(hams_out)  # [batch_size, out_dim]
        highway_out = self.dropout(self.highway_layer(hams_fc_out))  # [batch_size, out_dim]

        local_logits = [first_logits, second_logits, third_logits]
        global_logits = self.out_fc(highway_out)  # [batch_size, total_classes]
        global_scores = torch.sigmoid(global_logits)  
        local_scores = torch.cat([first_scores, second_scores, third_scores], dim=1)  # [batch_size, 
        # complete_scores = torch.add(self.alpha * global_scores, (1 - self.alpha) * local_scores)
        return local_logits, global_logits#, complete_scores


# 2022.4.28 Glove + LSTM
class BiLSTM(nn.Module):
    def __init__(self, kwargs, embedding_weight):
        super(BiLSTM, self).__init__()
        self.device = kwargs['device']
        self.vocab_size = embedding_weight.shape[0]
        self.embed_dim = embedding_weight.shape[1]
        # Embedding Layer
        embedding_weight = torch.from_numpy(embedding_weight).float()        
        embedding_weight = Variable(embedding_weight, requires_grad=kwargs["if_grad"])
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, _weight=embedding_weight)
        # Encoder layer
        self.bi_lstm = nn.LSTM(self.embed_dim, kwargs["lstm_hidden_dim"], bidirectional=True, batch_first=True) 

    def forward(self, **kwargs):
        emb = self.embedding(kwargs["title_text_token_ids"].to(self.device)) # [batch, len] --> [batch, len, embed_dim]
        lstm_out, _ = self.bi_lstm(emb)  # [batch, len, embed_dim] --> [batch, len, lstm_hidden_dim*2]
        lstm_out_pool = torch.mean(lstm_out, dim=1)  # [batch, lstm_hidden_dim*2]
        return lstm_out, lstm_out_pool

class Bert_Layer(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Bert_Layer, self).__init__()
        # self.use_cuda = kwargs['use_cuda']
        self.device = kwargs['device']
        self.bert_layer = BertModel.from_pretrained('resources/scibert_scivocab_cased')
        self.dim = kwargs['vocab_dim']

    def forward(self, **kwargs):
        bert_output = self.bert_layer(input_ids=kwargs['text_title_idx'].to(self.device),
                                 token_type_ids=kwargs['text_title_ids'].to(self.device),
                                 attention_mask=kwargs['text_title_mask'].to(self.device))
        return bert_output[0], bert_output[1]


class TwoLayerFFNNLayer(torch.nn.Module):
    '''
    2-layer FFNN with specified nonlinear function
    must be followed with some kind of prediction layer for actual prediction
    '''
    def __init__(self, **kwargs):
        super(TwoLayerFFNNLayer, self).__init__()
        self.output = kwargs['dropout']
        self.input_dim = kwargs['vocab_dim']
        self.hidden_dim = kwargs["fc_hidden_dim"]
        self.out_dim = kwargs["total_classes"]

        self.model = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                   nn.Tanh(),
                                   nn.Linear(self.hidden_dim, self.out_dim))

    def forward(self, att_input, pooled_emb):
        att_input = self.dropout(att_input)
        pooled_emb = self.dropout(pooled_emb)
        return self.model(pooled_emb)

# 2022.04.21 Hierarchical output by level with BERT
class LevelOutLayer(nn.Module):
    '''
    Hierarchical output by level
    '''
    def __init__(self, **kwargs):
        super(LevelOutLayer, self).__init__()
        self.fc_1 = nn.Linear(kwargs['vocab_dim'], kwargs['class_num'][0])
        self.fc_2 = nn.Linear(kwargs['class_num'][0], kwargs['class_num'][1])
        self.fc_3 = nn.Linear(kwargs['class_num'][1], kwargs['class_num'][2])
        
    def forward(self, att_input, pooled_emb):
        first_logits = self.fc_1(pooled_emb)
        second_logits = self.fc_2(first_logits)
        third_logits = self.fc_3(second_logits)
        local_logits = [first_logits, second_logits, third_logits]
        all_logits = torch.cat([first_logits, second_logits, third_logits], dim=1)
        return local_logits, all_logits

# 2022.5.18
class FGM(object):
    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}
 
    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                #print(name)
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)
 
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}