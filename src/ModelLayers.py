import torch, math
import torch.nn as nn
import torch.nn.functional as F

# 2022.5.11 
class LabelGraph(nn.Module):
    def __init__(self, **kwargs):
        '''
        :param N_label: [int]  标签总个数
        :param dim:  [int]  编码维度
        :param layer_hidden:[int] 图神经网络层数
        :param device: [str]  cuda
        '''
        super(LabelGraph, self).__init__()
        self.N_label = kwargs["total_classes"]
        self.device = kwargs["device"]
        self.dim = 64
        self.fc = nn.Linear(self.dim, kwargs["attention_dim"])
        self.embed_label = nn.Embedding(self.N_label, self.dim).to(self.device)
        self.layer_hidden = kwargs["graph_hidden_layers_num"]
        self.W_label = nn.ModuleList([nn.Linear(self.dim, self.dim).to(self.device)
                                            for _ in range(self.layer_hidden)])
    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_label[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)
    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)
    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, ID, adj, other_feature=None, aggregation='sum',normalize=False):
        '''
        Input:
            ID:   [tensor] [1, N_label]   (1,1553)
            adj:  [tensor] [N_label,N_label]   (1553,1553)
            other_feature: [tensor] [N_label, dim] 标签语义信息 (1553,64)
            aggregation : [str] 得到所有标签的embedding之后，聚合整个图表示的方式 sum or mean
            normalize :[bool]  每一层的label_embedding进入下一层迭代之前，是否进行归一化操作 True or False
        Output:
            label_vectors: [tensor] [N_label,dim]  每个标签的表示 (1553,64)
            graph_vectors: [tensor] [1,dim]    整个图的图表示  (1,64)
        '''
        label_vectors = self.embed_label(ID)  # + other_feature
        adjacencies = adj

        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, label_vectors, l)
            if normalize == True:
                label_vectors = F.normalize(hs, 2, 1)  # normalize.
            else:
                label_vectors = hs

        if aggregation == 'sum':
            graph_vectors = self.sum(label_vectors, self.N_label)
        elif aggregation == 'mean':
            graph_vectors = self.mean(label_vectors, self.N_label)
        
        label_vectors = self.fc(label_vectors)
        return label_vectors, graph_vectors

class SoftAttention(nn.Module):
    def __init__(self):
        super(SoftAttention, self).__init__()
        pass
    
    def forward(self, text_f, label_f):
        """
        soft attention module
        :param text_f -> torch.FloatTensor, (batch_size, K, dim)
        :param label_f ->  torch.FloatTensor, (N, dim)
        :return: label_align ->  torch.FloatTensor, (batch, N, dim)
        """
        att = torch.matmul(text_f, label_f.transpose(0, 1))
        weight_label = F.softmax(att.transpose(1, 2), dim=-1)
        label_align = torch.matmul(weight_label, text_f)
        return label_align

# class AttentionLayer(nn.Module):
#     def __init__(self, depth, **kwargs):
#         super(AttentionLayer, self).__init__()
#         self.vocab_dim = kwargs['vocab_dim']
#         self.attention_dim = kwargs['attention_dim']
#         self.class_num = kwargs['class_num'][depth]
#         self.score = nn.Sequential(
#             nn.Linear(self.vocab_dim, self.attention_dim),
#             nn.Tanh(),
#             nn.Linear(self.attention_dim, self.class_num)
#         )  
#         self.dropout = nn.Dropout(kwargs['dropout'])
        
#     def forward(self, att_input):
#         '''
#         点乘注意力
#         input:
#         att_input: [batch_size, max_len, vocab_dim]

#         output:
#         att_weight: [batch_size, class_num, max_len]
#         att_out: [batch_size, vocab_dim]
#         '''
#         att_weight = self.score(att_input) / math.sqrt(self.vocab_dim)  # [batch_size, max_len, class_num]
#         att_weight = F.softmax(att_weight, dim=1).transpose(1, 2)  # [batch_size, class_num, max_len]
#         att_out = torch.matmul(att_weight, att_input)  # [batch_size, class_num, vocab_dim]
#         att_out = torch.mean(att_out, dim=1).squeeze(1)  # [batch_size, vocab_dim]
#         return att_weight, att_out

class AttentionLayer(nn.Module):
    def __init__(self, depth, **kwargs):
        super(AttentionLayer, self).__init__()
        self.vocab_dim = kwargs['vocab_dim']
        self.attention_dim = kwargs['attention_dim']
        self.class_num = kwargs['class_num'][depth]
        self.if_graph = kwargs["if_graph"]
        self.score = nn.Sequential(
            nn.Linear(self.vocab_dim, self.attention_dim),
            nn.Tanh(), # [batch_size, max_len, dim]
            nn.Linear(self.attention_dim, self.class_num)  # 不引入graph，直接与随机初始化的label embedding做点乘
        ) 
        self.score_graph = nn.Sequential(
            nn.Linear(self.vocab_dim, self.attention_dim),
            nn.Tanh() # [batch_size, max_len, dim]
        )  
        self.dropout = nn.Dropout(kwargs['dropout'])
        
    def forward(self, att_input, graph_emb=None):
        '''
        点乘注意力
        input:
        att_input: [batch_size, max_len, vocab_dim]
        graph_emb: [K, dim]

        output:
        att_weight: [batch_size, class_num, max_len]
        att_out: [batch_size, vocab_dim]
        '''
        if self.if_graph:
            att_weight = self.score_graph(att_input) / math.sqrt(self.vocab_dim)  # [batch_size, max_len, att_dim]
            att_weight = torch.matmul(att_weight, graph_emb.transpose(0, 1))  # [batch_size, max_len, class_num]
        else:
            att_weight = self.score(att_input) / math.sqrt(self.vocab_dim) # [batch_size, max_len, class_num]
        att_weight = F.softmax(att_weight, dim=1).transpose(1, 2)  # [batch_size, class_num, max_len]
        att_out = torch.matmul(att_weight, att_input)  # [batch_size, class_num, vocab_dim]
        att_out = torch.mean(att_out, dim=1).squeeze(1)  # [batch_size, vocab_dim]
        return att_weight, att_out

class FcLayer(nn.Module):
    def __init__(self, if_local = False, if_out = False, **kwargs):
        super(FcLayer, self).__init__()
        self.vocab_dim = kwargs["vocab_dim"]
        if if_local:
            self.vocab_dim = kwargs["vocab_dim"] * 2
        if if_out:
            self.vocab_dim = kwargs["fc_hidden_dim"] * 3 
        self.fc_hidden_dim = kwargs["fc_hidden_dim"]
        self.fc = nn.Sequential(
            nn.Linear(self.vocab_dim, self.fc_hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, emb):
        '''
        原文是 Wx + b
        input: [batch_size, vocab_dim]
        output: [batch_size, fc_hidden_dim]
        '''
        fc_out = self.fc(emb)
        return fc_out


class LocalLayer(nn.Module):
    def __init__(self, depth, **kwargs):
        super(LocalLayer, self).__init__()
        self.class_num = kwargs['class_num'][depth]
        # self.vocab_dim = kwargs["vocab_dim"]
        self.fc_hidden_dim = kwargs['fc_hidden_dim']
        self.get_logits = nn.Linear(self.fc_hidden_dim, self.class_num)

    def forward(self, local_fc_out, att_weight):
        '''
        input:
            local_input: [batch_size, fc_hidden_dim]
            att_weight: [batch_size, class_num, max_len]
        
        output:
            logits: [batch_size, class_num]
            scores: [batch_size, class_num]
            visual: [batch_size, max_len]
        '''
        logits = self.get_logits(local_fc_out)
        scores = torch.sigmoid(logits)  # [batch_size, class_num]

        visual = torch.mul(att_weight, scores.unsqueeze(-1))
        visual = F.softmax(visual, dim = -1)
        visual = torch.mean(visual, dim = 1)
        # print(logits.shape)
        # print(scores.shape)
        # print(visual.shape)
        return logits, scores, visual  


class HighwayLayer(nn.Module):
    def __init__(self, **kwargs):
        """
        Highway Network (cf. http://arxiv.org/abs/1505.00387).
        t = sigmoid(Wx + b); h = relu(W'x + b')
        z = t * h + (1 - t) * x
        where t is transform gate, and (1 - t) is carry gate.
        """
        super(HighwayLayer, self).__init__()
        self.out_dim = kwargs["fc_hidden_dim"]
        self.layers_num = kwargs["highway_layers_num"]
        self.fc_1 = nn.Linear(self.out_dim, self.out_dim)
        self.fc_2 = nn.Linear(self.out_dim, self.out_dim)
    
    def forward(self, input):
        '''
        input = output = [batch_size, out_dim]
        '''
        for i in range(self.layers_num):
            t = torch.sigmoid(self.fc_1(input))
            h = torch.relu(self.fc_2(input)) 
            output = t * h + (1. - t) * input
            input = output
        return output


