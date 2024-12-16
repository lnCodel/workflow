""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
#import torch_geometric as tg
from utils import gen_adj_mat_tensor, cal_adj_mat_parameter
import numpy as np

def gen_tr_adj_mat(data_tr_list, adj_parameter): #每一层GCN后更新adj_list
    adj_metric = "cosine"  # cosine distance
    adj_train_list = []
    adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list,
                                                   adj_metric)  # 计算到node_num*edges_per_node的数字大小，
    # 接下来所有大于等于这个数字的余弦距离都留下
    adj_train_list= gen_adj_mat_tensor(data_tr_list, adj_parameter_adaptive, adj_metric)
        # adj_train_list中全是非0元素了

    return adj_train_list

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        #网络的权值初始化,
        if m.bias is not None:
           m.bias.data.fill_(0.0)

           

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        #super()函数可以隐式地将子类(sublass)里的method，与父类(superclass)里的method进行关联。
        # 这样的好处在于我们不用在子类里显式地重新创建父类method里的属性。
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        #nn.Parameter，主要作用是作为nn.Module中的可训练参数使用
    #将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
    # (net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            #偏置单元（bias unit），在有些资料里也称为偏置项（bias term）或者截距项（intercept term），它其实就是函数的截距，
        nn.init.xavier_normal_(self.weight.data)
        #xavier高斯初始化
        if self.bias is not None:
            self.bias.data.fill_(0.0)
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        #torch.mm(a, b)是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵。
        output = torch.sparse.mm(adj, support)
        # torch.sparse.mm:执行稀疏矩阵 mat1 和 稠密矩阵 mat2 的矩阵乘法.
        #类似于 torch.mm(), 如果 mat1 是一个n*m tensor, mat2 是一个 m*p tensor, 输出将会是  稠密的 n*p tensor.
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    

class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        # self.gc2 = GraphConvolution(in_dim+hgcn_dim[0], hgcn_dim[1])
        #前面所有层全部相加
        # self.gc3 = GraphConvolution(in_dim+hgcn_dim[0]+hgcn_dim[1], hgcn_dim[2])
        #只加前面一层的输出
        # self.gc3 = GraphConvolution(hgcn_dim[0] + hgcn_dim[1], hgcn_dim[2])
        #当前层
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout

    def forward(self,adj_parameter, x, adj):
        #beginning of first layer||||||beginning of first layer|||||beginning of first layer
        x1=x #输入网络的顶点特征
        # adj=adj*adj
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        adj = gen_tr_adj_mat(x, adj_parameter)
        #2-hub neigubors
        # adj=adj*adj
        #leaky_relu:激活函数
        x = F.dropout(x, self.dropout, training=self.training)
        x2=x#当前层输出
        # x2 = torch.cat((x1, x),1) #将input和H（1）串联
        # x2_now=x    #第一层的输出
        x = x2
        # beginning of second layer||||||beginning of second layer|||||beginning of second layer
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        adj = gen_tr_adj_mat(x, adj_parameter)
        #3-hub neighbors
        # adj = adj * adj
        x = F.dropout(x, self.dropout, training=self.training)
        x3=x#当前层输入
        # x3 = torch.cat((x2, x),1) ##将input和H（1）和H（2）串联
        # x3 = torch.cat((x2_now, x), 1)  ##将H（1）和H（2）串联
        # x3_now=x
        x = x3
        # beginning of third layer||||||beginning of third layer|||||beginning of third layer
        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)
        #x4=torch.cat((x3,x),1)#加上前面所有层的输出
        # x4=torch.cat((x3_now,x),1)#指甲上前一层
        x4=x#最后一层不加上之前的输出
        x=x4
        return x


class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        #nn.Linear()：用于设置网络中的全连接层，需要注意的是全连接层的输入与输出都是二维张量
        #nn.Sequential:一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            #nn.Linear()：用于设置网络中的全连接层，需要注意的是全连接层的输入与输出都是二维张量
            nn.LeakyReLU(0.25),
            #max(0.25x,x)
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)
        
    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
            #torch.sigmoid():这是一个方法，包含了参数和返回值。 输入为tensor 转变后为tensor
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),(-1,pow(self.num_cls,2),1))
                                       #in_list[0].unsqueeze(-1)： 在最后一维增加一个维度 in_list[1].unsqueeze(1) 在在第二维增加一个维度
        #torch.matmul((44,2,1)维度,(44,1,2)维度）->(44,2,2)维度 tensor
        # torch.reshape（  ...,-1,...   ）-1代表n n=tensor的长度/第一个参数\
        #x tensor(44,4,1)
        for i in range(2,num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)),(-1,pow(self.num_cls,i+1),1))
        #x tensor(44,8,1)
        vcdn_feat = torch.reshape(x, (-1,pow(self.num_cls,num_view)))
        #vcdn_feat  tensor(44,8)
        output = self.model(vcdn_feat)
        #output tensor(44,2)
        return output


# class Cheb_GCN(nn.Module):
#     def __init__(self, input_dim, hidden, num_class, dropout, training, edge_dropout = 0, Cheb_level = 3):
#         super(Cheb_GCN,self).__init__()
#         self.gc1 = tg.nn.ChebConv(input_dim, hidden, Cheb_level, normalization='sym', bias=False)
#         self.gc2 = tg.nn.ChebConv(hidden, num_class, Cheb_level, normalization='sym', bias=False)
#         self.dropout = dropout
#         self.training = training
#         self.edge_dropout = edge_dropout
#         self.model_init()
#
#     def model_init(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 m.weight.requires_grad = True
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#                     m.bias.requires_grad = True
#
#     def forward(self, x, edge_index, edge_weight):
#         if self.edge_dropout > 0:
#             if  self.training:
#                 # one_mask = torch.ones([edge_weight.shape).to('cpu')
#                 one_mask = torch.ones([edge_weight.shape[0], 1]).to('cpu')#.cuda()
#                 self.drop_mask = F.dropout(one_mask, self.edge_dropout, True)
#                 self.bool_mask = torch.squeeze(self.drop_mask.type(torch.bool))
#                 edge_index = edge_index[:, self.bool_mask]
#                 edge_weight = edge_weight[self.bool_mask]
#
#         x = F.relu(self.gc1(x, edge_index, edge_weight))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, edge_index, edge_weight)
#         return x

    
def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dopout=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = GCN_E(dim_list[i], dim_he_list, gcn_dopout)
        ##下面一行表示不加，只用当前层
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1], num_class)
        ##下面一行表示只加上前一层
        # model_dict["C{:}".format(i + 1)] = Classifier_1(dim_he_list[-2]+dim_he_list[-1], num_class)
        ##下面一行所有层全部相加
        # model_dict["C{:}".format(i + 1)] = Classifier_1(dim_list[i]+np.sum(dim_he_list), num_class)
        # print("hello")
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()), 
                lr=lr_e)
        #torch.optim.Adam:
        # params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
        #lr (float, 可选) – 学习率（默认：1e-3）
        #betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
            #betas = （beta1，beta2）
            #beta1：一阶矩估计的指数衰减率（如 0.9）
            #beta2：二阶矩估计的指数衰减率（如 0.999）。该超参数在稀疏梯度（如在 NLP 或计算机视觉任务中）中应该设置为接近 1 的数。
        #eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
        #weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict