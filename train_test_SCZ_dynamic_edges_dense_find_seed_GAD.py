""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,confusion_matrix,roc_curve
import torch
import torch.nn.functional as F
from models_dynamic_edges_dense_k_hup_neighbors import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter,save_model_dict
import pandas as pd
import csv
import xlrd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io
cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(data_folder, view_list,fMRI_type,data_subfolder,data_num,atlas):
    if data_subfolder=='GAD':
        label_file="GAD_label.csv"
    if data_subfolder=="COBRE":
        label_file = "COBRE_label.csv"
    source_folder=os.path.join(data_folder,data_subfolder,data_num)
    label_csv=pd.read_csv(os.path.join(source_folder,label_file))
    if fMRI_type == 1:
        data_fMRI_csv = pd.read_csv(
            os.path.join(source_folder,"fMRI",atlas,"ReHo_VMHC_fALFF_ALFF","VMHC","VMHC_features.csv"))
    elif fMRI_type == 2:
        data_fMRI_csv = pd.read_csv(
            os.path.join(source_folder, "fMRI", atlas, "ReHo_VMHC_fALFF_ALFF", "ReHo", "ReHo_features.csv"))
    elif fMRI_type == 3:
        data_fMRI_csv = pd.read_csv(
            os.path.join(source_folder, "fMRI", atlas, "ReHo_VMHC_fALFF_ALFF", "ALFF", "ALFF_features.csv"))
    elif fMRI_type == 4:
        data_fMRI_csv = pd.read_csv(
            os.path.join(source_folder, "fMRI", atlas, "ReHo_VMHC_fALFF_ALFF", "fALFF", "fALFF_features.csv"))
    elif fMRI_type == 5:
        data_fMRI_csv = pd.read_csv(
            os.path.join(source_folder, "fMRI", atlas, "fc_fisher",  "feature_fc.csv"))

    data_sMRI_csv = pd.read_csv(
        os.path.join(source_folder, "sMRI", atlas, "GMV", "GMV_features.csv"))
    # data_sMRI_csv = pd.read_csv(
    #     os.path.join(source_folder, "fMRI", "aal_116","fc_fisher",  "feature_fc.csv"))

    data_DTI=[]
    if data_subfolder=="GAD":
        data_DTI_csv = pd.read_csv(
            os.path.join(source_folder, "DTI", "AllAtlasResults", "WMlabelResults_FA.csv"))
        data_DTI = data_DTI_csv.values[:, 1:]


    label = np.array(label_csv.values[:, 1:])  # 所有数据的label
    data_fMRI = data_fMRI_csv.values[:, 1:]
    data_sMRI = data_sMRI_csv.values[:, 1:]
    ids = data_fMRI_csv.values[:, 0:1]
    ids = np.reshape(ids, (-1,))
    ids = ids.tolist()
    return label,data_fMRI,data_sMRI,data_DTI,ids



def gen_trte_adj_mat(data_tr_list, data_trteva_list, trteva_idx, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter[i], data_tr_list[i], adj_metric) #计算到node_num*edges_per_node的数字大小，
        #接下来所有大于等于这个数字的余弦距离都留下
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        #adj_train_list中全是非0元素了
        adj_test_list.append(gen_test_adj_mat_tensor(data_trteva_list[i], trteva_idx, adj_parameter_adaptive, adj_metric))
    return adj_train_list, adj_test_list


def gen_tr_adj_mat(data_tr_list, adj_parameter):
    adj_metric = "cosine"  # cosine distance
    adj_train_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i],
                                                       adj_metric)  # 计算到node_num*edges_per_node的数字大小，
        # 接下来所有大于等于这个数字的余弦距离都留下
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        # adj_train_list中全是非0元素了

    return adj_train_list


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, adj_parameter,train_VCDN=True):
    label=label.reshape(len(label,))#[44,1]->44
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    #交叉熵损失函数
    #交叉熵主要是用来判定实际的输出与期望的输出的接近程度
    for m in model_dict:
        model_dict[m].train()
        #如果模型中有BN层(Batch Normalization）和 Dropout，需要在训练时添加model.train()
        #model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)].zero_grad()
        #zero_grad:把梯度置零，也就是把loss关于weight的导数变成0(清空过往梯度)
        ci_loss = 0
        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](adj_parameter[i],data_list[i],adj_list[i]))
        #model(a,b)等价于model.forward(a,b)
        ci_loss = torch.mean(torch.mul(criterion(ci, label),sample_weight))
        #torch.mul:对两个张量逐元素乘法
        #criterion:上文定义过的交叉熵损失函数
        #label:tensor(44,) ci:tensor(44,2)  sample_weight:tensor(44,1)
        #criterion(ci, label):tensor(44,)
        #torch.mul(criterion(ci, label),sample_weight):tensor(44,44)
        #ci_loss:tensor:()
        ci_loss.backward()
        #反向传播，计算当前梯度
        optim_dict["C{:}".format(i+1)].step()
        #根据梯度更新网络参数
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item() #把loss的数值取出来
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](adj_parameter[i],data_list[i],adj_list[i])))
        c = model_dict["C"](ci_list)    
        c_loss = torch.mean(torch.mul(criterion(c, label),sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
    
    return loss_dict

def gen_tr_adj_mat_make_new_graph(data_tr_list, adj_parameter): #每一层GCN后更新adj_list
    adj_metric = "cosine"  # cosine distance
    adj_train_list = []
    adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list,
                                                   adj_metric)  # 计算到node_num*edges_per_node的数字大小，
    # 接下来所有大于等于这个数字的余弦距离都留下
    adj_train_list= gen_adj_mat_tensor(data_tr_list, adj_parameter_adaptive, adj_metric)
        # adj_train_list中全是非0元素了

    return adj_train_list
def test_epoch(data_list, adj_list, te_idx, model_dict,adj_parameter):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    adj_list_next = []
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](adj_parameter[i],data_list[i],adj_list[i])))
    if num_view >= 2:
        c = model_dict["C"](ci_list)
    else:
        c = ci_list[0]
    c1 = c[te_idx,:]
    prob1 = F.softmax(c1, dim=1).data.cpu().numpy()
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    for i in range(num_view):
        adj_list_next.append(gen_tr_adj_mat_make_new_graph(ci_list[i], adj_parameter[i]))
    return prob,prob1, adj_list_next

def test_data(fMRI, sMRI,DTI,types):
    data_fMRI_csv = pd.read_csv(fMRI)
    data_sMRI_csv = pd.read_csv(sMRI)
    data_fMRI = data_fMRI_csv.values[:, 1:]
    data_sMRI = data_sMRI_csv.values[:, 1:]
    ids = data_fMRI_csv.values[:, 0:1]

    if types == "GAD":
        data_DTI_csv = pd.read_csv(DTI)
        data_DTI = data_DTI_csv.values[:, 1:]
        return data_fMRI, data_sMRI, data_DTI,ids
    else:
        return data_fMRI, data_sMRI, None,ids
def save_Intermediate_results(v,adj_list,num,base_path="./xmn2/results/", type="original", type_shujv="",subject=""):
    dit = {}
    dit1 = {}
    for i in range(len(v)):
        dit[v[i]] = i
        dit1[i] = v[i]
    path = os.path.join(base_path,type_shujv)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, subject)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, str(num + 1))
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path,type)
    if not os.path.exists(path):
        os.makedirs(path)
    sa = ["fMRI","sMRI","DTI"]
    for i in range(len(adj_list)):
        G = nx.Graph()
        Matrix = adj_list[i].detach().cpu().to_dense().numpy()
        v1, Matrix,dict1,colors = make_graph(adj=Matrix, nodes=v,dic=dit,dic1=dit1,ak=i, subject=subject)
        G.add_nodes_from(v1)
        G.add_edges_from([(v1[i], v1[j]) for i in range(len(Matrix)) for j in range(len(Matrix)) if Matrix[i][j]])
        w = nx.get_edge_attributes(G, 'weight')
        if not w:
            w = {(u, v): format(Matrix[dict1[u]][dict1[v]],".2f") for (u, v) in G.edges()}
        # 绘制图形和边标签
        pos = nx.shell_layout(G)  # 布局设置
        if len(np.nonzero(Matrix)[1]) > 25:
            plt.figure(figsize=(20, 20))  # 指定图片的宽度和高度
            plt.title(type + " " + sa[i] + f"adjacency matrix model:{num}")
            nx.draw(G, pos, node_color=[colors[j] for j in v1], labels={j: j for j in v1},node_size=2000)  # 绘制节点和标签
            nx.draw_networkx_edge_labels(G, pos, edge_labels=w)  # 绘制边和标签
        elif len(np.nonzero(Matrix)[1]) < 10:
            plt.figure(figsize=(8, 8))
            plt.title(type + " " + sa[i] + f" adjacency matrix model:{num}")
            nx.draw(G, pos, node_color=[colors[j] for j in v1], labels={j: j for j in v1}, node_size=2000)  # 绘制节点和标签
            nx.draw_networkx_edge_labels(G, pos, edge_labels=w)  # 绘制边和标签
        else:
            plt.figure(figsize=(8, 8))
            plt.title(type + " " + sa[i] + f"adjacency matrix model:{num}")
            nx.draw(G, pos, node_color=[colors[j] for j in v1], labels={j: j for j in v1}, node_size=1500)  # 绘制节点和标签
            nx.draw_networkx_edge_labels(G, pos, edge_labels=w)  # 绘制边和标签
        plt.savefig(os.path.join(path,sa[i] + ".png"))  # 指定图片的路径和格式
    for i in range(len(adj_list)):
        Matrix = adj_list[i].detach().cpu().to_dense().numpy()
        scipy.io.savemat(os.path.join(path, sa[i] + "_adj.mat"), {"data":Matrix})
    scipy.io.savemat(os.path.join(path, "v_nodes.mat"), {"data":v})
    return base_path
def make_graph(adj,nodes,dic,dic1,ak=2, subject=""):
    n = len(adj) - 1
    try:
        n = nodes.index(subject)
    except ValueError:
        print(f"值 {subject} 未在列表中找到")
    dit = {}
    if adj[n][n] == 0:
        adj[n][n] = 1
    new_nodes = []
    colors = {}
    j = 0
    b = 0.005
    if ak == 2:
        b = 0.1
    for i in range(len(nodes)):
        if adj[i][n] >= b:
            dit[dic1[i]] = j
            j += 1
            new_nodes.append(dic1[i])
    new_adj = np.zeros((j, j))
    for i in range(len(new_adj)):
        a = new_nodes[i]
        for j in range(i,len(new_nodes)):
            if i != j or i == len(new_adj) - 1:
                b = new_nodes[j]
                new_adj[i][j] = adj[dic[a]][dic[b]]
                new_adj[j][i] = adj[dic[a]][dic[b]]
    for i in range(len(new_nodes)):
        if "test" in new_nodes[i] :
            colors[new_nodes[i]] = "pink"
        else:
            colors[new_nodes[i]] = "blue"
    return new_nodes,new_adj,dit,colors
def  train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch,fold,
               fMRI_type,fold_repeat_all,data_subfolder,
               N_SEED_all,adj_parameter,dim_he_list,data_num,atlas,model_folder,type_folder
                ,test_fMRI, test_sMRI, test_DTI):
    test_inverval = 1
    num_view = len(view_list)
    dim_hvcdn = pow(num_class,num_view) #这是干啥的 流程图里那个立方体
    if data_folder == './xmn2/model/SCZ/':
        source_folder = os.path.join(data_folder, data_subfolder, data_num)
        models_dir = os.path.join(source_folder, "models_"+atlas, model_folder,type_folder)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
    label,data_fMRI, data_sMRI, data_DTI, names = prepare_trte_data(data_folder, view_list,fMRI_type,data_subfolder,data_num,atlas)
    # acc_list, sen_list, spe_list, f1_score_list, auc_list = [], [], [], [], []

    new_name = []
    for na in names:
        new_name.append(na)
    train_val = []
    a = len(label)
    for i in range(len(label)):
        train_val.append(i)
    data_fMRI_test, data_sMRI_test, data_DTI_test, test_names = test_data(test_fMRI, test_sMRI, test_DTI,
                                                                          data_subfolder)
    data_fMRI = np.append(data_fMRI, data_fMRI_test, axis=0)
    data_sMRI = np.append(data_sMRI, data_sMRI_test, axis=0)
    data_DTI = np.append(data_DTI, data_DTI_test, axis=0)

    for i, na in enumerate(test_names):
        new_name.append(na[0] +"_test")


    test_sum = len(data_fMRI_test)
    label_int = []
    if data_subfolder == 'GAD':
        for i in range(test_sum):
            label = np.append(label, [[1]], axis=0)
    else:
        for i in range(test_sum):
            label = np.append(label, [[0]], axis=0)
    for i in range(len(label)):
        label_int.append((int)(label[i]))
    test_pos = np.zeros((fold, len(data_fMRI_test)), dtype=np.float32)
    test_pos1 = np.zeros((fold, len(data_fMRI_test)), dtype=np.float32)
    for fold_repeat,N_SEED in enumerate(N_SEED_all):
        acc_list, sen_list, spe_list, f1_score_list, auc_list = [], [], [], [], []
        skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=N_SEED)
        split_time=-1
        for train_index, test_index in skf.split(label[0: len(label_int) - len(data_fMRI_test)],
                                                 label_int[0: len(label_int) - len(data_fMRI_test)]):
            val_index1 = []
            for i in range(test_sum):
                val_index1.append(i + a)
            data_tr_list = []
            data_te_list = []
            data_va_list = []
            split_time = split_time +1
            labels_tr = label[train_index, :]
            labels_te = label[test_index, :]
            labels_va = label[val_index1, :]

            data_tr_list.append(preprocessing.scale(data_fMRI[train_index, :]))
            data_te_list.append(preprocessing.scale(data_fMRI[test_index, :]))
            data_va_list.append(preprocessing.scale(data_fMRI[val_index1, :]))

            data_tr_list.append(preprocessing.scale(data_sMRI[train_index, :]))
            data_te_list.append(preprocessing.scale(data_sMRI[test_index, :]))
            data_va_list.append(preprocessing.scale(data_sMRI[val_index1, :]))
            if data_subfolder=="GAD":
                data_tr_list.append(preprocessing.scale(data_DTI[train_index, :]))
                data_te_list.append(preprocessing.scale(data_DTI[test_index, :]))
                data_va_list.append(preprocessing.scale(data_DTI[val_index1, :]))
            num_view = len(view_list)
            labels_tr = labels_tr.astype(int)
            labels_te = labels_te.astype(int)
            labels_va = labels_va.astype(int)

            num_tr = data_tr_list[0].shape[0]
            num_te = data_te_list[0].shape[0]
            num_va = data_va_list[0].shape[0]

            data_mat_list = []
            for i in range(num_view):
                data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i],
                                                     data_va_list[i]), axis=0))
            data_tensor_list = []
            for i in range(len(data_mat_list)):
                data_tensor_list.append(torch.FloatTensor(data_mat_list[i].astype(float)))
                if cuda:
                    data_tensor_list[i] = data_tensor_list[i].cuda()
            trte_idx = {}
            trte_idx["tr"] = list(range(num_tr))
            trte_idx["te"] = list(range(num_tr, (num_tr + num_te)))
            trte_idx["va"] = list(range((num_tr + num_te), (num_tr + num_te + num_va)))
            data_tr_list = []
            data_trteva_list = []
            for i in range(len(data_tensor_list)):
                data_tr_list.append(data_tensor_list[i][trte_idx["tr"]].clone())
                data_trteva_list.append(torch.cat((data_tensor_list[i][trte_idx["tr"]].clone(),
                                                   data_tensor_list[i][trte_idx["te"]].clone(),
                                                   data_tensor_list[i][trte_idx["va"]].clone()), 0
                                                  )
                                        )


            labels_trte = np.concatenate((labels_tr, labels_te, labels_va))
            labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
            onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
            sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
            #为解决样本标签不平衡问题，sample_weight_tr为每一个训练样本设置一个权重，应用于每个模态GCN的损失函数中不同类别的损失，权重为其在训练数据中频率的倒数
            sample_weight_tr = torch.FloatTensor(sample_weight_tr)
            if cuda:
                labels_tr_tensor = labels_tr_tensor.cuda()
                onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
                sample_weight_tr = sample_weight_tr.cuda()
            adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trteva_list,
                                                        trte_idx, adj_parameter)
            n = len(test_names)
            names_test = new_name[-n:]
            for na in names_test:
                base_path = save_Intermediate_results(v=new_name, adj_list=adj_te_list, num=split_time, type="original",
                                                      type_shujv=data_subfolder, base_path="./xmn2/results/",
                                                      subject=na)

            dim_list = [x.shape[1] for x in data_tr_list]
            #每一模态的初始特征维度[116, 116, 48]
            model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
            for m in model_dict:
                if cuda:
                    model_dict[m].cuda()
            print("\nPretrain GCNs...")
            optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
            for epoch in range(num_epoch_pretrain):
                train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                            onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict,adj_parameter, train_VCDN=False)
            print("\nTraining...")
            acc_te_best = f1_te_best = auc_te_best = sen_te_best = spe_te_best =epoch_best= 0
            optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
            best_adj = 0
            for epoch in range(num_epoch+1):
                loss_dict=train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                            onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict,adj_parameter)
                # print(loss_dict)
                if epoch % test_inverval == 0:
                    te_prob1, te_prob, adj_test = test_epoch(data_trteva_list, adj_te_list, trte_idx["te"], model_dict,
                                                   adj_parameter)
                    print("Test: Epoch {:d}".format(epoch))
                    if num_class == 2:
                        # print(te_prob)
                        # print(te_prob.argmax(1))
                        tn, fp, fn, tp=confusion_matrix(labels_trte[trte_idx["te"]], te_prob.argmax(1)).ravel()
                        acc=accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                        sen=tp/(tp+fn)
                        spe=tn/(tn+fp)
                        auc=roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])
                        f1_scor=f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                        fpr, tpr, threshold = roc_curve(labels_trte[trte_idx["te"]], te_prob[:, 1])
                        if (acc_te_best < acc and auc_te_best < auc) or (acc_te_best == acc and auc_te_best < auc) \
                                or (acc_te_best < acc and auc_te_best == auc):
                            save_model_dict(models_dir, model_dict,N_SEED,split_time) #当前重复次数，当前次数内第几折
                            acc_te_best = acc
                            sen_te_best=sen
                            spe_te_best=spe
                            auc_te_best = auc
                            f1_te_best = f1_scor
                            epoch_best=epoch
                            b = te_prob1[:, 0]
                            test_pos[split_time] = b[val_index1]
                            test_pos1[split_time] = te_prob1[:, 1][val_index1]
                            best_adj = adj_test
                        print(epoch_best,acc_te_best,sen_te_best,spe_te_best,auc_te_best,f1_te_best)
                    if epoch==num_epoch:
                        acc_list.append(acc_te_best)
                        sen_list.append(sen_te_best)
                        spe_list.append(spe_te_best)
                        f1_score_list.append(f1_te_best)
                        auc_list.append(auc_te_best)

            n = len(test_names)
            names_test = new_name[-n:]
            for na in names_test:
                base_path = save_Intermediate_results(v=new_name, adj_list=best_adj, num=split_time,
                                                      type="Intermediate",
                                                      type_shujv=data_subfolder, base_path="./xmn2/results/",
                                                      subject=na)
        source_folder = os.path.join(data_folder, data_subfolder, data_num,"models_"+atlas, model_folder,type_folder)
        seedfile=open(os.path.join(source_folder,"s.txt"),"a+")
        for i in range(1, len(data_fMRI_test) + 1):
            seedfile.write(f"The predicted probability of patient {new_name[-i]} being depressed is  %.4f +- %.4f\n" % (np.mean(test_pos),np.std(test_pos)))
            seedfile.write(f"The predicted probability of patient {new_name[-i]} being normal is %.4f +- %.4f\n" % (np.mean(test_pos1), np.std(test_pos1)))
        seedfile.close()
        seedcsv=open(os.path.join(source_folder,"results_dynamic_edges_mogonet_5-fold_10-repeat-aal116_find_sMRI_dim.csv"),"a+",encoding='utf-8',newline='' "")
        csv_writer = csv.writer(seedcsv)
        csv_writer.writerow([N_SEED,dim_he_list[0],adj_parameter,np.mean(acc_list),np.mean(sen_list),np.mean(spe_list),np.mean(auc_list),np.mean(f1_score_list)])
        seedcsv.close()
        print("The image is saved in {./xmn2/results/}")
        a123 = os.path.join(source_folder, "s.txt")
        print(f"The prediction results are saved in {a123}")
