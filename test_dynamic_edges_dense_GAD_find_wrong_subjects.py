""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,confusion_matrix,roc_curve
import torch
import torch.nn.functional as F
from models_dynamic_edges_dense_k_hup_neighbors import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter,load_model_dict,save_Intermediate_results
import pandas as pd
import csv
import xlrd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(data_folder, view_list,fMRI_type,data_subfolder,data_num,atlas,fMRI,sMRI,DIT):
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
        data_fMRI_csv_test = pd.read_csv(fMRI)
    elif fMRI_type == 4:
        data_fMRI_csv = pd.read_csv(
            os.path.join(source_folder, "fMRI", atlas, "ReHo_VMHC_fALFF_ALFF", "fALFF", "fALFF_features.csv"))
    elif fMRI_type == 5:
        data_fMRI_csv = pd.read_csv(
            os.path.join(source_folder, "fMRI", atlas, "fc_fisher",  "feature_fc.csv"))

    data_sMRI_csv = pd.read_csv(
        os.path.join(source_folder, "sMRI", atlas, "GMV", "GMV_features.csv"))
    data_sMRI_csv_test = pd.read_csv(sMRI)
     # data_sMRI_csv = pd.read_csv(
    #     os.path.join(source_folder, "fMRI", "aal_116","fc_fisher",  "feature_fc.csv"))
    test_DIT = []
    data_DTI=[]
    if data_subfolder == "GAD":
        data_DTI_csv = pd.read_csv(
            os.path.join(source_folder, "DTI", "AllAtlasResults", "WMlabelResults_FA.csv"))
        data_DTI = data_DTI_csv.values[:, 1:]
        test_DIT = pd.read_csv(DIT).values[:, 1:]
    label = np.array(label_csv.values[:, 1:])  # 所有数据的label
    data_fMRI = data_fMRI_csv.values[:, 1:]
    data_sMRI = data_sMRI_csv.values[:, 1:]
    data_sMRI_test = data_sMRI_csv_test.values[:, 1:]
    data_fMRI_test = data_fMRI_csv_test.values[:, 1:]
    ids = data_fMRI_csv.values[:, 0:1]
    ids = np.reshape(ids, (-1,))
    ids = ids.tolist()
    ids_test = data_sMRI_csv_test.values[:, 0:1]
    ids_test = np.reshape(ids_test, (-1,))
    ids_test = ids_test.tolist()
    return label,data_fMRI,data_sMRI,data_DTI,data_fMRI_test,data_sMRI_test,test_DIT,ids, ids_test
def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine"  # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter[i], data_tr_list[i],
                                                       adj_metric)  # 计算到node_num*edges_per_node的数字大小，
        # 接下来所有大于等于这个数字的余弦距离都留下
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        # adj_train_list中全是非0元素了
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))

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

def gen_tr_adj_mat_make_new_graph(data_tr_list, adj_parameter): #每一层GCN后更新adj_list
    adj_metric = "cosine"  # cosine distance
    adj_train_list = []
    adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list,
                                                   adj_metric)  # 计算到node_num*edges_per_node的数字大小，
    # 接下来所有大于等于这个数字的余弦距离都留下
    adj_train_list= gen_adj_mat_tensor(data_tr_list, adj_parameter_adaptive, adj_metric)
        # adj_train_list中全是非0元素了

    return adj_train_list

def test_epoch(data_list, adj_list, te_idx, model_dict, adj_parameter):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    adj_list_next = []
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i + 1)](
            model_dict["E{:}".format(i + 1)](adj_parameter[i], data_list[i], adj_list[i])))
    if num_view >= 2:
        c = model_dict["C"](ci_list)
    else:
        c = ci_list[0]
    c = c[te_idx, :]
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    for i in range(num_view):
        adj_list_next.append(gen_tr_adj_mat_make_new_graph(ci_list[i], adj_parameter[i]))
    return prob,adj_list_next


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c,
               num_epoch_pretrain, num_epoch, fold,
               fMRI_type, fold_repeat_all, data_subfolder,
               N_SEED_all, adj_parameter, dim_he_list,data_num,atlas,model_folder,type_folder,preprocessing_label,fMRI,sMRI,DIT, save_results="/home/lining/xmn1/results/"):
    base_path = ""
    if data_subfolder == "GAD":
        print("Use fMRI,sMRI,DTI")
    elif data_subfolder == 'COBRE':
        print("Use fMRI,sMRI")
    test_inverval = 1
    models_dir = ""
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)  # 这是干啥的 流程图里那个立方体
    source_folder = os.path.join(data_folder, data_subfolder, data_num)
    type_folder=type_folder
    models_dir = os.path.join(source_folder, "models_"+atlas,model_folder,type_folder)
    print("model:",model_folder)
    label, data_fMRI, data_sMRI, data_DTI,test_fMRI,test_sMRI,test_DIT,names,test_names = prepare_trte_data(data_folder, view_list, fMRI_type, data_subfolder,data_num,atlas,fMRI=fMRI,sMRI=sMRI,DIT=DIT)
    data_fMRI = np.append(data_fMRI,test_fMRI,axis=0)
    data_sMRI = np.append(data_sMRI,test_sMRI,axis=0)
    for name in test_names:
        names.append(name)
    if data_subfolder == 'GAD':
        data_DTI = np.append(data_DTI,test_DIT,axis=0)
    acc_list, sen_list, spe_list, f1_score_list, auc_list = [], [], [], [], []
    yuce_0, yu_ce_1 = [], []
    label_int = []
    if data_subfolder == 'GAD':
        label = np.append(label, [[1]], axis=0)
    else:
        label = np.append(label, [[0]],axis=0)
    for i in range(len(label)):
        label_int.append((int)(label[i]))
    # fold_repeat = fold_repeat_all
    # for fold_repeat in range(fold_repeat_all):
    for fold_repeat_time, N_SEED in enumerate(N_SEED_all):
        fold_repeat=N_SEED
        # N_SEED_next = N_SEED_next +10*fold_repeat
        skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=N_SEED)
        # skf = StratifiedKFold(n_splits=fold, shuffle=True)
        split_time = -1
        for train_index, test_index in skf.split(label[0:len(label_int) - 1], label_int[0:len(label_int) - 1]):
            data_tr_list = []
            data_te_list = []
            test_index = np.append(test_index, len(label_int) - 1)
            split_time = split_time + 1
            labels_tr = label[train_index, :]
            labels_te = label[test_index, :]
            if data_subfolder == "GAD" and preprocessing_label==1:
                data_tr_list.append(preprocessing.scale(data_fMRI[train_index, :]))
                data_te_list.append(preprocessing.scale(data_fMRI[test_index, :]))
                data_tr_list.append(preprocessing.scale(data_sMRI[train_index, :]))
                data_te_list.append(preprocessing.scale(data_sMRI[test_index, :]))
                data_tr_list.append(preprocessing.scale(data_DTI[train_index, :]))
                data_te_list.append(preprocessing.scale(data_DTI[test_index, :]))
            elif data_subfolder == "GAD" and preprocessing_label == 0:
                data_tr_list.append(data_sMRI[train_index, :])
                data_te_list.append(data_sMRI[test_index, :])
                data_tr_list.append(data_DTI[train_index, :])
                data_te_list.append(data_DTI[test_index, :])
            elif data_subfolder == "COBRE" and preprocessing_label==1:
                data_tr_list.append(preprocessing.scale(data_fMRI[train_index, :]))
                data_te_list.append(preprocessing.scale(data_fMRI[test_index, :]))
                data_tr_list.append(preprocessing.scale(data_sMRI[train_index, :]))
                data_te_list.append(preprocessing.scale(data_sMRI[test_index, :]))
            elif data_subfolder == "COBRE" and preprocessing_label == 0:
                data_tr_list.append(data_fMRI[train_index, :])
                data_te_list.append(data_fMRI[test_index, :])
                data_tr_list.append(data_sMRI[train_index, :])
                data_te_list.append(data_sMRI[test_index, :])
            num_view = len(view_list)
            labels_tr = labels_tr.astype(int)
            labels_te = labels_te.astype(int)
            num_tr = data_tr_list[0].shape[0]
            num_te = data_te_list[0].shape[0]
            data_mat_list = []
            for i in range(num_view):
                data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
            data_tensor_list = []
            for i in range(len(data_mat_list)):
                data_tensor_list.append(torch.FloatTensor(data_mat_list[i].astype(float)))
                if cuda:
                    data_tensor_list[i] = data_tensor_list[i].cuda()
            idx_dict = {}
            idx_dict["tr"] = list(range(num_tr))
            idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
            data_train_list = []
            data_all_list = []
            for i in range(len(data_tensor_list)):
                data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
                data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                                data_tensor_list[i][idx_dict["te"]].clone()), 0))
            labels_trte = np.concatenate((labels_tr, labels_te))

            data_tr_list, data_trte_list, trte_idx, labels_trte = data_train_list, data_all_list, idx_dict, labels_trte

            labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
            onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
            sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
            # 为解决样本标签不平衡问题，sample_weight_tr为每一个训练样本设置一个权重，应用于每个模态GCN的损失函数中不同类别的损失，权重为其在训练数据中频率的倒数
            sample_weight_tr = torch.FloatTensor(sample_weight_tr)
            if cuda:
                labels_tr_tensor = labels_tr_tensor.cuda()
                onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
                sample_weight_tr = sample_weight_tr.cuda()
            adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
            dim_list = [x.shape[1] for x in data_tr_list]
            # 每一模态的初始特征维度[116, 116, 48]
            model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
            for m in model_dict:
                if cuda:
                    model_dict[m].cuda()

            model_dict=load_model_dict(models_dir, model_dict,fold_repeat,split_time)
            te_prob,adj_test = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict, adj_parameter)
            _ = save_Intermediate_results(v=names,adj_list=adj_te_list,num=split_time,type="original",type_shujv=data_subfolder,base_path=save_results,name=names[-1])
            base_path = save_Intermediate_results(v=names,adj_list=adj_test,num=split_time,type="Intermediate",type_shujv=data_subfolder,base_path=save_results,name=names[-1])
            if num_class == 2:
                a = labels_trte[trte_idx["te"]]
                tn, fp, fn, tp = confusion_matrix(labels_trte[trte_idx["te"]], te_prob.argmax(1)).ravel()
                acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                sen = tp / (tp + fn)
                spe = tn / (tn + fp)
                auc = roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])
                f1_scor = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                fpr, tpr, threshold = roc_curve(labels_trte[trte_idx["te"]], te_prob[:, 1])
                yuce_0.append(te_prob[-1][0])
                yu_ce_1.append(te_prob[-1][1])
                acc_list.append(acc)
                sen_list.append(sen)
                spe_list.append(spe)
                f1_score_list.append(f1_scor)
                auc_list.append(auc)
    predicts = os.path.join(save_results,data_subfolder,names[-1],"predict.txt")
    with open(predicts, 'w') as f:
        f.write('Probability of depression:{:.4f}'.format(np.mean(yu_ce_1)))
        f.write("\n")
        if np.mean(yu_ce_1) < 0.5:
            f.write('Subject health\n')
        else:
            f.write('Subject suffers from depression\n')
    #print("Intermediate results are stored in:",base_path)

