""" Example for MOGONET classification
"""

#find the best 10 seed
#1 2 3中类型都可
from train_test_SCZ_dynamic_edges_dense_find_seed_GAD import *
#run the code
# from train_test_SCZ_dynamic_edges_dense_two_modalities_k_fold import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
if __name__ == "__main__":
    #三个
    data_folder = './xmn2/model/SCZ/'
    view_list = [1,2,3]
    num_epoch_pretrain = 100
    num_epoch = 300
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    if data_folder == './xmn2/model/SCZ/':
        num_class = 2
    data_subfolder='GAD'
    print(data_subfolder)
    fold_repeat_all = 10
    N_SEED_all = [227]
    fMRI_type=3
    print(fMRI_type)
    if fMRI_type==1:
        type_folder = "vmhc+gmv+fa"
    if fMRI_type==2:
        type_folder = "reho+gmv+fa"
    if fMRI_type==3:
        type_folder = "3+s+d"
    if fMRI_type == 4:
        type_folder = "falff+gmv+fa"
    atlas="aal_116"
    data_num = "56_new"
    model_folder="models_true_VCDN"
    print(fMRI_type)
    fold=5
    N_SEED = 1
    adj_num = 10
    adj_parameter = [adj_num, adj_num, adj_num]
    type_folder=type_folder+"_adj_parameter_"+str(adj_parameter[0])
    print(data_subfolder, fMRI_type, type_folder)
    dim_he_list = [48, 48, 24]
    fold_repeat_all=0
    if N_SEED == 1:
        train_test(data_folder, view_list, num_class,
                   lr_e_pretrain, lr_e, lr_c,
                   num_epoch_pretrain, num_epoch,fold,
                   fMRI_type,fold_repeat_all,data_subfolder,
                   N_SEED_all,adj_parameter,dim_he_list,data_num,atlas,
                   model_folder,type_folder,
                   test_fMRI="./xmn2/model/SCZ/GAD/fMRI.csv",
                   test_sMRI="./xmn2/model/SCZ/GAD/sMRI.csv",
                   test_DTI="./xmn2/model/SCZ/GAD/DTI.csv"
                   )
    fold_repeat_all = fold_repeat_all + 1
        # fMRI_type+=1