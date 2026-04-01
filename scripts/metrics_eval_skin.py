import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import  accuracy_score, f1_score, roc_auc_score
from anytree import Node, search
import warnings

warnings.filterwarnings("ignore")

SKIN = Node("SKIN")

benign = Node("benign",parent=SKIN) 

benign_keratinocytic = Node("benign_keratinocytic", parent=benign) 
melanocytic = Node("benign_melanocytic", parent=benign) 
other_b = Node("benign_other",parent=benign)  ## microbe is the class  Organisms
vascular = Node("benign_vascular",parent=benign)

actinic_cheilitis = Node("benign_keratinocytic_actinic cheilitis",parent=benign_keratinocytic)
cutaneous_horn = Node("benign_keratinocytic_cutaneous horn",parent=benign_keratinocytic)
lichenoid = Node("benign_keratinocytic_lichenoid",parent=benign_keratinocytic)
porokeratosis = Node("benign_keratinocytic_porokeratosis",parent=benign_keratinocytic)
seborrhoeic = Node("benign_keratinocytic_seborrhoeic",parent=benign_keratinocytic)
solar_lentigo = Node("benign_keratinocytic_solar lentigo",parent=benign_keratinocytic)
wart = Node("benign_keratinocytic_wart",parent=benign_keratinocytic)
acral = Node("benign_melanocytic_acral",parent=melanocytic)
atypical = Node("benign_melanocytic_atypical",parent=melanocytic)
blue = Node("benign_melanocytic_blue",parent=melanocytic)
compound = Node("benign_melanocytic_compound",parent=melanocytic)
congenital = Node("benign_melanocytic_congenital",parent=melanocytic)
dermal = Node("benign_melanocytic_dermal",parent=melanocytic)
halo = Node("benign_melanocytic_halo",parent=melanocytic)
ink_spot_lentigo = Node("benign_melanocytic_ink spot lentigo",parent=melanocytic)
involutingregressing = Node("benign_melanocytic_involutingregressing",parent=melanocytic)
irritated = Node("benign_melanocytic_irritated",parent=melanocytic)
junctional = Node("benign_melanocytic_junctional",parent=melanocytic)
lentigo = Node("benign_melanocytic_lentigo",parent=melanocytic)
papillomatous = Node("benign_melanocytic_papillomatous",parent=melanocytic)
ungual = Node("benign_melanocytic_ungual",parent=melanocytic)
benign_nevus = Node("benign_melanocytic_benign nevus",parent=melanocytic)
chrondrodermatitis = Node("benign_other_chrondrodermatitis",parent=other_b)
dermatofibroma = Node("benign_other_dermatofibroma",parent=other_b)
eczema = Node("benign_other_eczema",parent=other_b)
excoriation = Node("benign_other_excoriation",parent=other_b)
nail_dystrophy = Node("benign_other_nail dystrophy",parent=other_b)
scar = Node("benign_other_scar",parent=other_b)
hyperplasia = Node("benign_other_sebaceous hyperplasia",parent=other_b)
angioma = Node("benign_vascular_angioma",parent=vascular)
haematoma = Node("benign_vascular_haematoma",parent=vascular)
other = Node("benign_vascular_other",parent=vascular)
telangiectasia = Node("benign_vascular_telangiectasia",parent=vascular)

malignant = Node("malignant",parent=SKIN)

bcc = Node("malignant_bcc", parent=malignant) 
malignant_keratinocytic = Node("malignant_keratinocytic", parent=malignant) 
melanoma = Node("malignant_melanoma",parent=malignant)  ## microbe is the class  Organisms
scc = Node("malignant_scc",parent=malignant)

basal_cell_carcinoma = Node("malignant_bcc_basal cell carcinoma",parent=bcc)
pigmented_basal_cell_carcinoma = Node("malignant_bcc_pigmented basal cell carcinoma",parent=bcc)
superficial_basaal_cell_carcinoma = Node("malignant_bcc_superficial basaal cell carcinoma",parent=bcc)
actinic = Node("malignant_keratinocytic_actinic",parent=malignant_keratinocytic)

lentigo_maligna = Node("malignant_melanoma_lentigo maligna",parent=melanoma)
malignant_melanoma = Node("malignant_melanoma_melanoma",parent=melanoma)
scc_in_situ = Node("malignant_scc_scc in situ",parent=scc)
squamous_cell_carcinoma = Node("malignant_scc_squamous cell carcinoma",parent=scc)

class_names = ['benign_keratinocytic_actinic cheilitis', 'benign_keratinocytic_cutaneous horn', 'benign_keratinocytic_lichenoid', 'benign_keratinocytic_porokeratosis', 'benign_keratinocytic_seborrhoeic', 'benign_keratinocytic_solar lentigo', 'benign_keratinocytic_wart', 'benign_melanocytic_acral', 'benign_melanocytic_atypical', 'benign_melanocytic_blue', 'benign_melanocytic_compound', 'benign_melanocytic_congenital', 'benign_melanocytic_dermal', 'benign_melanocytic_halo', 'benign_melanocytic_ink spot lentigo', 'benign_melanocytic_involutingregressing', 'benign_melanocytic_irritated', 'benign_melanocytic_junctional', 'benign_melanocytic_lentigo', 'benign_melanocytic_papillomatous', 'benign_melanocytic_ungual', 'benign_melanocytic_benign nevus', 'benign_other_chrondrodermatitis', 'benign_other_dermatofibroma', 'benign_other_eczema', 'benign_other_excoriation', 'benign_other_nail dystrophy', 'benign_other_scar', 'benign_other_sebaceous hyperplasia', 'benign_vascular_angioma', 'benign_vascular_haematoma', 'benign_vascular_other', 'benign_vascular_telangiectasia', 'malignant_bcc_basal cell carcinoma', 'malignant_bcc_pigmented basal cell carcinoma', 'malignant_bcc_superficial basaal cell carcinoma', 'malignant_keratinocytic_actinic', 'malignant_melanoma_lentigo maligna', 'malignant_melanoma_melanoma', 'malignant_scc_scc in situ', 'malignant_scc_squamous cell carcinoma']

level_1_names = ['benign', 'malignant']
level_2_names = ['benign_keratinocytic', 'benign_melanocytic', 'benign_other', 'benign_vascular', 'malignant_bcc', 'malignant_keratinocytic', 'malignant_melanoma', 'malignant_scc']
level_3_names = ['benign_keratinocytic_actinic cheilitis', 'benign_keratinocytic_cutaneous horn', 'benign_keratinocytic_lichenoid', 'benign_keratinocytic_porokeratosis', 'benign_keratinocytic_seborrhoeic', 'benign_keratinocytic_solar lentigo', 'benign_keratinocytic_wart', 'benign_melanocytic_acral', 'benign_melanocytic_atypical', 'benign_melanocytic_blue', 'benign_melanocytic_compound', 'benign_melanocytic_congenital', 'benign_melanocytic_dermal', 'benign_melanocytic_halo', 'benign_melanocytic_ink spot lentigo', 'benign_melanocytic_involutingregressing', 'benign_melanocytic_irritated', 'benign_melanocytic_junctional', 'benign_melanocytic_lentigo', 'benign_melanocytic_papillomatous', 'benign_melanocytic_ungual', 'benign_melanocytic_benign nevus', 'benign_other_chrondrodermatitis', 'benign_other_dermatofibroma', 'benign_other_eczema', 'benign_other_excoriation', 'benign_other_nail dystrophy', 'benign_other_scar', 'benign_other_sebaceous hyperplasia', 'benign_vascular_angioma', 'benign_vascular_haematoma', 'benign_vascular_other', 'benign_vascular_telangiectasia', 'malignant_bcc_basal cell carcinoma', 'malignant_bcc_pigmented basal cell carcinoma', 'malignant_bcc_superficial basaal cell carcinoma', 'malignant_keratinocytic_actinic', 'malignant_melanoma_lentigo maligna', 'malignant_melanoma_melanoma', 'malignant_scc_scc in situ', 'malignant_scc_squamous cell carcinoma']
level_1_names2id = dict(zip(level_1_names, range(len(level_1_names))))
level_2_names2id = dict(zip(level_2_names, range(len(level_2_names))))
level_3_names2id = dict(zip(level_3_names, range(len(level_3_names))))
iaw=0
level_names = [level_1_names, level_2_names,level_3_names,]
classname2newid = dict(zip(class_names, range(len(class_names))))
newid2classname= {v:k for k,v in classname2newid.items()}
# print('Number of total annotated classes: {}'.format(len(class_names)))
for i in range(3):
    print('Number of level {} classes: {}'.format(i+1, len(level_names[i])))
    

classname_paths = {}
for class_name in class_names:
    class_name_node = search.find_by_attr(SKIN, class_name)
    
    path_node_classes = [x.name for x in class_name_node.path]#[1:] #exclude the root node of 'TCT'
#     print('The nodes in the path (from root node) to reach the nodes of {}'.format(class_name))
#     print(path_node_classes)
    classname_paths[class_name] = path_node_classes
    
def evaluate(df, method=2, to_level=3, cal_auc=True):
    df_res = df.copy()
    if method==1:
        print('Using partial data...')  # 
    else:
        print('Using all data...')   # default way
        df_res['level_1'] = df_res['label'].apply(lambda x: find_level_name_v2(x, level=1))
        df_res['level_2'] = df_res['label'].apply(lambda x: find_level_name_v2(x, level=2))
        df_res['level_3'] = df_res['label'].apply(lambda x: find_level_name_v2(x, level=3))
        df_res['level_1_id'] = df_res['level_1'].map(level_1_names2id)
        df_res['level_2_id'] = df_res['level_2'].apply(lambda x: level_2_names2id.get(x, -1))
        df_res['level_3_id'] = df_res['level_3'].apply(lambda x: level_3_names2id.get(x, -1))
    acc_lst = []
    hierarchical_distances = []
    auc_lst = []
    f1_score_lst = []
    middle_class_metric = []
    #### level 1
    level_1_acc = accuracy_score(df_res['level_1'], df_res['level_1_pred'])
    #acc_lst.append(level_1_acc)
    level_1_distance = df_res[['level_1','level_1_pred']].apply(lambda x: lca_height(x[0], x[1]), axis =1).mean() # intra-level distance
    #hierarchical_distances.append(level_1_distance)
    if cal_auc:
        #print(df_res['level_1_id'])
        auc_lst.append(roc_auc_score(df_res['level_1_id'], df_res.filter(regex='order*',axis=1).iloc[:,1], multi_class='ovr'))
    f1_score_lst.append(f1_score(df_res['level_1'], df_res['level_1_pred'], average='macro'))

    print('Level_1 accuracy: {:.4f}'.format(level_1_acc))
    print('level_1 hierarchical distanse: {:.4f}'.format(level_1_distance))
    # print('level_1 auc: {:.4f}'.format(auc_lst))
    print('#'*30)
    
    #### level 2
    df_level2 = df_res[(~df_res['level_2'].isna())&(df_res['level_2']!='AGC')] # exclude AGC
    # df_level2['level_2_pred'].to_csv('level_2_pred.txt', index=False, header=False)
    level_2_distance = df_level2[['level_2','level_2_pred']].apply(lambda x: lca_height(x[0], x[1]), axis =1).mean()
    hierarchical_distances.append(level_2_distance)
    
    level_2_acc = accuracy_score(df_level2['level_2'], df_level2['level_2_pred'])
    acc_lst.append(level_2_acc)
    if cal_auc:
        auc_lst.append(roc_auc_score(df_level2['level_2_id'], df_level2.filter(regex='family*',axis=1), multi_class='ovr'))
    f1_score_lst.append(f1_score(df_level2['level_2'], df_level2['level_2_pred'], average='macro'))
    print('Level_2 accuracy: {:.4f}'.format(level_2_acc))
    print('level_2 hierarchical distande: {:.4f}'.format(level_2_distance))
    print('#'*30)
    
    if to_level == 2:
        pass
    else:
        #### level 3
        df_level3 = df_res[(~df_res['level_3'].isna())&(~df_res['level_3'].isin(['AGC','AGC-NOS', 'ADC']))]
    #     df_level3_normalcls = df_level3[~df_level3['level_3_pred'].isna()]
        level_3_distance = df_level3[['level_3','level_3_pred']].apply(lambda x: lca_height(x[0], x[1]), axis =1).mean()
        hierarchical_distances.append(level_3_distance)
    #     df_level3.fillna(value='undercls',inplace=True)
        ## only consider intra-level confusion 
        level_3_acc = accuracy_score(df_level3['level_3'], df_level3['level_3_pred'])
        acc_lst.append(level_3_acc)
        if cal_auc:
            #print(df_level3['level_3_id'])
            unique_labels = sorted(df_level3['level_3_id'].unique())
            # print(sorted(unique_labels))
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            
            mapped_label = df_level3['level_3_id'].map(label_mapping)
            # print(sorted(mapped_label.unique()))
            filtered_scores = df_level3.filter(regex='species*',axis=1).iloc[:,unique_labels]
            def softmax(x):
                e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return e_x / e_x.sum(axis=1, keepdims=True)
            normalized_scores = softmax(filtered_scores.values)
            #print(normalized_scores.shape)
            # assert len(filtered_scores.columns) == len(unique_labels)
            roc_auc = roc_auc_score(mapped_label, normalized_scores, multi_class='ovr')
            
            auc_lst.append(roc_auc)
        f1_score_lst.append(f1_score(df_level3['level_3'], df_level3['level_3_pred'], average='macro'))
        print('Level_3 accuracy: {:.4f}'.format(level_3_acc))
        print('level_3 hierarchical distande: {:.4f}'.format(level_3_distance))
        print('#'*30)
      
        print('Average accuracy: {:.4f}'.format(np.mean(acc_lst)))
        print('Average hierarchical distance: {:.4f}'.format(np.mean(hierarchical_distances)))
        print('auc: {},{}'.format(auc_lst,(auc_lst[1]+auc_lst[2])/2))
    res_data = acc_lst + hierarchical_distances + auc_lst# + f1_score_lst
    return res_data

def add_order_family_pred_score(df):
    for class_name in level_1_names:
        class_name_node = search.find_by_attr(SKIN, class_name)
        leaf_node_names =[x.name for x in  class_name_node.leaves]
#         print(['species_{}'.format(x) for x in leaf_node_names])
        df['order_{}'.format(class_name)] = df[['species_{}'.format(x) for x in leaf_node_names]].sum(axis=1)
    
    for class_name in level_2_names:
        class_name_node = search.find_by_attr(SKIN, class_name)
        # print(class_name_node)
        leaf_node_names =[x.name for x in  class_name_node.leaves]
        df['family_{}'.format(class_name)] = df[['species_{}'.format(x) for x in leaf_node_names]].sum(axis=1)
    return df

def add_level_pred(df):
    df['level_1_pred'] = [x.split('_')[-1] for x in df.filter(regex='order*',axis=1).idxmax(axis=1)]
    # print(df.filter(regex='family*',axis=1).idxmax(axis=1))
    df['level_2_pred'] = [x.split('family_')[-1] for x in df.filter(regex='family*', axis=1).idxmax(axis=1)]
    
    df['level_3_pred'] = [x.split('species_')[-1] for x in df.filter(regex='species*',axis=1).idxmax(axis=1)]
    
    df['level_1_pred_score'] = df.filter(regex='order*',axis=1).max(axis=1)
    df['level_2_pred_score'] = df.filter(regex='family*',axis=1).max(axis=1)
    df['level_3_pred_score'] = df.filter(regex='species*',axis=1).max(axis=1)
    return df

def find_level_name_v2(class_name, level=1):
    """fill the finest label using the coarse lable"""
    class_name = class_name.strip()
    path_node_classes = classname_paths[class_name]
    if len(path_node_classes)>level:
        return path_node_classes[level]
    else:
        return path_node_classes[-1]
    
def lca_height(class_name1, class_name2, logarithmic=True):
    """lowest common ancestor height, taking the level into acount np.log(1+height)
    """

    node1 = search.find_by_attr(SKIN, class_name1)
    node2 = search.find_by_attr(SKIN, class_name2)
    # print(node2)
    node1_path_names = [x.name for x in node1.path]
    #global iaw
    #iaw = iaw+1
    # print(iaw) 
    node2_path_names = [x.name for x in node2.path]
    if len(node1_path_names) == len(node2_path_names):
        height = 0
        for name1, name2 in list(zip(node1_path_names, node2_path_names))[::-1]:
            if name1==name2:
                return np.log(1+height) if logarithmic else height
            else:
                height +=1
    #             return name1
    else:
        common_length = len(set(node1_path_names).intersection(set(node2_path_names)))
        longest_length = max(len(node1_path_names), len(node2_path_names))
        height = longest_length - common_length
        return height


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--res-csv", default='HierSwin_res.csv', help="test csv results of HiCervix")
    args = parser.parse_args()

    df_hierswin = pd.read_csv(args.res_csv)
    df_hierswin = add_order_family_pred_score(df_hierswin)
    df_hierswin = add_level_pred(df_hierswin)
    hierswin = evaluate(df_hierswin,method=2)
