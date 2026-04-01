import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import  accuracy_score, f1_score, roc_auc_score
from anytree import Node, search
import warnings
warnings.filterwarnings('ignore')

GD = Node("GD")

benign = Node("Benign_neoplastic_hyperplastic", parent=GD)
inflammatory = Node("Inflammatory_or_Adjacent", parent=GD)
lithiasis = Node("Lithiasis", parent=GD)
malignant = Node("Malignant", parent=GD)

Node("Polyps_and_cholesterol_crystals", parent=benign)
Node("Adenomyomatosis", parent=benign)

Node("Cholecystitis", parent=inflammatory)
Node("Membranous_and_gangrenous_cholecystitis", parent=inflammatory)
Node("Perforation", parent=inflammatory)
Node("Various_causes_of_gallbladder_wall_thickening", parent=inflammatory)
Node("Abdomen_and_retroperitoneum", parent=inflammatory)

Node("Gallstones", parent=lithiasis)

Node("Carcinoma", parent=malignant)

class_names = ["Adenomyomatosis", "Polyps_and_cholesterol_crystals", "Abdomen_and_retroperitoneum", "Cholecystitis", "Membranous_and_gangrenous_cholecystitis", "Perforation", "Various_causes_of_gallbladder_wall_thickening",
                    "Gallstones", "Carcinoma"]
level_1_names = ["Benign_neoplastic_hyperplastic","Inflammatory_or_Adjacent","Lithiasis","Malignant"]
level_2_names = ["Adenomyomatosis", "Polyps_and_cholesterol_crystals", "Abdomen_and_retroperitoneum", "Cholecystitis", "Membranous_and_gangrenous_cholecystitis", "Perforation", "Various_causes_of_gallbladder_wall_thickening",
                    "Gallstones", "Carcinoma"]
level_1_names2id = dict(zip(level_1_names, range(len(level_1_names))))
level_2_names2id = dict(zip(level_2_names, range(len(level_2_names))))

level_names = [level_1_names, level_2_names,]
classname2newid = dict(zip(class_names, range(len(class_names))))
newid2classname= {v:k for k,v in classname2newid.items()}
# print('Number of total annotated classes: {}'.format(len(class_names)))
for i in range(2):
    print('Number of level {} classes: {}'.format(i+1, len(level_names[i])))
    

classname_paths = {}
for class_name in class_names:
    class_name_node = search.find_by_attr(GD, class_name)
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
        # print(df_res['class_name'])
        df_res['level_1'] = df_res['class_name'].apply(lambda x: find_level_name_v2(x, level=1))
        df_res['level_2'] = df_res['class_name'].apply(lambda x: find_level_name_v2(x, level=2))
        
        df_res['level_1_id'] = df_res['level_1'].map(level_1_names2id)
        df_res['level_2_id'] = df_res['level_2'].apply(lambda x: level_2_names2id.get(x, -1))
        
    acc_lst = []
    hierarchical_distances = []
    auc_lst = []
    f1_score_lst = []
    middle_class_metric = []
    #### level 1
    level_1_acc = accuracy_score(df_res['level_1'], df_res['level_1_pred'])
    acc_lst.append(level_1_acc)

    level_1_distance = df_res[['level_1','level_1_pred']].apply(lambda x: lca_height(x[0], x[1]), axis =1).mean() # intra-level distance
    hierarchical_distances.append(level_1_distance)
    if cal_auc:
        auc_lst.append(roc_auc_score(df_res['level_1_id'], df_res.filter(regex='order*',axis=1), multi_class='ovr'))
    f1_score_lst.append(f1_score(df_res['level_1'], df_res['level_1_pred'], average='macro'))

    print('Level_1 accuracy: {:.4f}'.format(level_1_acc))
    print('level_1 hierarchical distanse: {:.4f}'.format(level_1_distance))
    # print('level_1 auc: {:.4f}'.format(auc_lst))
    print('#'*30)
    
    #### level 2
    df_level2 = df_res[(~df_res['level_2'].isna())&(df_res['level_2']!='AGC')] # exclude AGC
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
        print('#'*30)
      
        print('Average accuracy: {:.4f}'.format(np.mean(acc_lst)))
        print('Average hierarchical distance: {:.4f}'.format(np.mean(hierarchical_distances)))
        print('auc: {:.4f},{:.4f},{:.4f}'.format(auc_lst[0], auc_lst[1], (auc_lst[0]+auc_lst[1])/2))
    res_data = acc_lst + hierarchical_distances + auc_lst# + f1_score_lst
    return res_data

def add_order_family_pred_score(df):
    for class_name in level_1_names:
        class_name_node = search.find_by_attr(GD, class_name)
        leaf_node_names =[x.name for x in  class_name_node.leaves]
#         print(['species_{}'.format(x) for x in leaf_node_names])
        df['order^{}'.format(class_name)] = df[['species_{}'.format(x) for x in leaf_node_names]].sum(axis=1)
    
    for class_name in level_2_names:
        class_name_node = search.find_by_attr(GD, class_name)
        leaf_node_names =[x.name for x in  class_name_node.leaves]
        df['family^{}'.format(class_name)] = df[['species_{}'.format(x) for x in leaf_node_names]].sum(axis=1)
    return df

def add_level_pred(df):
    df['level_1_pred'] = [x.split('^')[-1] for x in df.filter(regex='order*',axis=1).idxmax(axis=1)]
    df['level_2_pred'] = [x.split('^')[-1] for x in df.filter(regex='family*',axis=1).idxmax(axis=1)]
    
    df['level_1_pred_score'] = df.filter(regex='order*',axis=1).max(axis=1)
    df['level_2_pred_score'] = df.filter(regex='family*',axis=1).max(axis=1)
    return df

def find_level_name_v2(class_name, level=1):
    """fill the finest label using the coarse lable"""
    class_name = class_name.strip()
    # print(class_name)
    path_node_classes = classname_paths[class_name]
    if len(path_node_classes)>level:
        return path_node_classes[level]
    else:
        return path_node_classes[-1]
    
def lca_height(class_name1, class_name2, logarithmic=True):
    """lowest common ancestor height, taking the level into acount np.log(1+height)
    """
    node1 = search.find_by_attr(GD, class_name1)
    node2 = search.find_by_attr(GD, class_name2)
    node1_path_names = [x.name for x in node1.path]
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
