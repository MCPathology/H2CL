import argparse
import os
import json
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import albumentations as albu

import torch.optim
import numpy as np
import pickle
import os.path as osp
import pandas as pd
import cv2
import time
from dataset import InputDataset


from better_mistakes.data.softmax_cascade import SoftmaxCascade
from better_mistakes.model.init import init_model_on_gpu, GlobalClassification, Hier
from better_mistakes.data.transforms import val_transforms
from better_mistakes.model.run_xent import run
from better_mistakes.model.run_visualization import run_nn
from better_mistakes.model.labels import make_all_soft_labels
from better_mistakes.util.label_embeddings import create_embedding_layer
from better_mistakes.util.devise_and_bd import generate_sorted_embedding_tensor
from better_mistakes.util.config import load_config
from better_mistakes.model.losses import HierarchicalCrossEntropyLoss, CosineLoss, RankingLoss, CosinePlusXentLoss, YOLOLoss
from better_mistakes.trees import load_hierarchy, get_weighting, load_distances, get_classes,DistanceDict

DATASET_NAMES = ["tiered-imagenet-224", "inaturalist19-224"]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print('==> Preparing data..')
def load_data(test_csv, distributed):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ])

    test_albu_transform = albu.Compose([
        albu.PadIfNeeded(min_height=1000, min_width=1000,
            border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), always_apply=True),
        albu.CenterCrop(700, 700, always_apply=True),
        #albu.Resize(448, 448, interpolation=cv2.INTER_LINEAR, always_apply=True),
        albu.Resize(384, 384, interpolation=cv2.INTER_LINEAR, always_apply=True),
        ])


    print("Loading test data")
    dataset_test = InputDataset(test_csv, False, test_transform, opts.data,
            albu_transform=test_albu_transform)

    print("Creating data loaders")
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return  dataset_test, test_sampler


def main(test_opts):
    gpus_per_node = torch.cuda.device_count()
    print('gpus_per_node {}'.format(gpus_per_node))

    expm_json_path = test_opts.experiments_json
    with open(expm_json_path) as fp:
        opts = json.load(fp)
        # convert dictionary to namespace
        opts = argparse.Namespace(**opts)
        
        opts.out_folder = None
        opts.epochs = 0
        opts.gpu = 0
    print(opts.arch)
    batch_size = test_opts.batch_size
    dataset_test, test_sampler = load_data(test_opts.test_csv, False)

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size,
        sampler=test_sampler, num_workers=16, pin_memory=True)


    with open(f'data/{opts.data}_tree.pkl', "rb") as f:
        hierarchy = pickle.load(f)
        
    if opts.loss == "yolo-v2":
        classes, _ = get_classes(hierarchy, output_all_nodes=True)
    else:
        if opts.data == 'cervix':
            classes =  ['Normal', 'ECC', 'RPC', 'MPC', 'PG', 'Atrophy', 'EMC', 'HCG', 'ASC-US',
                     'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-FN', 'AGC-ECC-NOS', 'AGC-EMC-NOS',
                      'ADC-ECC', 'ADC-EMC', 'FUNGI', 'ACTINO', 'TRI', 'HSV', 'CC']
        elif opts.data == 'skin':
            classes = ['actinic_cheilitis', 'cutaneous_horn', 'lichenoid', 'porokeratosis', 'seborrhoeic', 'solar_lentigo', 'wart', 'acral', 'atypical', 'blue', 'compound', 'congenital', 'dermal', 'halo', 'ink_spot_lentigo', 'involutingregressing', 'irritated', 'junctional', 'lentigo', 'papillomatous', 'ungual', 'benign_nevus', 'chrondrodermatitis', 'dermatofibroma', 'eczema', 'excoriation', 'nail_dystrophy', 'scar', 'sebaceous_hyperplasia', 'angioma', 'haematoma',                           'other_vascular', 'telangiectasia', 'basal_cell_carcinoma', 'pigmented_basal_cell_carcinoma', 'superficial_basaal_cell_carcinoma', 'actinic', 'lentigo_maligna', 'malignant_melanoma', 'scc_in_situ', 'squamous_cell_carcinoma']
        elif opts.data == 'GD':
            classes = ["Adenomyomatosis", "Polyps_and_cholesterol_crystals", "Abdomen_and_retroperitoneum", "Cholecystitis", "Membranous_and_gangrenous_cholecystitis", "Perforation", "Various_causes_of_gallbladder_wall_thickening", "Gallstones", "Carcinoma"]
    opts.num_classes = len(classes)

        # Model, loss, optimizer ------------------------------------------------------------------------------------------------------------------------------

        # more setup for devise and b+d
    if opts.devise:
        assert not opts.barzdenzler
        embeddings_mat, sorted_keys = generate_sorted_embedding_tensor(opts)
        embeddings_mat = embeddings_mat / np.linalg.norm(embeddings_mat, axis=1, keepdims=True)
        emb_layer, _, opts.embedding_size = create_embedding_layer(embeddings_mat)
        assert is_sorted(sorted_keys)

    if opts.barzdenzler:
        assert not opts.devise
        embeddings_mat, sorted_keys = generate_sorted_embedding_tensor(opts)
        embeddings_mat = embeddings_mat / np.linalg.norm(embeddings_mat, axis=1, keepdims=True)
        emb_layer, _, opts.embedding_size = create_embedding_layer(embeddings_mat)
        assert is_sorted(sorted_keys)

    # setup loss
    if opts.loss == "cross-entropy":
        loss_function = nn.CrossEntropyLoss().cuda(opts.gpu)
    elif opts.loss == "soft-labels":
        loss_function = nn.KLDivLoss().cuda(opts.gpu)
    elif opts.loss == "hierarchical-cross-entropy":
        weights = get_weighting(hierarchy, "exponential", value=opts.alpha)
        loss_function = HierarchicalCrossEntropyLoss(hierarchy, classes, weights).cuda(opts.gpu)
    elif opts.loss == "yolo-v2":

        cascade = SoftmaxCascade(hierarchy, classes).cuda(opts.gpu)
        num_leaf_classes = len(hierarchy.treepositions("leaves"))
        weights = get_weighting(hierarchy, "exponential", value=opts.beta)
        loss_function = YOLOLoss(hierarchy, classes, weights).cuda(opts.gpu)

        def yolo2_corrector(output):
            return cascade.final_probabilities(output)[:, :num_leaf_classes]

    elif opts.loss == "cosine-distance":
        loss_function = CosineLoss(emb_layer).cuda(opts.gpu)
    elif opts.loss == "ranking-loss":
        loss_function = RankingLoss(emb_layer, batch_size=opts.batch_size, single_random_negative=opts.devise_single_negative, margin=0.1).cuda(opts.gpu)
    elif opts.loss == "cosine-plus-xent":
        loss_function = CosinePlusXentLoss(emb_layer).cuda(opts.gpu)
    else:
        raise RuntimeError("Unkown loss {}".format(opts.loss))

    # for yolo, we need to decode the output of the classifier as it outputs the conditional probabilities
    corrector = yolo2_corrector if opts.loss == "yolo-v2" else lambda x: x

    # Test ------------------------------------------------------------------------------------------------------------------------------------------------
    summaries, summaries_table = dict(), dict()
    # setup model
    if opts.arch == "resnet18":
        opts.opts.feature_dim = 512
    elif opts.arch == "resnet50":
        opts.feature_dim = 2048
    elif opts.arch == "swinT":
        opts.feature_dim = 1536
    elif opts.arch == "mobilenet_v2":
        opts.feature_dim = 1280
    else:
        opts.feature_dim = 768
    backbone = init_model_on_gpu(gpus_per_node, opts)
    attention = GlobalClassification(opts.feature_dim, 512, opts.num_classes).to('cuda')
    model = Hier(backbone, attention)
    checkpoint_path = test_opts.checkpoint_path
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path,map_location='cpu')

        model.load_state_dict(checkpoint["state_dict"])
        print("=> loaded checkpoint '{}'".format(checkpoint_path))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))
        raise RuntimeError

    if opts.visualization:
        summary, _ = run_nn(test_loader, model, loss_function, classes, opts, 0, 0, emb_layer, embeddings_mat, is_inference=True)
    else:
        if opts.data == 'GD':
            summary_train, steps, _ = run_GD(
                train_loader, model, loss_function, classes, opts, epoch, steps, optimizer, is_inference=False, corrector=corrector,
            )
        else:
            summary_train, steps, _ = run(
                train_loader, model, loss_function, classes, opts, epoch, steps, optimizer, is_inference=False, corrector=corrector,
            True, corrector=corrector)
    
    with open(f'data/{opts.data}_level_names_dict.pkl','rb') as fo:
        level_names_dict = pickle.load(fo)

    df_species = pd.DataFrame(species_probs, columns=['species_' + x for x in level_names_dict['species']])
    df_test = pd.read_csv(test_opts.test_csv)
    df_res = pd.concat([df_test, df_species], axis = 1)
    df_res.to_csv(osp.join(opts.output, 
                            osp.basename(test_opts.test_csv).replace('.','_res.')), encoding='utf-8-sig',index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_json", default="experiment_test.json", help="json containing experiments and epochs to run")
    parser.add_argument("--checkpoint_path", help="Path to ckpt ")
    parser.add_argument("--test-csv", default='test_hierswin.csv', help="test csv of HiCervix")
    parser.add_argument("--batch-size", default=16, type=int, help="total batch size")

    parser.add_argument("--workers", default=4, type=int, help="number of data loading workers")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument("--visualization", action="store_true")
    test_opts = parser.parse_args()

    main(test_opts)
