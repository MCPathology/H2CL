import time
import numpy as np
import os.path
import torch
import torch.nn as nn
from conditional import conditional
import tensorboardX
from better_mistakes.model.performance import accuracy
from better_mistakes.model.labels import make_batch_onehot_labels, make_batch_soft_labels
import torch.nn.functional as F
from tqdm import tqdm
from .tct_get_tree_target import *
from better_mistakes.model.init import SupervisedContrastiveLoss, HypSupervisedContrastiveLoss, HGContrastiveLoss, HGHypContrastiveLoss
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import hyptorch.pmath as pmath
from htsne_impl import TSNE as hTSNE
from collections import defaultdict
import pandas as pd
import scienceplots
from matplotlib.lines import Line2D
plt.style.use(['science', 'ieee'])

topK_to_consider = (1, 5)

# lists of ids for loggings performance measures
accuracy_ids = ["accuracy_top/%02d" % i for i in topK_to_consider]
dist_avg_ids = ["_avg/%02d" % i for i in topK_to_consider]
dist_top_ids = ["_top/%02d" % i for i in topK_to_consider]
dist_avg_mistakes_ids = ["_mistakes/avg%02d" % i for i in topK_to_consider]
hprec_ids = ["_precision/%02d" % i for i in topK_to_consider]
hmAP_ids = ["_mAP/%02d" % i for i in topK_to_consider]

def run(loader, model, loss_function, classes, opts, epoch, prev_steps, optimizer=None, is_inference=True, corrector=lambda x: x):
    """
    Runs training or inference routine for standard classification with soft-labels style losses
    """
    max_dist = max(distances.distances.values())
    order_euc_feature = []
    family_euc_feature = []
    order_hyp_feature = []
    family_hyp_feature = []
    order_label = []
    family_label = []
    vis = UMAPv()
    # for each class, create the optimal set of retrievals (used to calculate hierarchical precision @k)
    best_hier_similarities = _make_best_hier_similarities(classes, distances, max_dist)

    # Using different logging frequencies for training and validation
    log_freq = 1 if is_inference else opts.log_freq

    # strings useful for logging
    descriptor = "VAL" if is_inference else "TRAIN"
    loss_id = "loss/" + opts.loss
    dist_id = "ilsvrc_dist"

    # Initialise TensorBoard
    with_tb = opts.out_folder is not None
    
    if with_tb:
        tb_writer = tensorboardX.SummaryWriter(os.path.join(opts.out_folder, "tb", descriptor.lower()))

    # Initialise accumulators to store the several measures of performance (accumulate as sum)
    num_logged = 0
    loss_accum = 0.0
    time_accum = 0.0
    
    species_probs = []
    
    h_o_loss = HypSupervisedContrastiveLoss()
    e_o_loss = SupervisedContrastiveLoss()
    h_f_loss = HGHypContrastiveLoss()
    e_f_loss = HGContrastiveLoss()

    # Affects the behaviour of components such as batch-norm
    if is_inference:
        model.eval()
    else:
        model.train()

    with conditional(is_inference, torch.no_grad()):
        time_load0 = time.time()
        progress_bar = tqdm(loader, desc="Processing Batches")
        for batch_idx, (embeddings, [order, family, target], order_embedding, family_embedding, family_mask) in enumerate(progress_bar):
            this_load_time = time.time() - time_load0
            this_rest0 = time.time()
            if opts.gpu is not None:
                embeddings = embeddings.cuda(opts.gpu, non_blocking=True)

            output, text_loss, order_hyp, family_hyp, order_euc, family_euc = model(embeddings, order_embedding, family_embedding, family_mask)
            
            species_probs.extend(F.softmax(output, dim=1).tolist())
            
            if opts.loss == "soft-labels":
                output = torch.nn.functional.log_softmax(output, dim=1)
                if opts.soft_labels:
                    target_distribution = make_batch_soft_labels(all_soft_labels, target, opts.num_classes, opts.batch_size, opts.gpu)
                else:
                    target_distribution = make_batch_onehot_labels(target, opts.num_classes, opts.batch_size, opts.gpu)
                loss = loss_function(output, target_distribution)
            else:
                loss = loss_function(output, target)
            
            order_loss_H = h_o_loss(family_hyp, order)
            family_loss_H = h_f_loss(order_hyp, order, family)
            order_loss_E = e_o_loss(family_euc, order)
            family_loss_E = e_f_loss(order_euc, order, family)
            group_loss = order_loss_H + family_loss_H + family_loss_E + order_loss_E
            
            p_h = 0.01
            p_t = 1.0
            progress_bar.set_postfix(text_loss=text_loss.item(), group_loss=group_loss.item())
            
            total_loss = loss + p_h*group_loss + p_t*text_loss
            if not is_inference:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            this_rest_time = time.time() - this_rest0
            time_accum += this_load_time + this_rest_time
            time_load0 = time.time()

            tot_steps = prev_steps if is_inference else prev_steps + batch_idx

            output = corrector(output)
            
    return {}, tot_steps, species_probs

def run_GD(loader, model, loss_function, classes, opts, epoch, prev_steps, optimizer=None, is_inference=True, corrector=lambda x: x):
    """
    Runs training or inference routine for standard classification with soft-labels style losses
    """

    # Using different logging frequencies for training and validation
    log_freq = 1 if is_inference else opts.log_freq

    # strings useful for logging
    descriptor = "VAL" if is_inference else "TRAIN"
    loss_id = "loss/" + opts.loss
    dist_id = "ilsvrc_dist"

    # Initialise TensorBoard
    with_tb = opts.out_folder is not None
    
    if with_tb:
        tb_writer = tensorboardX.SummaryWriter(os.path.join(opts.out_folder, "tb", descriptor.lower()))

    # Initialise accumulators to store the several measures of performance (accumulate as sum)
    num_logged = 0
    loss_accum = 0.0
    time_accum = 0.0
    norm_mistakes_accum = 0.0
    flat_accuracy_accums = np.zeros(len(topK_to_consider), dtype=np.float64)
    hdist_accums = np.zeros(len(topK_to_consider))
    hdist_top_accums = np.zeros(len(topK_to_consider))
    hdist_mistakes_accums = np.zeros(len(topK_to_consider))
    hprecision_accums = np.zeros(len(topK_to_consider))
    hmAP_accums = np.zeros(len(topK_to_consider))
    
    species_probs = []
    
    h_o_loss = HypSupervisedContrastiveLoss()
    e_o_loss = SupervisedContrastiveLoss()
    h_f_loss = HGHypContrastiveLoss()
    e_f_loss = HGContrastiveLoss()

    # Affects the behaviour of components such as batch-norm
    if is_inference:
        model.eval()
    else:
        model.train()

    with conditional(is_inference, torch.no_grad()):
        time_load0 = time.time()
        progress_bar = tqdm(loader, desc="Processing Batches")
        for batch_idx, (embeddings, [family, target], family_embedding, target_embedding) in enumerate(progress_bar):

            this_load_time = time.time() - time_load0
            this_rest0 = time.time()

            if opts.gpu is not None:
                embeddings = embeddings.cuda(opts.gpu, non_blocking=True)

            output, text_loss, order_hyp, family_hyp, order_euc, family_euc = model(embeddings, family_embedding, target_embedding)
            species_probs.extend(F.softmax(output, dim=1).tolist())
            
            # for soft-labels we need to add a log_softmax and get the soft labels
            if opts.loss == "soft-labels":
                output = torch.nn.functional.log_softmax(output, dim=1)
                if opts.soft_labels:
                    target_distribution = make_batch_soft_labels(all_soft_labels, target, opts.num_classes, opts.batch_size, opts.gpu)
                else:
                    target_distribution = make_batch_onehot_labels(target, opts.num_classes, opts.batch_size, opts.gpu)
                loss = loss_function(output, target_distribution)
            else:
                target = target.to('cuda')
                loss = loss_function(output, target)
            
            order_loss_H = h_o_loss(family_hyp, family)
            family_loss_H = h_f_loss(order_hyp, family, target)
            order_loss_E = e_o_loss(family_euc, family)
            family_loss_E = e_f_loss(order_euc, family, target)
            group_loss = order_loss_H + family_loss_H + family_loss_E + order_loss_E
            
            p_h = 0.01
            p_t = 1.0
            progress_bar.set_postfix(text_loss=text_loss.item(), group_loss=group_loss.item())
            
            total_loss = loss  + p_t*text_loss + p_h*group_loss

            if not is_inference:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # start/reset timers
            this_rest_time = time.time() - this_rest0
            time_accum += this_load_time + this_rest_time
            time_load0 = time.time()

            # only update total number of batch visited for training
            tot_steps = prev_steps if is_inference else prev_steps + batch_idx

            # correct output of the classifier (for yolo-v2)
            output = corrector(output)
#             print(output.shape)

            # if it is time to log, compute all measures, store in summary and pass to tensorboard.
            
#     return summary, tot_steps, species_probs
    return {}, tot_steps, species_probs

def _make_best_hier_similarities(classes, distances, max_dist):
    """
    For each class, create the optimal set of retrievals (used to calculate hierarchical precision @k)
    """
    distance_matrix = np.zeros([len(classes), len(classes)])
    best_hier_similarities = np.zeros([len(classes), len(classes)])

    for i in range(len(classes)):
        for j in range(len(classes)):
            distance_matrix[i, j] = distances[(classes[i], classes[j])]

    for i in range(len(classes)):
        best_hier_similarities[i, :] = 1 - np.sort(distance_matrix[i, :]) / max_dist

    return best_hier_similarities



def _generate_summary(
        loss_accum,
        flat_accuracy_accums,
        hdist_accums,
        hdist_top_accums,
        hdist_mistakes_accums,
        hprecision_accums,
        hmAP_accums,
        num_logged,
        norm_mistakes_accum,
        loss_id,
        dist_id,
):
    """
    Generate dictionary with epoch's summary
    """
    summary = dict()
    summary[loss_id] = loss_accum / num_logged
    # -------------------------------------------------------------------------------------------------
    summary.update({accuracy_ids[i]: flat_accuracy_accums[i] / num_logged for i in range(len(topK_to_consider))})
    summary.update({dist_id + dist_avg_ids[i]: hdist_accums[i] / num_logged for i in range(len(topK_to_consider))})
    summary.update({dist_id + dist_top_ids[i]: hdist_top_accums[i] / num_logged for i in range(len(topK_to_consider))})
    summary.update(
        {dist_id + dist_avg_mistakes_ids[i]: hdist_mistakes_accums[i] / (norm_mistakes_accum * topK_to_consider[i]) for i in range(len(topK_to_consider))}
    )
    summary.update({dist_id + hprec_ids[i]: hprecision_accums[i] / num_logged for i in range(len(topK_to_consider))})
    summary.update({dist_id + hmAP_ids[i]: hmAP_accums[i] / num_logged for i in range(len(topK_to_consider))})
    return summary


def _update_tb_from_summary(summary, writer, steps, loss_id, dist_id):
    """
    Update tensorboard from the summary for the epoch
    """
    writer.add_scalar(loss_id, summary[loss_id], steps)

    for i in range(len(topK_to_consider)):
        writer.add_scalar(accuracy_ids[i], summary[accuracy_ids[i]] * 100, steps)
        writer.add_scalar(dist_id + dist_avg_ids[i], summary[dist_id + dist_avg_ids[i]], steps)
        writer.add_scalar(dist_id + dist_top_ids[i], summary[dist_id + dist_top_ids[i]], steps)
        writer.add_scalar(dist_id + dist_avg_mistakes_ids[i], summary[dist_id + dist_avg_mistakes_ids[i]], steps)
        writer.add_scalar(dist_id + hprec_ids[i], summary[dist_id + hprec_ids[i]] * 100, steps)
        writer.add_scalar(dist_id + hmAP_ids[i], summary[dist_id + hmAP_ids[i]] * 100, steps)
