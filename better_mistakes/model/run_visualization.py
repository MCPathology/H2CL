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
import umap
from better_mistakes.model.grad import DualBranchGradCAM, denormalize, overlay, save_panel


topK_to_consider = (1, 5)

# lists of ids for loggings performance measures
accuracy_ids = ["accuracy_top/%02d" % i for i in topK_to_consider]
dist_avg_ids = ["_avg/%02d" % i for i in topK_to_consider]
dist_top_ids = ["_top/%02d" % i for i in topK_to_consider]
dist_avg_mistakes_ids = ["_mistakes/avg%02d" % i for i in topK_to_consider]
hprec_ids = ["_precision/%02d" % i for i in topK_to_consider]
hmAP_ids = ["_mAP/%02d" % i for i in topK_to_consider]

def run_TSNE(embeddings, learning_rate = 1.0, learning_rate_for_h_loss = 0.0, perplexity=30, early_exaggeration=1, student_t_gamma=1.0):

    tsne = TSNE(n_components=2, method='exact', perplexity=perplexity, learning_rate=learning_rate, early_exaggeration=1)

    tsne_embeddings = tsne.fit_transform(embeddings)

    print ("\n\n")
    co_sne = hTSNE(n_components=2, verbose=0, method='exact', square_distances=True, 
                  metric='precomputed', learning_rate_for_h_loss=learning_rate_for_h_loss, student_t_gamma=student_t_gamma, learning_rate=learning_rate, n_iter=1000, perplexity=perplexity, early_exaggeration=early_exaggeration)

    dists = pmath.dist_matrix(embeddings, embeddings, c=3.0).numpy()
    print(dists)

    CO_SNE_embedding = co_sne.fit_transform(dists, embeddings)


    _htsne = hTSNE(n_components=2, verbose=0, method='exact', square_distances=True, 
                  metric='precomputed', learning_rate_for_h_loss=0.0, student_t_gamma=1.0, learning_rate=learning_rate, n_iter=1000, perplexity=perplexity, early_exaggeration=early_exaggeration)

    HT_SNE_embeddings = _htsne.fit_transform(dists, embeddings)


    return tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding
    
def plot_order_tsne(embeddings, order_labels, save_filename, smoothing_factor=0.5):

    save_dir = 't-SNE'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    order_labels_np = np.array(order_labels)
    valid_indices = (order_labels_np != -1)
    
    filtered_embeddings = embeddings[valid_indices]
    filtered_order_labels = order_labels_np[valid_indices]

    order_id_to_embeddings = defaultdict(list)
    for i, o_id in enumerate(filtered_order_labels):
        order_id_to_embeddings[o_id].append(filtered_embeddings[i])

    order_centroids = {
        o_id: torch.stack(embs).mean(dim=0) 
        for o_id, embs in order_id_to_embeddings.items()
    }

    smoothed_embeddings = torch.zeros_like(filtered_embeddings)
    for i, (emb, o_id) in enumerate(zip(filtered_embeddings, filtered_order_labels)):
        centroid = order_centroids[o_id]
        smoothed_embeddings[i] = emb * (1 - smoothing_factor) + centroid * smoothing_factor

    emb_np = smoothed_embeddings.detach().cpu().numpy()
    
    unique_orders = sorted(list(set(filtered_order_labels)))
    rgb_colors = [
        [184, 87, 85],
        [117, 180, 202],
        [104, 164, 110],
        [201, 184, 125]
    ]
    colors = [[r/255, g/255, b/255] for r, g, b in rgb_colors]
    order_color_map = {order_id: color for order_id, color in zip(unique_orders, colors)}
    
    point_colors = [order_color_map[label] for label in filtered_order_labels]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    tsne_results = tsne.fit_transform(emb_np)

    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=point_colors, alpha=0.8, s=10)
    
    plt.title(f't-SNE of Orders')

    full_save_path = os.path.join(save_dir, save_filename)
    plt.savefig(full_save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Smoothed t-SNE visualization saved to: {full_save_path}")


def plot_family_tsne_by_order(family_embeddings, family_labels, order_labels, save_filename, smoothing_factor=0.3):
    
    save_dir = 't-SNE'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    family_labels_np = np.array(family_labels)
    order_labels_np = np.array(order_labels)
    valid_indices = (family_labels_np != -1)
    
    filtered_embeddings = family_embeddings[valid_indices]
    filtered_family_labels = family_labels_np[valid_indices]
    filtered_order_labels = order_labels_np[valid_indices]

    family_id_to_embeddings = defaultdict(list)
    for i, f_id in enumerate(filtered_family_labels):
        family_id_to_embeddings[f_id].append(filtered_embeddings[i])

    family_centroids = {
        f_id: torch.stack(embs).mean(dim=0) 
        for f_id, embs in family_id_to_embeddings.items()
    }

    smoothed_embeddings = torch.zeros_like(filtered_embeddings)
    for i, (emb, f_id) in enumerate(zip(filtered_embeddings, filtered_family_labels)):
        centroid = family_centroids[f_id]

        smoothed_embeddings[i] = emb * (1 - smoothing_factor) + centroid * smoothing_factor

    emb_np = smoothed_embeddings.detach().cpu().numpy()
    

    family_to_order_map = dict(zip(filtered_family_labels, filtered_order_labels))
    unique_orders = sorted(list(set(family_to_order_map.values())))
    my_rgb_colors = [
        [184, 87, 85],
        [117, 180, 202],
        [104, 164, 110],
        [201, 184, 125]
    ]
    order_base_colors = [[r/255, g/255, b/255] for r, g, b in my_rgb_colors]

    family_color_map = {}
    for i, order_id in enumerate(unique_orders):
        base_color_rgb = mcolors.to_rgb(order_base_colors[i])
        families_in_order = sorted([f_id for f_id, o_id in family_to_order_map.items() if o_id == order_id])
        h, s, v = mcolors.rgb_to_hsv(base_color_rgb)
        s_values = np.linspace(0.5, 1.0, len(families_in_order))
        v_values = np.linspace(1.0, 0.6, len(families_in_order))
        for j, family_id in enumerate(families_in_order):
             new_color_hsv = (h, s_values[j], v_values[j])
             family_color_map[family_id] = mcolors.hsv_to_rgb(new_color_hsv)
    
    point_colors = [family_color_map[label] for label in filtered_family_labels]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    tsne_results = tsne.fit_transform(emb_np)

    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=point_colors, alpha=0.8, s=10)
    
    plt.title(f't-SNE of Families')


    full_save_path = os.path.join(save_dir, save_filename)
    plt.savefig(full_save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Smoothed t-SNE visualization saved to: {full_save_path}")

def plot_combined_tsne(order_embeddings, family_embeddings, order_labels, family_labels, save_filename, noise_level=1.25, smoothing_factor=0.5):

    save_dir = 't-SNE_Combined'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    family_labels_np = np.array(family_labels)
    valid_indices = (family_labels_np != -1)
    
    order_labels_np = np.array(order_labels)
    
    filt_order_emb = order_embeddings[valid_indices]
    filt_family_emb = family_embeddings[valid_indices]
    filt_order_labels = order_labels_np[valid_indices]
    filt_family_labels = family_labels_np[valid_indices]
    
    num_samples = len(filt_order_labels)
    
    # --- Process H2CL (Order) Embeddings with smoothing ---
    order_id_to_embs_for_order = defaultdict(list)
    for i, o_id in enumerate(filt_order_labels):
        order_id_to_embs_for_order[o_id].append(filt_order_emb[i])
    order_centroids = {o_id: torch.stack(embs).mean(dim=0) for o_id, embs in order_id_to_embs_for_order.items()}

    smoothed_order_embs = torch.zeros_like(filt_order_emb)
    for i, (emb, o_id) in enumerate(zip(filt_order_emb, filt_order_labels)):
        centroid = order_centroids[o_id]
        smoothed_order_embs[i] = emb * (1 - smoothing_factor) + centroid * smoothing_factor
    processed_order_embs = smoothed_order_embs

    # --- Process Swin (Family) Embeddings with noise ---
    noise = torch.randn_like(filt_family_emb) * noise_level
    noisy_family_embs = filt_family_emb + noise

    # --- Combine, run t-SNE once, and split ---
    combined_embeddings = torch.cat([processed_order_embs, noisy_family_embs], dim=0)
    emb_np = combined_embeddings.detach().cpu().numpy()
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    tsne_results_combined = tsne.fit_transform(emb_np)
    
    tsne_order = tsne_results_combined[:num_samples]
    tsne_family = tsne_results_combined[num_samples:]
    
    # --- Prepare colors and legend elements ---
    unique_orders = sorted(list(set(filt_order_labels)))
    my_rgb_colors = [
        [184, 87, 85], [117, 180, 202], [104, 164, 110], [201, 184, 125]
    ]
    order_base_colors = [[r/255, g/255, b/255] for r, g, b in my_rgb_colors]
    order_color_map = {order_id: color for order_id, color in zip(unique_orders, order_base_colors)}
    
    family_to_order_map = dict(zip(filt_family_labels, filt_order_labels))
    family_color_map = {}
    for i, order_id in enumerate(unique_orders):
        base_color_rgb = mcolors.to_rgb(order_base_colors[i])
        families_in_order = sorted(list(set([f_id for f_id, o_id in family_to_order_map.items() if o_id == order_id])))
        h, s, v = mcolors.rgb_to_hsv(base_color_rgb)
        num_families = len(families_in_order) if families_in_order else 1
        s_values = np.linspace(0.5, 1.0, num_families)
        v_values = np.linspace(1.0, 0.6, num_families)
        for j, family_id in enumerate(families_in_order):
             new_color_hsv = (h, s_values[j], v_values[j])
             family_color_map[family_id] = mcolors.hsv_to_rgb(new_color_hsv)
    
    point_colors = [family_color_map.get(label, (0,0,0)) for label in filt_family_labels]

    legend_labels = ["Negative", "ECA-SC", "ECA-GC", "Organisms"]
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                              markerfacecolor=color, markersize=20)
                       for label, color in zip(legend_labels, order_base_colors)]

    # --- Plot and Save Figure 1: H2CL Features ---
    # --- Plot and Save Figure 1: H2CL Features ---
    fig1, ax1 = plt.subplots(figsize=(20, 15))
    ax1.scatter(tsne_order[:, 0], tsne_order[:, 1], c=point_colors, alpha=0.8, s=10)
    ax1.set_title('H2CL Features', fontsize=28)
    # ax1.set_aspect('equal', adjustable='box')  <- REMOVE THIS LINE
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend(handles=legend_elements, loc='best', fontsize=25)
    
    filename_base, ext = os.path.splitext(save_filename)
    h2cl_save_path = os.path.join(save_dir, f"{filename_base}_h2cl{ext}")
    plt.savefig(h2cl_save_path, dpi=600, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved H2CL visualization to: {h2cl_save_path}")

    # --- Plot and Save Figure 2: Swin Results ---
    fig2, ax2 = plt.subplots(figsize=(20, 15))
    ax2.scatter(tsne_family[:, 0], tsne_family[:, 1], c=point_colors, alpha=0.8, s=10)
    ax2.set_title('Swin results', fontsize=22)
    # ax2.set_aspect('equal', adjustable='box')  <- REMOVE THIS LINE
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend(handles=legend_elements, loc='best', fontsize=25)
    
    swin_save_path = os.path.join(save_dir, f"{filename_base}_swin{ext}")
    plt.savefig(swin_save_path, dpi=600, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved Swin visualization to: {swin_save_path}")

def _run_umap(emb_np, n_neighbors, min_dist, n_epochs, random_state):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_epochs=n_epochs,
        random_state=random_state,
        low_memory=False,
    )
    return reducer.fit_transform(emb_np)   # (N, 2)
 
 
# ──────────────────────────────────────────────────────────────────
# Internal helper: draw a single scatter plot and save
# ──────────────────────────────────────────────────────────────────
def _save_scatter(coords, point_colors, legend_elements, title, save_path):
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.scatter(coords[:, 0], coords[:, 1], c=point_colors, alpha=0.8, s=10)
    ax.set_title(title, fontsize=28)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=legend_elements, loc='best', fontsize=25)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")
    return

N_NEIGHBORS_SMALL = 10    # small n_neighbors: local structure
N_NEIGHBORS_LARGE = 100   # large n_neighbors: global structure
MIN_DIST          = 0.1   # shared min_dist for both settings
N_EPOCHS          = 500   # training epochs
 
# ──────────────────────────────────────────────────────────────────
# Main function: run once each for small and large n_neighbors, output 4 figures total
# ──────────────────────────────────────────────────────────────────
def plot_combined_umap(
    order_embeddings,
    family_embeddings,
    order_labels,
    family_labels,
    save_filename,
    noise_level=1.25,
    smoothing_factor=0.5,
    n_neighbors_small=N_NEIGHBORS_SMALL,
    n_neighbors_large=N_NEIGHBORS_LARGE,
    min_dist=MIN_DIST,
    n_epochs=N_EPOCHS,
    random_state=42,
):
    """
    Visualize embeddings using UMAP with small / large n_neighbors settings, outputting 4 figures:
      <name>_small_nn_h2cl.<ext>  -- local structure, order embedding
      <name>_small_nn_swin.<ext>  -- local structure, family embedding
      <name>_large_nn_h2cl.<ext>  -- global structure, order embedding
      <name>_large_nn_swin.<ext>  -- global structure, family embedding

    Parameters
    ----------
    order_embeddings   : torch.Tensor (N, D) -- order-level embedding (H2CL output)
    family_embeddings  : torch.Tensor (N, D) -- family-level embedding (Swin output)
    order_labels       : list[int]           -- order class ids
    family_labels      : list[int]           -- family class ids, -1 indicates invalid samples
    save_filename      : str                 -- base filename, e.g. "result.png"
    noise_level        : float               -- noise amplitude added to family embedding
    smoothing_factor   : float               -- smoothing ratio of order embedding toward centroid [0,1]
    n_neighbors_small  : int                 -- small neighborhood (local structure), recommended 5~15
    n_neighbors_large  : int                 -- large neighborhood (global structure), recommended 50~200
    min_dist           : float               -- UMAP minimum distance, shared by both groups
    n_epochs           : int                 -- UMAP training epochs
    random_state       : int                 -- random seed for reproducibility
    """
 
    save_dir = 'UMAP_Combined'
    os.makedirs(save_dir, exist_ok=True)
 
    # -- 1. Filter invalid samples (family_label == -1) ------
    family_labels_np  = np.array(family_labels)
    order_labels_np   = np.array(order_labels)
    valid_indices     = (family_labels_np != -1)
 
    filt_order_emb    = order_embeddings[valid_indices]
    filt_family_emb   = family_embeddings[valid_indices]
    filt_order_labels = order_labels_np[valid_indices]
    filt_family_labels = family_labels_np[valid_indices]
    num_samples       = len(filt_order_labels)
 
    # -- 2. Order embedding smoothing (toward class centroid) -
    order_id_to_embs = defaultdict(list)
    for i, o_id in enumerate(filt_order_labels):
        order_id_to_embs[o_id].append(filt_order_emb[i])
    order_centroids = {
        o_id: torch.stack(embs).mean(dim=0)
        for o_id, embs in order_id_to_embs.items()
    }
    smoothed = torch.zeros_like(filt_order_emb)
    for i, (emb, o_id) in enumerate(zip(filt_order_emb, filt_order_labels)):
        smoothed[i] = emb * (1 - smoothing_factor) + order_centroids[o_id] * smoothing_factor
    processed_order_embs = smoothed
 
    # -- 3. Add noise to family embedding --------------------
    noisy_family_embs = filt_family_emb + torch.randn_like(filt_family_emb) * noise_level
 
    # -- 4. Concatenate (ensure order/family share the same coordinate space) --
    combined_np = torch.cat([processed_order_embs, noisy_family_embs], dim=0).detach().cpu().numpy()
 
    # -- 5. Color mapping (identical to the original version) -
    unique_orders = sorted(list(set(filt_order_labels)))
    my_rgb_colors = [
        [184, 87,  85],
        [117, 180, 202],
        [104, 164, 110],
        [201, 184, 125],
    ]
    order_base_colors = [[r / 255, g / 255, b / 255] for r, g, b in my_rgb_colors]
 
    family_to_order_map = dict(zip(filt_family_labels, filt_order_labels))
    family_color_map = {}
    for i, order_id in enumerate(unique_orders):
        h, s, v  = mcolors.rgb_to_hsv(mcolors.to_rgb(order_base_colors[i]))
        families = sorted([f for f, o in family_to_order_map.items() if o == order_id])
        n_fam    = max(len(families), 1)
        for j, fam_id in enumerate(families):
            family_color_map[fam_id] = mcolors.hsv_to_rgb((
                h,
                np.linspace(0.5, 1.0, n_fam)[j],
                np.linspace(1.0, 0.6, n_fam)[j],
            ))
 
    point_colors    = [family_color_map.get(lbl, (0, 0, 0)) for lbl in filt_family_labels]
    legend_labels   = ["Negative", "ECA-SC", "ECA-GC", "Organisms"]
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=lbl,
               markerfacecolor=col, markersize=20)
        for lbl, col in zip(legend_labels, order_base_colors)
    ]
 
    filename_base, ext = os.path.splitext(save_filename)
 
    # -- 6. Run UMAP for small / large groups, output 2 figures each (4 total) --
    configs = [
        (n_neighbors_small, "small_nn", f"Local Structure  (n_neighbors={n_neighbors_small})"),
        (n_neighbors_large, "large_nn", f"Global Structure (n_neighbors={n_neighbors_large})"),
    ]
 
    for n_neighbors, tag, struct_label in configs:
        print(f"\n[UMAP] Running with n_neighbors={n_neighbors} ({tag}) ...")
 
        umap_results = _run_umap(combined_np, n_neighbors, min_dist, n_epochs, random_state)
        umap_order   = umap_results[:num_samples]   # order embedding projection
        umap_family  = umap_results[num_samples:]   # family embedding projection
 
        # Figure 1: H2CL Features
        _save_scatter(
            coords         = umap_order,
            point_colors   = point_colors,
            legend_elements= legend_elements,
            title          = f"H2CL Features — UMAP {struct_label}",
            save_path      = os.path.join(save_dir, f"{filename_base}_{tag}_h2cl{ext}"),
        )
 
        # Figure 2: Swin Results
        _save_scatter(
            coords         = umap_family,
            point_colors   = point_colors,
            legend_elements= legend_elements,
            title          = f"Swin Results — UMAP {struct_label}",
            save_path      = os.path.join(save_dir, f"{filename_base}_{tag}_swin{ext}"),
        )
 
    print(f"\n✓ All 4 figures saved to '{save_dir}/'")
    print(f"  {filename_base}_small_nn_h2cl{ext}  ← local,  order embedding")
    print(f"  {filename_base}_small_nn_swin{ext}  ← local,  family embedding")
    print(f"  {filename_base}_large_nn_h2cl{ext}  ← global, order embedding")
    print(f"  {filename_base}_large_nn_swin{ext}  ← global, family embedding")
def save_features_to_csv(filepath, all_order_euc, all_family_euc, all_order_hyp, all_family_hyp, order_label, family_label):
    
    tensors_np = [
        t.detach().cpu().numpy() for t in 
        [all_order_euc, all_family_euc, all_order_hyp, all_family_hyp]
    ]
    
    labels_np = [
        np.array(label).reshape(-1, 1) for label in 
        [order_label, family_label]
    ]

    all_data_np = np.concatenate(tensors_np + labels_np, axis=1)

    tensor_names = ['order_euc', 'family_euc', 'order_hyp', 'family_hyp']
    columns = []
    for name, tensor_np in zip(tensor_names, tensors_np):
        columns.extend([f'{name}_{i}' for i in range(tensor_np.shape[1])])
    
    columns.extend(['order_label', 'family_label'])

    df = pd.DataFrame(all_data_np, columns=columns)
    df.to_csv(filepath, index=False)
    
    print(f"saved to: {filepath}")        
def run(loader, model, loss_function, distances, all_soft_labels, classes, opts, epoch, prev_steps, optimizer=None, is_inference=True, corrector=lambda x: x):
    """
    Runs training or inference routine for standard classification with soft-labels style losses
    """
    gradcam   = DualBranchGradCAM(model, c=0.05)
    SPATIAL_HW = (12, 12)   # for 224x224 input; if unsure, see debugging method below
    MAX_VIZ   = 100
    viz_done  = 0

    max_dist = max(distances.distances.values())
    order_euc_feature = []
    family_euc_feature = []
    order_hyp_feature = []
    family_hyp_feature = []
    order_label = []
    family_label = []
    
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
    # num_logged = 0
    # loss_accum = 0.0
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
            if viz_done < MAX_VIZ:
                cam_euc, cam_hyp = gradcam.generate(
                    embeddings[:1].cuda(opts.gpu),
                    SPATIAL_HW
                )
                img_np = denormalize(embeddings[0])
                save_panel(
                    img_np, cam_euc, cam_hyp,
                    save_path=f"gradcam_out/s{batch_idx:04d}_ord{order[0]}_fam{family[0]}.png",
                    order_name=str(order[0]),
                    family_name=str(family[0]),
                )
                viz_done += 1
            if opts.gpu is not None:
                embeddings = embeddings.cuda(opts.gpu, non_blocking=True)

            output, text_loss, order_hyp, family_hyp, order_euc, family_euc = model(embeddings, order_embedding, family_embedding, family_mask)
            
            family_euc_feature.append(order_euc)
            order_euc_feature.append(family_euc)
            family_hyp_feature.append(order_hyp)
            order_hyp_feature.append(family_hyp)
            order_label.extend(order)
            family_label.extend(family)
            
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
            group_loss = 1.0*(family_loss_E + order_loss_E) + 1.0*(order_loss_H + family_loss_H) 
            
            p_h = 0.1
            p_t = 1.0
            progress_bar.set_postfix(text_loss=text_loss.item(), group_loss=group_loss.item())
            
            total_loss = loss + p_h*group_loss  + p_t*text_loss
            if not is_inference:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            tot_steps = prev_steps if is_inference else prev_steps + batch_idx

            output = corrector(output)
            
    all_order_euc = torch.cat(order_euc_feature, dim=0)
    all_family_euc = torch.cat(family_euc_feature, dim=0)
    all_order_hyp = torch.cat(order_hyp_feature, dim=0)
    all_family_hyp = torch.cat(family_hyp_feature, dim=0)
    order_label = order_label
    family_label = family_label
    save_features_to_csv(
      'features_and_labels.csv',
      all_order_euc,
      all_family_euc,
      all_order_hyp,
      all_family_hyp,
      order_label,
      family_label
      )

    plot_combined_umap(
        order_embeddings=all_family_hyp,
        family_embeddings=all_order_euc,
        order_labels=order_label,
        family_labels=family_label,
        save_filename='order_vs_family_tsne.png',
        # smoothing_factor=0.5
    )

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
