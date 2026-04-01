import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class DualBranchGradCAM:
    def __init__(self, model, c=0.05):
        """
        model: Your SwinT instance (with modified head)
        hook target: model.layers[3], output (B, L, C), L = H*W
        """
        self.model = model
        self.c = c
        self._feat = None   # forward hook stores feature (retains computation graph)
        self._grad = None   # backward hook stores gradient

        # layers[3] is the entire stage; attach hook at stage output
        # SwinT BasicLayer output is the result of the last block
        self._fh = model.backbone.layers[3].register_forward_hook(self._save_feat)
        self._bh = model.backbone.layers[3].register_full_backward_hook(self._save_grad)

    def _save_feat(self, m, inp, out):
        # out: (B, L, C), do not detach to retain computation graph for gradient backpropagation
        self._feat = out

    def _save_grad(self, m, gin, gout):
        self._grad = gout[0].detach()  # (B, L, C)

    def remove_hooks(self):
        self._fh.remove()
        self._bh.remove()

    def _logmap0(self, x):
        """Poincare ball -> tangent space (logarithmic map at origin)"""
        sqrt_c = self.c ** 0.5
        norm = x.norm(dim=-1, keepdim=True).clamp(1e-8, 1 / sqrt_c - 1e-5)
        return (1.0 / sqrt_c) * torch.arctanh(sqrt_c * norm) * x / norm

    def _build_cam(self, grad, feat, spatial_hw):
        """
        grad, feat: (B, L, C)
        spatial_hw: (H, W), satisfying H*W == L
        Returns: (H, W) numpy float [0,1]
        """
        H, W = spatial_hw
        B, L, C = feat.shape

        # feat is still in the computation graph; detach before further operations
        f = feat.detach().reshape(B, H, W, C).permute(0, 3, 1, 2)  # B,C,H,W
        g = grad.reshape(B, H, W, C).permute(0, 3, 1, 2)

        weights = g.mean(dim=(2, 3), keepdim=True)       # B,C,1,1
        cam = F.relu((weights * f).sum(dim=1))            # B,H,W
        cam = cam[0].cpu().float().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    @torch.enable_grad()
    def generate(self, x, spatial_hw):
        """
        x:           (1, 3, H_img, W_img) single image, cuda tensor
        spatial_hw:  spatial resolution of layers[3] output
                     input 224 -> (7,7), 384 -> (12,12), 448 -> (14,14)
                     if unsure, run once and check _feat.shape

        Returns:
            cam_euc (H,W), cam_hyp (H,W)
        """
        self.model.eval()
        x = x.float()
        x.requires_grad_(True)

        # -- forward: obtain 4 embeddings -------------------------
        # directly call model.forward, which triggers the hook internally
        order_hyp, family_hyp, order_euc, family_euc = self.model.backbone(x)
        # corresponds to return: order_embedding_H, family_embedding_H,
        #                   order_embedding_E, family_embedding_E

        # -- first backward pass: Euclidean branch ----------------
        self.model.zero_grad()
        score_euc = order_euc[0].norm()          # scalar
        score_euc.backward(retain_graph=True)
        grad_euc = self._grad.clone()            # hook has been updated

        # -- second backward pass: Hyperbolic branch --------------
        # order_hyp is on the Poincare ball; project to tangent space to remove spherical metric effects
        # but backpropagation still starts from order_hyp (it is connected to the computation graph)
        self.model.zero_grad()
        order_hyp_tan = self._logmap0(order_hyp)  # tangent space projection
        score_hyp = order_hyp_tan[0].norm()       # scalar
        score_hyp.backward(retain_graph=False)
        grad_hyp = self._grad.clone()

        # -- generate CAM ------------------------------------------
        feat = self._feat  # stored by forward hook, (B, L, C)
        cam_euc = self._build_cam(grad_euc, feat, spatial_hw)
        cam_hyp = self._build_cam(grad_hyp, feat, spatial_hw)

        return cam_euc, cam_hyp


# -- visualization utility functions ----------------------
def denormalize(t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = t.clone().cpu().float()
    for c, m, s in zip(range(3), mean, std):
        img[c] = img[c] * s + m
    return img.permute(1, 2, 0).numpy().clip(0, 1)

def overlay(img_np, cam, alpha=0.45):
    H, W = img_np.shape[:2]
    cam_up = cv2.resize(cam, (W, H))
    heat = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip((1 - alpha) * img_np + alpha * heat, 0, 1)

def save_panel(img_np, cam_euc, cam_hyp, save_path,
               order_name="", family_name=""):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].imshow(img_np)
    # axes[0].set_title("Original", fontsize=11)
    axes[1].imshow(overlay(img_np, cam_euc))
    # axes[1].set_title("Euclidean branch\n(order_embedding_E, i=2)",
    #                   fontsize=10, color="#185FA5")
    axes[2].imshow(overlay(img_np, cam_hyp))
    # axes[2].set_title("Hyperbolic branch\n(order_embedding_H, i=4)",
    #                   fontsize=10, color="#993C1D")
    for ax in axes:
        ax.axis("off")
    # fig.suptitle(f"Order: {order_name}   Family: {family_name}",
    #              fontsize=11, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)