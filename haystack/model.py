# haystack/model.py
# HayStack: Zero-Shot to Few-Shot Adaptive Classifier
# Supports: ResNet, ViT, DINO, DINOv2, CLIP + Full Zero-Shot (K=0)
# Paper: https://arxiv.org/abs/XXXX.XXXXX

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class HayStackMultiBackbone(nn.Module):
    """
    HayStack: Adaptive few-shot classifier with zero-shot capability.
    - K=0: Zero-shot using learned prior prototype + subspace
    - K<3: Hybrid PCA + prior
    - K≥3: Pure PCA
    """
    def __init__(
        self,
        backbone_name: str = 'resnet18',
        max_adapt_dim: int = 256,
        prior_dim: int = 32,
        sigma: float = 0.05,
        device: str = 'cpu'
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.max_adapt_dim = max_adapt_dim
        self.prior_dim = prior_dim
        self.sigma = sigma
        self.device = device

        # Detect type
        self.is_clip = backbone_name.startswith('clip_')
        self.is_dino = 'dino' in backbone_name and 'dinov2' not in backbone_name
        self.is_dinov2 = 'dinov2' in backbone_name
        self.is_vit = 'vit' in backbone_name or 'swin' in backbone_name

        # Load backbone
        self.feat, self.feat_dim = self._load_backbone(backbone_name)
        self.feat = self.feat.to(device).eval()

        # === ZERO-SHOT PRIOR (learned in real implementation) ===
        # In real system: trained on 1000+ classes
        # Here: random for demo, replace with real prior in production
        self.prior_U = torch.randn(self.feat_dim, prior_dim, requires_grad=False).to(device)
        self.prior_prototype = torch.randn(prior_dim, requires_grad=False).to(device)
        self.prior_prototype = F.normalize(self.prior_prototype, dim=0)

    def _load_backbone(self, name):
        if self.is_clip:
            import clip
            model_name = name.split('_', 1)[1]
            model, _ = clip.load(model_name, device=self.device)
            return model.visual, model.visual.output_dim

        elif self.is_dino:
            model = torch.hub.load('facebookresearch/dino:main', name, pretrained=True)
            return model, model.embed_dim if hasattr(model, 'embed_dim') else 384

        elif self.is_dinov2:
            model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
            return model, model.embed_dim

        else:
            # TorchVision
            if name == 'mobilenet_v2':
                net = models.mobilenet_v2(pretrained=True)
                return net.features, 1280
            elif name == 'efficientnet_b0':
                net = models.efficientnet_b0(pretrained=True)
                return net.features, 1280
            elif name.startswith('resnet'):
                net = getattr(models, name)(pretrained=True)
                return nn.Sequential(*list(net.children())[:-2]), net.fc.in_features
            elif name == 'vit_b_16':
                net = models.vit_b_16(pretrained=True)
                return net, 768
            elif name == 'swin_t':
                net = models.swin_t(pretrained=True)
                return net, 768
            else:
                raise ValueError(f"Unsupported backbone: {name}")

    def pyramid_pool(self, f, scales=[1, 2, 4]):
        """
        Multi-scale pyramid pooling.
        Input: [B, C, H, W] or [B, C] or [B, N, C]
        """
        if f.dim() == 2:  # [B, C]
            return F.normalize(f, dim=1)
        if f.dim() == 3:  # [B, N, C] → treat as patches
            B, N, C = f.shape
            s = int(N ** 0.5)
            if s * s == N:
                f = f.permute(0, 2, 1).view(B, C, s, s)
            else:
                return F.normalize(f.mean(1, keepdim=True), dim=1)

        B, C, H, W = f.shape
        pooled = []
        for s in scales:
            p = F.adaptive_avg_pool2d(f, s)
            p = p.view(B, C, -1).mean(dim=2)
            pooled.append(p)
        return F.normalize(torch.cat(pooled, dim=1), p=2, dim=1)

    @torch.no_grad()
    def extract_R(self, x):
        if self.is_clip:
            f = self.feat(x)
        else:
            f = self.feat(x)
        return self.pyramid_pool(f)

    def build_psi(self, R_refs, K, max_dim=None):
        """
        Build adaptive prototype Ψ
        - K=0: Zero-shot (prior prototype + prior subspace)
        - K<3: Hybrid PCA + prior
        - K≥3: Pure PCA
        """
        if max_dim is None:
            max_dim = self.max_adapt_dim
        device = R_refs.device if R_refs is not None else self.device

        if K == 0:
            # ZERO-SHOT MODE
            U_prior = self.prior_U[:, :self.prior_dim]
            Psi_prior = self.prior_prototype[:self.prior_dim].unsqueeze(0)
            Psi_prior = F.normalize(Psi_prior, dim=1)
            return Psi_prior, self.prior_dim, U_prior

        # FEW-SHOT MODE
        noise = torch.randn_like(R_refs, device=device) * self.sigma
        R_jit = F.normalize(R_refs + noise, dim=1)
        pca_dim = min(max(K * 16, 48), max_dim)

        cov = R_jit.T @ R_jit / K
        U, S, _ = torch.linalg.svd(cov, full_matrices=False)
        U_pca = U[:, :pca_dim]

        if K < 3:
            # HYBRID: PCA + PRIOR
            adapt_dim = pca_dim + self.prior_dim
            U_hybrid = torch.cat([U_pca, self.prior_U[:, :self.prior_dim]], dim=1)
        else:
            # PURE PCA
            adapt_dim = pca_dim
            U_hybrid = U_pca

        R_proj = R_jit @ U_hybrid
        Psi = R_proj.mean(0, keepdim=True)
        Psi = F.normalize(Psi, dim=1)

        return Psi, adapt_dim, U_hybrid