# encoding: utf-8
"""
Prototype-Anchored Contrastive Refinement (PACR) Loss
======================================================

A modern training-time loss designed to internalize the benefits of NFC
(Neighbor Feature Centralization) into the model's feature space.

Key Design Choices (2024-2025 best practices):
1. EMA Prototypes — not learned via backprop, updated via momentum averaging.
   More stable than center loss's learnable centers.
2. Cosine similarity — not L2 distance. This is CRITICAL: center loss uses
   L2 distance which forces absolute position collapse. Cosine only cares 
   about angular alignment, preserving discriminative spread.
3. Hard Negative Mining — only backprops through the most confusing negative
   prototypes, providing sharper gradients and faster convergence.
4. Contrastive (InfoNCE) formulation — doesn't just pull toward own prototype,
   but actively pushes away from wrong prototypes. This is complementary to
   triplet loss (which operates on instances, not prototypes).

Why center loss fails but this should work:
  Center loss: L = ||f - c_y||^2  → forces absolute spatial collapse
  PACR:        L = -log(softmax(cos(f, p_y)/τ))  → only requires RANKING
               (be more similar to your prototype than to others)

References:
  - SPRED (ICCV 2025): Self-Reinforcing Prototype Evolution
  - SAC Loss (2024): Semantic Alignment Contrastive for cloth-invariant ReID
  - Proxy Anchor (CVPR 2020) + Progressive Propagation (ECCV 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeAnchoredContrastiveLoss(nn.Module):
    """
    Prototype-Anchored Contrastive Refinement (PACR) loss.
    
    Maintains EMA-updated identity prototypes and trains the model to
    produce features that rank their own prototype highest among all
    prototypes — the training-time equivalent of NFC.
    
    Args:
        num_classes: Number of identity classes
        feat_dim: Feature dimension (1024 for EVA-02)
        momentum: EMA momentum for prototype updates (higher = slower updates)
        temperature: Softmax temperature (lower = tighter clusters)
        hard_negatives: Number of hardest negative prototypes to use (0 = all)
        warmup_epochs: Epochs before PACR loss activates (let base losses stabilize first)
    """
    
    def __init__(self, num_classes, feat_dim=1024, momentum=0.999,
                 temperature=0.07, hard_negatives=32, warmup_epochs=5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.temperature = temperature
        self.hard_negatives = min(hard_negatives, num_classes - 1)
        self.warmup_epochs = warmup_epochs
        
        # Prototype bank — registered as buffer (not optimized, persisted in state_dict)
        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes, dtype=torch.long))
        
    @torch.no_grad()
    def _initialize_prototype(self, features, label_idx):
        """First-time initialization: set prototype directly from features."""
        centroid = F.normalize(features.mean(dim=0), dim=0)
        self.prototypes[label_idx] = centroid
        self.prototype_counts[label_idx] = 1
    
    @torch.no_grad()
    def _ema_update_prototype(self, features, label_idx):
        """EMA update: smoothly blend new observations into prototype."""
        centroid = F.normalize(features.mean(dim=0), dim=0)
        self.prototypes[label_idx] = (
            self.momentum * self.prototypes[label_idx] +
            (1 - self.momentum) * centroid
        )
        self.prototypes[label_idx] = F.normalize(self.prototypes[label_idx], dim=0)
        self.prototype_counts[label_idx] += 1
    
    @torch.no_grad()
    def update_prototypes(self, features, labels):
        """Update all prototypes seen in this batch."""
        features = F.normalize(features.detach(), dim=1)
        for label in labels.unique():
            mask = (labels == label)
            label_idx = label.item()
            batch_features = features[mask]
            
            if self.prototype_counts[label_idx] == 0:
                self._initialize_prototype(batch_features, label_idx)
            else:
                self._ema_update_prototype(batch_features, label_idx)
    
    def forward(self, features, labels, current_epoch=0):
        """
        Compute PACR loss with gradual ramp-up.
        
        Args:
            features: [B, D] raw bio features (before any projection)
            labels: [B] identity labels (0-indexed)
            current_epoch: current training epoch for warmup scheduling
            
        Returns:
            loss: scalar tensor
        """
        # Phase 1: Pure warmup — only collect prototypes, zero loss
        if current_epoch < self.warmup_epochs:
            self.update_prototypes(features.detach(), labels)
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        features_norm = F.normalize(features, dim=1)  # [B, D]
        
        # Check that we have valid prototypes for all labels in batch
        active_mask = self.prototype_counts > 0
        if not active_mask[labels].all():
            # Some labels haven't been seen yet — update and skip
            self.update_prototypes(features.detach(), labels)
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # Cast prototypes to match feature dtype (AMP compatibility)
        # .clone() is CRITICAL: we update prototypes in-place after loss computation,
        # so we need a separate copy for the autograd graph
        prototypes = self.prototypes.clone().to(dtype=features_norm.dtype)
        
        # Similarity to all active prototypes: [B, num_classes]
        sim = features_norm @ prototypes.T / self.temperature
        
        # Hard Negative Mining: select the most confusing negatives
        if self.hard_negatives > 0 and self.hard_negatives < self.num_classes - 1:
            # Get positive similarity for each sample
            pos_sim = sim.gather(1, labels.unsqueeze(1))  # [B, 1]
            
            # Create negative mask (exclude own prototype)
            neg_mask = torch.ones_like(sim, dtype=torch.bool)
            neg_mask.scatter_(1, labels.unsqueeze(1), False)
            
            # For each sample, find top-k hardest (most similar) negatives
            neg_sim = sim.clone()
            neg_sim[~neg_mask] = float('-inf')  # mask positives
            hard_neg_sim, _ = neg_sim.topk(self.hard_negatives, dim=1)  # [B, k]
            
            # Combine: positive at index 0, then hard negatives
            logits = torch.cat([pos_sim, hard_neg_sim], dim=1)  # [B, 1+k]
            target = torch.zeros(features.size(0), dtype=torch.long, 
                               device=features.device)  # positive is at idx 0
            
            loss = F.cross_entropy(logits, target)
        else:
            # Standard softmax over all prototypes
            loss = F.cross_entropy(sim, labels)
        
        # Phase 2: Gradual ramp-up — scale loss from 0→1 over ramp_epochs
        # This prevents the sudden activation dip observed when PACR kicks in
        ramp_epochs = 15  # gradually increase over 15 epochs after warmup
        epochs_since_warmup = current_epoch - self.warmup_epochs
        if epochs_since_warmup < ramp_epochs:
            import math
            # Cosine ramp: 0 → 1 smoothly
            ramp_factor = 0.5 * (1 - math.cos(math.pi * epochs_since_warmup / ramp_epochs))
            loss = loss * ramp_factor
        
        # Update prototypes after loss computation (detached)
        self.update_prototypes(features.detach(), labels)
        
        return loss


class CrossInstanceConsistencyLoss(nn.Module):
    """
    Cross-Instance Consistency (CIC) regularizer.
    
    For each identity in the batch (which has NUM_INSTANCES samples),
    encourages pairwise similarities to be HIGH and UNIFORM.
    
    This is complementary to PACR: while PACR aligns features to prototypes,
    CIC ensures that multiple views of the same identity are consistent
    with each other (reducing intra-class variance without collapse).
    
    Unlike center loss (which forces L2 distance to center → collapse),
    CIC uses cosine similarity with a soft target (mean similarity),
    which preserves the spread while reducing outliers.
    """
    
    def __init__(self, target_similarity=0.85):
        super().__init__()
        self.target_similarity = target_similarity
    
    def forward(self, features, labels):
        """
        Args:
            features: [B, D] bio features
            labels: [B] identity labels
        Returns:
            loss: scalar
        """
        features_norm = F.normalize(features, dim=1)
        loss = torch.tensor(0.0, device=features.device)
        count = 0
        
        for label in labels.unique():
            mask = (labels == label)
            if mask.sum() < 2:
                continue
            
            # Pairwise cosine similarities among same-identity features
            group_feats = features_norm[mask]  # [n, D]
            sim_matrix = group_feats @ group_feats.T  # [n, n]
            
            # Exclude self-similarity (diagonal)
            n = group_feats.size(0)
            off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=features.device)
            pairwise_sims = sim_matrix[off_diag_mask]
            
            # Push pairwise similarities toward target
            # MSE loss: (sim - target)^2
            loss += ((pairwise_sims - self.target_similarity) ** 2).mean()
            count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss
