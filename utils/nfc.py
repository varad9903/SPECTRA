"""
Neighbor Feature Centralization (NFC) for test-time feature enhancement.

Three modes:
1. Fixed NFC — Original Pose2ID (Yuan et al., CVPR 2025) with fixed k1/k2
2. Adaptive NFC v2 (ANFC) — Large search window + mutual check only (no threshold)
   + blending factor to preserve discriminability
Both are TRAINING-FREE, TEST-TIME ONLY enhancements.
"""

import torch


def pairwise_distance(x, y):
    """Compute pairwise Euclidean distance between two feature sets."""
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def NFC(feat, k1=2, k2=2):
    """Fixed-k Neighbor Feature Centralization (original Pose2ID).
    
    For each feature, find mutual nearest neighbors (samples that consider
    each other as top-k neighbors). Aggregate mutual neighbors' features
    to centralize toward the identity center.
    
    Args:
        feat: [N, D] tensor of L2-normalized features
        k1: number of top neighbors to consider for each sample
        k2: number of top neighbors to verify mutual relationship
    
    Returns:
        [N, D] tensor of centralized features (NOT re-normalized — caller should normalize)
    """
    feat = feat.clone()
    
    # Compute pairwise distances within the set
    dist = pairwise_distance(feat.to('cuda'), feat.to('cuda')).to('cpu')
    
    # Mask self-distances
    eye = torch.eye(dist.size(0)).to(dist.device)
    dist[eye == 1] = 1000
    
    # Find top-k1 nearest neighbors
    val, rank = dist.topk(k1, largest=False)
    
    # Mutual neighbor check: j is a mutual neighbor of i only if
    # j is in i's top-k1 AND i is in j's top-k2
    mutual_topk_list = []
    for i in range(rank.size(0)):
        mutual_list = []
        for j in rank[i]:
            if i in rank[j][:k2]:
                mutual_list.append(j.item())
        mutual_topk_list.append(mutual_list)
    
    # Aggregate: add mutual neighbors' features
    feat_copy = feat.clone()
    for i in range(rank.size(0)):
        if mutual_topk_list[i]:  # only if there are mutual neighbors
            feat[i] += feat_copy[mutual_topk_list[i]].sum(dim=0)
    
    return feat


def AdaptiveNFC(feat, max_k=20, alpha=0.7):
    """Adaptive Neighbor Feature Centralization v2 (ANFC).
    
    Key improvements over v1:
    1. No threshold — uses mutual check with large search window as the only filter
       (the mutual check itself IS the adaptive mechanism)
    2. Blending factor alpha — preserves individual discriminability for Rank-1
       f_new = f_original + alpha * sum(mutual_neighbors)
       alpha < 1.0 means gentler aggregation, preserving the original feature's discriminative power
    3. Equal weighting for mutual neighbors (more robust than similarity weighting for CC-ReID)
    
    Args:
        feat: [N, D] tensor of L2-normalized features
        max_k: search window size — larger finds more true neighbors but is slower
               (NOT a selection threshold — mutual check handles filtering)
        alpha: blending factor for neighbor aggregation (0.0 = no NFC, 1.0 = full NFC)
    
    Returns:
        [N, D] tensor of centralized features (NOT re-normalized — caller should normalize)
    """
    feat = feat.clone()
    N = feat.size(0)
    
    # Step 1: Cosine similarity (features are L2-normalized)
    sim = torch.mm(feat, feat.t())  # [N, N]
    sim.fill_diagonal_(-1.0)  # exclude self
    
    # Step 2: Find top-max_k neighbors by similarity
    topk_sim, topk_idx = sim.topk(max_k, dim=1, largest=True)  # [N, max_k]
    
    # Step 3: Build index sets for fast mutual check
    topk_sets = []
    for i in range(N):
        topk_sets.append(set(topk_idx[i].tolist()))
    
    # Step 4: Mutual check + equal-weight aggregation with blending
    feat_copy = feat.clone()
    total_neighbors = 0
    neighbor_counts = []
    
    for i in range(N):
        mutual_neighbors = []
        for rank_idx in range(max_k):
            j = topk_idx[i, rank_idx].item()
            # Mutual check: is i also in j's top-max_k?
            if i in topk_sets[j]:
                mutual_neighbors.append(j)
        
        # Equal-weight aggregation with blending factor
        if mutual_neighbors:
            neighbor_sum = feat_copy[mutual_neighbors].sum(dim=0)
            feat[i] += alpha * neighbor_sum
            total_neighbors += len(mutual_neighbors)
        neighbor_counts.append(len(mutual_neighbors))
    
    avg_neighbors = total_neighbors / N if N > 0 else 0
    min_n = min(neighbor_counts) if neighbor_counts else 0
    max_n = max(neighbor_counts) if neighbor_counts else 0
    print(f'   ANFC stats: avg {avg_neighbors:.1f} neighbors/sample (min={min_n}, max={max_n}), alpha={alpha}')
    
    return feat
