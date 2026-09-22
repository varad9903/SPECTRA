import torch


def pairwise_distance(x, y):
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def NFC(feat, k1=2, k2=2):
    feat = feat.clone()
    
    dist = pairwise_distance(feat.to('cuda'), feat.to('cuda')).to('cpu')
    
    eye = torch.eye(dist.size(0)).to(dist.device)
    dist[eye == 1] = 1000
    
    val, rank = dist.topk(k1, largest=False)
    
    mutual_topk_list = []
    for i in range(rank.size(0)):
        mutual_list = []
        for j in rank[i]:
            if i in rank[j][:k2]:
                mutual_list.append(j.item())
        mutual_topk_list.append(mutual_list)
    
    feat_copy = feat.clone()
    for i in range(rank.size(0)):
        if mutual_topk_list[i]:
            feat[i] += feat_copy[mutual_topk_list[i]].sum(dim=0)
    
    return feat


def AdaptiveNFC(feat, max_k=20, alpha=0.7):
    feat = feat.clone()
    N = feat.size(0)
    
    sim = torch.mm(feat, feat.t())
    sim.fill_diagonal_(-1.0)
    
    topk_sim, topk_idx = sim.topk(max_k, dim=1, largest=True)
    
    topk_sets = []
    for i in range(N):
        topk_sets.append(set(topk_idx[i].tolist()))
    
    feat_copy = feat.clone()
    total_neighbors = 0
    neighbor_counts = []
    
    for i in range(N):
        mutual_neighbors = []
        for rank_idx in range(max_k):
            j = topk_idx[i, rank_idx].item()
            if i in topk_sets[j]:
                mutual_neighbors.append(j)
        
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


def _admitted_mask(orig_norm, gen_norm, valid_mask, sim_threshold):
    cos_sim = torch.einsum('nd,nkd->nk', orig_norm, gen_norm)
    return (cos_sim > sim_threshold) & valid_mask.bool()


def _neighbour_index(feats, top_k, set_ids=None, chunk=1024):
    N = feats.size(0)
    k = min(top_k, max(N - 1, 1))
    idx_out = torch.empty(N, k, dtype=torch.long, device=feats.device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sim = feats[s:e] @ feats.t()
        rows = torch.arange(s, e, device=feats.device)
        sim[torch.arange(e - s, device=feats.device), rows] = -2.0
        if set_ids is not None:
            sim[set_ids[s:e].unsqueeze(1) != set_ids.unsqueeze(0)] = -2.0
        idx_out[s:e] = sim.topk(k, dim=1).indices
    return idx_out


def neighbour_calibrated_enhancement(orig_feats, gen_feats, valid_mask, alpha=0.7,
                                     sim_threshold=0.5, top_k=14, set_ids=None,
                                     verbose=True):

    orig_norm = torch.nn.functional.normalize(orig_feats.float(), dim=1)
    gen_norm = torch.nn.functional.normalize(gen_feats.float(), dim=-1)
    if alpha == 0.0:
        return orig_norm

    q_mask = _admitted_mask(orig_norm, gen_norm, valid_mask, sim_threshold)
    n_adm = q_mask.float().sum(dim=1, keepdim=True)
    gen_img_mean = (gen_norm * q_mask.unsqueeze(-1).float()).sum(dim=1) / n_adm.clamp(min=1)
    has_gen = (n_adm.squeeze(1) > 0)

    if set_ids is not None and not torch.is_tensor(set_ids):
        uniq = {v: i for i, v in enumerate(sorted(set(set_ids)))}
        set_ids = torch.tensor([uniq[v] for v in set_ids], device=orig_norm.device)

    nbr = _neighbour_index(orig_norm, top_k, set_ids)
    N = orig_norm.size(0)
    self_idx = torch.arange(N, device=orig_norm.device).unsqueeze(1)
    group = torch.cat([self_idx, nbr], dim=1)

    w = has_gen[group].float()
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=1)
    orig_group_mean = (orig_norm[group] * w.unsqueeze(-1)).sum(dim=1) / w_sum
    gen_group_mean = (gen_img_mean[group] * w.unsqueeze(-1)).sum(dim=1) / w_sum

    offset = orig_group_mean - gen_group_mean
    gen_cal = torch.nn.functional.normalize(gen_norm + offset.unsqueeze(1), dim=-1)
    agg = (gen_cal * q_mask.unsqueeze(-1).float()).sum(dim=1) / n_adm.clamp(min=1)

    if verbose:
        print(f'   neighbour calibration: k={nbr.size(1)}, alpha={alpha}, '
              f'tau={sim_threshold}, admitted {n_adm.mean().item():.1f}/'
              f'{gen_norm.size(1)} views per image, '
              f'|offset| mean {offset.norm(dim=1).mean().item():.4f}')

    return torch.nn.functional.normalize(orig_norm + alpha * agg, dim=1)


def mean_enhancement(orig_feats, gen_feats, valid_mask, alpha=0.7, sim_threshold=None):

    orig_norm = torch.nn.functional.normalize(orig_feats.float(), dim=1)
    gen_norm = torch.nn.functional.normalize(gen_feats.float(), dim=-1)
    if alpha == 0.0:
        return orig_norm
    mask = valid_mask.bool()
    if sim_threshold is not None:
        mask = _admitted_mask(orig_norm, gen_norm, mask, sim_threshold)
    n_adm = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
    agg = (gen_norm * mask.unsqueeze(-1).float()).sum(dim=1) / n_adm
    return torch.nn.functional.normalize(orig_norm + alpha * agg, dim=1)
