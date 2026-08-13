# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss,euclidean_dist
from .center_loss import CenterLoss
from .arcface import CrossEntropy
from .prototype_loss import PrototypeAnchoredContrastiveLoss, CrossInstanceConsistencyLoss


# ==============================================================================
# --- Feature Memory Bank ---
# ==============================================================================

class FeatureMemoryBank:
    """FIFO queue storing text features from past batches for richer contrastive learning.
    
    Instead of contrasting only within the current batch (e.g., 32-64 samples),
    the memory bank provides thousands of additional negatives from prior iterations.
    """
    def __init__(self, size=4096, dim=768, device='cuda'):
        self.size = size
        self.features = torch.zeros(size, dim, device=device)
        self.labels = torch.full((size,), -1, dtype=torch.long, device=device)
        self.ptr = 0
        self.full = False
    
    @torch.no_grad()
    def enqueue(self, features, labels):
        """Add current batch features to the queue (FIFO)."""
        features = features.detach()
        labels = labels.detach()
        bs = features.shape[0]
        
        if bs >= self.size:
            # Batch larger than queue: just fill the entire queue
            self.features[:] = features[:self.size]
            self.labels[:] = labels[:self.size]
            self.ptr = 0
            self.full = True
            return
        
        end = self.ptr + bs
        if end <= self.size:
            self.features[self.ptr:end] = features
            self.labels[self.ptr:end] = labels
            self.ptr = end % self.size
        else:
            # Wrap around
            overflow = end - self.size
            self.features[self.ptr:] = features[:bs - overflow]
            self.labels[self.ptr:] = labels[:bs - overflow]
            self.features[:overflow] = features[bs - overflow:]
            self.labels[:overflow] = labels[bs - overflow:]
            self.ptr = overflow
        
        if end >= self.size:
            self.full = True
    
    def get_valid(self):
        """Return all valid (non-empty) features and labels."""
        if self.full:
            return self.features, self.labels
        elif self.ptr > 0:
            return self.features[:self.ptr], self.labels[:self.ptr]
        else:
            return None, None
    
    def is_ready(self, min_size=64):
        """Check if the bank has enough features to be useful."""
        current = self.size if self.full else self.ptr
        return current >= min_size


# ==============================================================================
# --- Loss Functions ---
# ==============================================================================

def clip_contrastive_loss(image_features, text_features, logit_scale=1.0):
    logits_per_image = logit_scale*image_features  @ text_features.T
    labels = torch.arange(image_features.shape[0],device=image_features.device)
    loss= F.cross_entropy(logits_per_image, labels)
    
    acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
    return loss,acc

def clip_contrastive_score_loss(score, logit_scale=torch.zeros(1)):
    logit_scale=logit_scale.exp().to(score.device)
    logits_per_image = logit_scale*score
    labels = torch.arange(score.shape[0],device=score.device)
    loss= F.cross_entropy(logits_per_image, labels)
    
    acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
    return loss,acc

def clip_sigmoid_loss(image_features, text_features, logit_scale=1.0,logit_bias=0.0):
    logit_scale=logit_scale.type(image_features.type())
    logit_bias=logit_bias.type(image_features.type())
    logits_per_image =logit_scale*image_features  @ text_features.T+logit_bias
    labels = 2*torch.eye(image_features.shape[0],device=image_features.device)-1
    #binary cross entropy
    loss = F.binary_cross_entropy_with_logits(logits_per_image,labels)
    acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
    return loss,acc
    
def clip_l2_loss(score, logit_scale=torch.zeros(1)):
    #diagnal scores
    positive_score=score.diag()
    loss=(1-positive_score).sum()/score.shape[0]
   
    labels = torch.arange(score.shape[0],device=score.device)
    acc = (score.argmax(-1) == labels).sum() / score.shape[0]
    return loss,acc


def symmetric_contrastive_loss(image_features, text_features, logit_scale):
    """Symmetric ITC loss: image→text + text→image contrastive.
    
    Standard practice in CLIP, ALBEF, DRDnet, etc. Both directions
    must learn to match, improving alignment quality.
    """
    N = image_features.shape[0]
    labels = torch.arange(N, device=image_features.device)
    
    logits_i2t = logit_scale * image_features @ text_features.T
    logits_t2i = logit_scale * text_features @ image_features.T
    
    loss_i2t = F.cross_entropy(logits_i2t, labels)
    loss_t2i = F.cross_entropy(logits_t2i, labels)
    
    loss = (loss_i2t + loss_t2i) / 2
    acc = (logits_i2t.argmax(-1) == labels).sum() / N
    return loss, acc


def contrastive_loss_with_memory(image_features, text_features, 
                                  mem_features, mem_labels, labels, logit_scale):
    """Contrastive loss extended with memory bank negatives.
    
    For each image, computes similarity against:
    - Current batch text features (positives on diagonal)
    - Memory bank text features (mostly negatives, some may be same-identity)
    
    This effectively gives a much larger pool of negatives without
    increasing batch size or GPU memory.
    """
    B = image_features.shape[0]
    
    # Combine current text with memory: [B + M, D]
    all_text = torch.cat([text_features, mem_features.type(text_features.dtype)], dim=0)
    
    # Compute similarity: [B, B + M]
    logits = logit_scale * image_features @ all_text.T
    
    # Target: each image's positive is at its batch index (diagonal)
    target = torch.arange(B, device=logits.device)
    
    loss = F.cross_entropy(logits, target)
    acc = (logits[:, :B].argmax(-1) == target).sum() / B
    return loss, acc


def orthogonality_loss(bio_features, nonbio_features):
    """Penalize correlation between bio and non-bio projected features.
    
    Forces the bio projection and non-bio GRL projections to capture
    orthogonal (independent) information, improving disentanglement.
    Uses cosine similarity: ideally bio ⊥ nonbio → cos(bio, nonbio) = 0.
    """
    bio_norm = F.normalize(bio_features, dim=-1)
    nonbio_norm = F.normalize(nonbio_features, dim=-1)
    # Mean squared cosine similarity (should be minimized to 0)
    correlation = (bio_norm * nonbio_norm).sum(dim=-1).pow(2).mean()
    return correlation


def make_loss(cfg, num_classes, lossTypes=None):    # modified by gu
    if lossTypes is None:
        lossTypes=cfg.MODEL.LOSS_TYPE

    sampler = cfg.DATA.SAMPLER
    ortho_weight = cfg.MODEL.ORTHO_WEIGHT if hasattr(cfg.MODEL, 'ORTHO_WEIGHT') else 0.1
    grl_weight = cfg.MODEL.GRL_LOSS_WEIGHT if hasattr(cfg.MODEL, 'GRL_LOSS_WEIGHT') else 1.0
   
    feat_dim = 1024
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    
    # PACR (Prototype-Anchored Contrastive Refinement) loss
    pacr_criterion = None
    cic_criterion = None
    if 'pacr' in lossTypes:
        pacr_momentum = cfg.MODEL.PACR_MOMENTUM if hasattr(cfg.MODEL, 'PACR_MOMENTUM') else 0.999
        pacr_temp = cfg.MODEL.PACR_TEMPERATURE if hasattr(cfg.MODEL, 'PACR_TEMPERATURE') else 0.07
        pacr_hard_neg = cfg.MODEL.PACR_HARD_NEGATIVES if hasattr(cfg.MODEL, 'PACR_HARD_NEGATIVES') else 32
        pacr_warmup = cfg.MODEL.PACR_WARMUP_EPOCHS if hasattr(cfg.MODEL, 'PACR_WARMUP_EPOCHS') else 5
        pacr_criterion = PrototypeAnchoredContrastiveLoss(
            num_classes=num_classes, feat_dim=feat_dim,
            momentum=pacr_momentum, temperature=pacr_temp,
            hard_negatives=pacr_hard_neg, warmup_epochs=pacr_warmup
        ).cuda()
        print(f"PACR loss enabled: momentum={pacr_momentum}, temp={pacr_temp}, "
              f"hard_neg={pacr_hard_neg}, warmup={pacr_warmup}")
    
    cic_weight = cfg.MODEL.CIC_LOSS_WEIGHT if hasattr(cfg.MODEL, 'CIC_LOSS_WEIGHT') else 0.0
    if cic_weight > 0:
        cic_target = cfg.MODEL.CIC_TARGET_SIM if hasattr(cfg.MODEL, 'CIC_TARGET_SIM') else 0.85
        cic_criterion = CrossInstanceConsistencyLoss(target_similarity=cic_target).cuda()
        print(f"CIC loss enabled: weight={cic_weight}, target_sim={cic_target}")
    if 'triplet' in cfg.DATA.SAMPLER:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(lossTypes))
   
    
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)
    else:
        xent=torch.nn.CrossEntropyLoss()
        
        
    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATA.SAMPLER == 'softmax_triplet':
        # Compute steps_per_epoch for PACR warmup (derive epoch from step)
        # This will be set properly once we know the train_loader length
        steps_per_epoch_est = 551  # PRCC: ~17896 / 32 ≈ 559, but actual may differ
        
        def loss_func(score, feat, target, target_cam, caption_feature,
                      clothes_ids, train_writer, step, memory_bank=None, teacher_features=None):
            loss=torch.tensor(0.0,device=feat[0].device)
            if 'ce' in lossTypes:
                if isinstance(score, list):
                    ID_LOSS = [xent(scor, target) for scor in score[0:]]
                    ID_LOSS = sum(ID_LOSS)
                    id_acc = (score[0].argmax(-1) == target).sum() / len(target)
                elif isinstance(score, dict):
                    if isinstance(score['cls_score'], list):
                        ID_LOSS = [xent(s, target) for s in score['cls_score']]
                       
                        for i in range(len(score['cls_score'])):
                            train_writer.add_scalar('loss/id_'+str(i), ID_LOSS[i].item(), step)
                            id_acc = (score['cls_score'][i].argmax(-1) == target).sum() / len(target)
                            train_writer.add_scalar('acc/id_'+str(i), id_acc.item(), step)
                        ID_LOSS = sum(ID_LOSS)
                        id_acc = (score['cls_score'][1].argmax(-1) == target).sum() / len(target)
                    else:
                        ID_LOSS = xent(score['cls_score'], target)
                        id_acc = (score['cls_score'].argmax(-1) == target).sum() / len(target)
                else:
                    ID_LOSS = xent(score, target)
                    id_acc = (score.argmax(-1) == target).sum() / len(target)
                train_writer.add_scalar('loss/id', ID_LOSS.item(), step)
                train_writer.add_scalar('acc/id', id_acc.item(), step)
                loss+=cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS 
                
            if 'triplet' in lossTypes:
                if isinstance(feat, list):
                    #TRI_LOSS = [triplet(feats, target)[0] for feats in feat]
                    TRI_LOSS = triplet(feat[0], target)[0]
                    #TRI_LOSS = sum(TRI_LOSS)
                else:
                    TRI_LOSS = triplet(feat, target)[0]
                train_writer.add_scalar('loss/triplet', TRI_LOSS.item(), step)
                loss+=cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                

                
            if 'clipBio' in lossTypes  or 'clipBioReverse' in lossTypes:
                batch_size=feat[0].shape[0]
                if cfg.MODEL.LAST_LAYER in ['transformer']:
                    image_features_bio,image_features_nonbio=feat
                elif cfg.MODEL.LAST_LAYER in ['fc']:
                    image_features_bio,image_features_nonbio,weight_bio,weight_nonbio=feat
                elif cfg.MODEL.LAST_LAYER in ['clipFc','clipMLP']:
                    image_features_bio=score['clip_bio_score']    
                    if 'clip_nonbio_score' in score:
                        image_features_nonbio=score['clip_nonbio_score']
                    else:
                        image_features_nonbio=None
                # if 'cip_bio_reverse_score' in score:
                #     image_feature_reverse=score['clip_bio_reverse_score']
                #     #

                image_features_bio=torch.nn.functional.normalize(image_features_bio,dim=-1)
                if image_features_nonbio is not None:
                    image_features_nonbio=torch.nn.functional.normalize(image_features_nonbio,dim=-1)
                if image_features_bio.shape[1]>200:
                    image_features_bio=image_features_bio.unsqueeze(1)
                # if caption_feature.shape[1]>200:
                #     caption_feature=caption_feature.unsqueeze(1)
                        
                caption_feature=caption_feature.type(image_features_bio.type())
                text_features_bio=caption_feature[:,:image_features_bio.shape[1]]
                text_features_bio=torch.nn.functional.normalize(text_features_bio,dim=-1)
                text_features_nonbio=caption_feature[:,image_features_bio.shape[1]:]
                text_features_nonbio=torch.nn.functional.normalize(text_features_nonbio,dim=-1)
                #text_features_bio.norm(dim=-1, keepdim=True)
            

            
            if 'clipBio' in lossTypes:
                loss_clip_all=0
                for i in range(image_features_bio.shape[1]):
                    # ---- Symmetric ITC: image→text + text→image ----
                    img_bio_i = image_features_bio[:,i]
                    txt_bio_i = text_features_bio[:,i]
                    
                    scale_val = score['clip_bio_scale']
                    if isinstance(scale_val, torch.Tensor):
                        logit_scale = scale_val.exp().to(img_bio_i.device)
                    else:
                        # Use the original constant value (designed to work with raw cosine sim)
                        logit_scale = torch.tensor(float(scale_val), device=img_bio_i.device)
                    
                    # Check if memory bank is ready for extended contrastive
                    if memory_bank is not None and memory_bank.is_ready():
                        mem_feat, mem_labels = memory_bank.get_valid()
                        mem_feat = mem_feat.to(device=img_bio_i.device, dtype=img_bio_i.dtype)
                        loss_clip, i2t_acc = contrastive_loss_with_memory(
                            img_bio_i, txt_bio_i, mem_feat, mem_labels, target, logit_scale)
                        # Also add text→image direction (symmetric, batch-only)
                        logits_t2i = logit_scale * txt_bio_i @ img_bio_i.T
                        labels_t2i = torch.arange(batch_size, device=img_bio_i.device)
                        loss_t2i = F.cross_entropy(logits_t2i, labels_t2i)
                        loss_clip = (loss_clip + loss_t2i) / 2
                    else:
                        # Standard symmetric contrastive (no memory yet)
                        loss_clip, i2t_acc = symmetric_contrastive_loss(img_bio_i, txt_bio_i, logit_scale)
                    
                    loss_clip_all += loss_clip
                    train_writer.add_scalar('acc/clip_bio_'+str(i), i2t_acc.item(), step)
                    train_writer.add_scalar('loss/clip_bio_'+str(i), loss_clip.item(), step)
               
                loss+=loss_clip_all
                
                # ---- Enqueue current features into memory bank ----
                if memory_bank is not None and memory_bank.size > 0:
                    memory_bank.enqueue(
                        text_features_bio[:,0].detach(),
                        target
                    )
                
            if 'clipBioReverse' in lossTypes :
                image_feature_reverse=score['clip_bio_reverse_score']
                loss_clip_reverse=0
              
                for i in range(image_feature_reverse.shape[1]):
                    image_feature_reverse_i=torch.nn.functional.normalize(image_feature_reverse[:, i])
                    image_feature_reverse_i=image_feature_reverse_i.type(text_features_nonbio.type())
                    scale_val_nonbio = score['clip_nonbio_scale'][i]
                    if isinstance(scale_val_nonbio, torch.Tensor):
                        logit_scale_nonbio = scale_val_nonbio.exp().to(image_feature_reverse_i.device)
                    else:
                        # Use the original constant value
                        logit_scale_nonbio = torch.tensor(float(scale_val_nonbio), device=image_feature_reverse_i.device)
                        
                    logits_per_bio_reverse = logit_scale_nonbio * image_feature_reverse_i @ text_features_nonbio[:,i].T
                    # contrastive loss over full batch
                    labels = torch.arange(batch_size, device=image_feature_reverse_i.device)
                    loss_image = F.cross_entropy(logits_per_bio_reverse, labels)
                    
                    i2t_acc = (logits_per_bio_reverse.argmax(-1) == labels).sum() / len(logits_per_bio_reverse)
                    train_writer.add_scalar('acc/clip_bio_reverse_'+str(i), i2t_acc.item(), step)
                    #print(loss_image)
                    loss_clip_reverse += loss_image 
                    train_writer.add_scalar('loss/clip_bio_reverse_'+str(i), loss_clip_reverse.item(), step)      
                loss+=grl_weight * loss_clip_reverse

            # ---- Orthogonality loss: bio ⊥ non-bio ----
            if 'clipBioReverse' in lossTypes and ortho_weight > 0:
                if 'bio_projected' in score:
                    bio_proj = score['bio_projected']  # [B, D]
                    nonbio_proj = score['clip_bio_reverse_score']  # [B, nonBio_num, D]
                    # Average across non-bio heads for single orthogonality signal
                    nonbio_avg = nonbio_proj.mean(dim=1)  # [B, D]
                    loss_ortho = orthogonality_loss(bio_proj, nonbio_avg)
                    loss += ortho_weight * loss_ortho
                    train_writer.add_scalar('loss/orthogonality', loss_ortho.item(), step)

            # ---- Center Loss: pull features toward learnable identity centroids ----
            if 'center' in lossTypes:
                # feat[0] is feat_bio: [B, 1024]
                feat_bio = feat[0] if isinstance(feat, list) else feat
                center_loss_val = center_criterion(feat_bio, target)
                loss += cfg.SOLVER.CENTER_LOSS_WEIGHT * center_loss_val
                train_writer.add_scalar('loss/center', center_loss_val.item(), step)

            # ---- PACR: Prototype-Anchored Contrastive Refinement ----
            if 'pacr' in lossTypes and pacr_criterion is not None:
                feat_bio = feat[0] if isinstance(feat, list) else feat
                # Derive current epoch from step
                current_epoch = step // steps_per_epoch_est
                pacr_weight = cfg.MODEL.PACR_LOSS_WEIGHT if hasattr(cfg.MODEL, 'PACR_LOSS_WEIGHT') else 0.5
                pacr_loss_val = pacr_criterion(feat_bio, target, current_epoch=current_epoch)
                loss += pacr_weight * pacr_loss_val
                if pacr_loss_val.item() > 0:
                    train_writer.add_scalar('loss/pacr', pacr_loss_val.item(), step)
            
            # ---- CIC: Cross-Instance Consistency ----
            if cic_criterion is not None and cic_weight > 0:
                feat_bio = feat[0] if isinstance(feat, list) else feat
                cic_loss_val = cic_criterion(feat_bio, target)
                loss += cic_weight * cic_loss_val
                train_writer.add_scalar('loss/cic', cic_loss_val.item(), step)

            # ---- Self-Distillation: match teacher (ANFC-refined) features ----
            if 'distill' in lossTypes and teacher_features is not None:
                feat_student = feat[0] if isinstance(feat, list) else feat
                cos_sim = F.cosine_similarity(feat_student, teacher_features, dim=1)
                DISTILL_LOSS = (1.0 - cos_sim).mean()
                distill_weight = cfg.MODEL.DISTILL_LOSS_WEIGHT if hasattr(cfg.MODEL, 'DISTILL_LOSS_WEIGHT') else 1.0
                loss += distill_weight * DISTILL_LOSS
                train_writer.add_scalar('loss/distill', DISTILL_LOSS.item(), step)

            return loss

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
            'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


