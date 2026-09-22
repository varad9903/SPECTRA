import torch
import torch.nn as nn
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss, euclidean_dist
from .center_loss import CenterLoss
from .arcface import CrossEntropy


def clip_contrastive_loss(image_features, text_features, logit_scale=1.0):
    logits_per_image = logit_scale * image_features @ text_features.T
    labels = torch.arange(image_features.shape[0], device=image_features.device)
    loss = F.cross_entropy(logits_per_image, labels)
    acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
    return loss, acc


def clip_contrastive_score_loss(score, logit_scale=torch.zeros(1)):
    logit_scale = logit_scale.exp().to(score.device)
    logits_per_image = logit_scale * score
    labels = torch.arange(score.shape[0], device=score.device)
    loss = F.cross_entropy(logits_per_image, labels)
    acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
    return loss, acc


def clip_l2_loss(score, logit_scale=torch.zeros(1)):
    positive_score = score.diag()
    loss = (1 - positive_score).sum() / score.shape[0]
    labels = torch.arange(score.shape[0], device=score.device)
    acc = (score.argmax(-1) == labels).sum() / score.shape[0]
    return loss, acc


def symmetric_contrastive_loss(image_features, text_features, logit_scale):
    N = image_features.shape[0]
    labels = torch.arange(N, device=image_features.device)
    logits_i2t = logit_scale * image_features @ text_features.T
    logits_t2i = logit_scale * text_features @ image_features.T
    loss = (F.cross_entropy(logits_i2t, labels) + F.cross_entropy(logits_t2i, labels)) / 2
    acc = (logits_i2t.argmax(-1) == labels).sum() / N
    return loss, acc


def make_loss(cfg, num_classes, lossTypes=None):
    if lossTypes is None:
        lossTypes = cfg.MODEL.LOSS_TYPE

    sampler = cfg.DATA.SAMPLER
    grl_weight = cfg.MODEL.GRL_LOSS_WEIGHT if hasattr(cfg.MODEL, 'GRL_LOSS_WEIGHT') else 1.0

    feat_dim = 1024
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)

    if 'triplet' in cfg.DATA.SAMPLER:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
    else:
        print('expected sampler triplet but got {}'.format(lossTypes))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        xent = torch.nn.CrossEntropyLoss()

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATA.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, caption_feature,
                      clothes_ids, train_writer, step, teacher_features=None):
            loss = torch.tensor(0.0, device=feat[0].device)

            if 'ce' in lossTypes:
                if isinstance(score, list):
                    ID_LOSS = sum(xent(scor, target) for scor in score)
                    id_acc = (score[0].argmax(-1) == target).sum() / len(target)
                elif isinstance(score, dict):
                    if isinstance(score['cls_score'], list):
                        ID_LOSS = [xent(s, target) for s in score['cls_score']]
                        for i, il in enumerate(ID_LOSS):
                            train_writer.add_scalar('loss/id_' + str(i), il.item(), step)
                            id_acc = (score['cls_score'][i].argmax(-1) == target).sum() / len(target)
                            train_writer.add_scalar('acc/id_' + str(i), id_acc.item(), step)
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
                loss += cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS

            if 'triplet' in lossTypes:
                TRI_LOSS = triplet(feat[0], target)[0] if isinstance(feat, list) else triplet(feat, target)[0]
                train_writer.add_scalar('loss/triplet', TRI_LOSS.item(), step)
                loss += cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

            if 'clipBio' in lossTypes or 'clipBioReverse' in lossTypes:
                batch_size = feat[0].shape[0]
                if cfg.MODEL.LAST_LAYER in ['clipFc', 'clipMLP']:
                    image_features_bio = score['clip_bio_score']
                    image_features_nonbio = score.get('clip_nonbio_score', None)
                elif cfg.MODEL.LAST_LAYER in ['transformer']:
                    image_features_bio, image_features_nonbio = feat
                else:
                    image_features_bio, image_features_nonbio, _, _ = feat

                image_features_bio = F.normalize(image_features_bio, dim=-1)
                if image_features_nonbio is not None:
                    image_features_nonbio = F.normalize(image_features_nonbio, dim=-1)
                if image_features_bio.shape[1] > 200:
                    image_features_bio = image_features_bio.unsqueeze(1)

                caption_feature = caption_feature.type(image_features_bio.type())
                text_features_bio = F.normalize(caption_feature[:, :image_features_bio.shape[1]], dim=-1)
                text_features_nonbio = F.normalize(caption_feature[:, image_features_bio.shape[1]:], dim=-1)

            if 'clipBio' in lossTypes:
                loss_clip_all = 0
                for i in range(image_features_bio.shape[1]):
                    img_bio_i = image_features_bio[:, i]
                    txt_bio_i = text_features_bio[:, i]
                    scale_val = score['clip_bio_scale']
                    if isinstance(scale_val, torch.Tensor):
                        logit_scale = scale_val.exp().to(img_bio_i.device)
                    else:
                        logit_scale = torch.tensor(float(scale_val), device=img_bio_i.device)

                    loss_clip, i2t_acc = symmetric_contrastive_loss(img_bio_i, txt_bio_i, logit_scale)

                    loss_clip_all += loss_clip
                    train_writer.add_scalar('acc/clip_bio_' + str(i), i2t_acc.item(), step)
                    train_writer.add_scalar('loss/clip_bio_' + str(i), loss_clip.item(), step)
                loss += loss_clip_all

            if 'clipBioReverse' in lossTypes:
                image_feature_reverse = score['clip_bio_reverse_score']
                loss_clip_reverse = 0
                for i in range(image_feature_reverse.shape[1]):
                    rev_i = F.normalize(image_feature_reverse[:, i], dim=-1).type(text_features_nonbio.type())
                    scale_val_nonbio = score['clip_nonbio_scale'][i]
                    if isinstance(scale_val_nonbio, torch.Tensor):
                        logit_scale_nonbio = scale_val_nonbio.exp().to(rev_i.device)
                    else:
                        logit_scale_nonbio = torch.tensor(float(scale_val_nonbio), device=rev_i.device)
                    logits = logit_scale_nonbio * rev_i @ text_features_nonbio[:, i].T
                    labels = torch.arange(batch_size, device=rev_i.device)
                    loss_image = F.cross_entropy(logits, labels)
                    i2t_acc = (logits.argmax(-1) == labels).sum() / len(logits)
                    train_writer.add_scalar('acc/clip_bio_reverse_' + str(i), i2t_acc.item(), step)
                    loss_clip_reverse += loss_image
                    train_writer.add_scalar('loss/clip_bio_reverse_' + str(i), loss_clip_reverse.item(), step)
                loss += grl_weight * loss_clip_reverse

            if 'center' in lossTypes:
                feat_bio = feat[0] if isinstance(feat, list) else feat
                center_loss_val = center_criterion(feat_bio, target)
                loss += cfg.SOLVER.CENTER_LOSS_WEIGHT * center_loss_val
                train_writer.add_scalar('loss/center', center_loss_val.item(), step)

            return loss

    else:
        print('expected sampler softmax_triplet but got {}'.format(cfg.DATALOADER.SAMPLER))

    return loss_func, center_criterion
