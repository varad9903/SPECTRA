import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval,R1_mAP_eval_LTCC
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from loss.make_loss import FeatureMemoryBank
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import gc


def unfreeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = True    

def do_train(cfg,
             model,
             train_loader,
             local_rank,
             dataset,
             val_loader = None,
             val_loader_same = None):
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    
    loss_fn, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)


    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("DIFFER.train")
    logger.info('start training')
    

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    if device:
        model.to(local_rank)
        #loss_fn.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            #
            print('Using {} GPUs for training, current GPU number {}'.format(torch.cuda.device_count(),local_rank))
            # rank = dist.get_rank()
            # device_id = rank % torch.cuda.device_count()
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)
            #loss_fn = torch.nn.parallel.DistributedDataParallel(loss_fn, device_ids=[local_rank])
    train_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'train'))
   
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    nfc_mode = cfg.TEST.NFC_MODE if hasattr(cfg.TEST, 'NFC_MODE') else 'none'
    nfc_k1 = cfg.TEST.NFC_K1 if hasattr(cfg.TEST, 'NFC_K1') else 2
    nfc_k2 = cfg.TEST.NFC_K2 if hasattr(cfg.TEST, 'NFC_K2') else 2
    nfc_alpha = cfg.TEST.NFC_ALPHA if hasattr(cfg.TEST, 'NFC_ALPHA') else 0.7
    nfc_max_k = cfg.TEST.NFC_MAX_K if hasattr(cfg.TEST, 'NFC_MAX_K') else 20
    nfc_apply_to = cfg.TEST.NFC_APPLY_TO if hasattr(cfg.TEST, 'NFC_APPLY_TO') else 'both'
    nfc_kwargs = dict(nfc_mode=nfc_mode, nfc_k1=nfc_k1, nfc_k2=nfc_k2, nfc_alpha=nfc_alpha, nfc_max_k=nfc_max_k, nfc_apply_to=nfc_apply_to)
    logger.info(f'NFC config: mode={nfc_mode}, k1={nfc_k1}, k2={nfc_k2}, alpha={nfc_alpha}, max_k={nfc_max_k}, apply_to={nfc_apply_to}')
    
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, **nfc_kwargs)  # ltcc
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, **nfc_kwargs)

    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, **nfc_kwargs)  # prcc
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, **nfc_kwargs)
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, ignore_cam=True, **nfc_kwargs)


    scaler = amp.GradScaler()
    best_rank1 = -np.inf
    best_epoch = 0
    best_mAP = -np.inf
    best_mAP_epoch = 0
    start_train_time = time.time()

    # --- Feature Memory Bank for extended contrastive learning ---
    clip_dim = cfg.MODEL.CLIP_DIM if hasattr(cfg.MODEL, 'CLIP_DIM') else 768
    mem_size = cfg.MODEL.MEMORY_BANK_SIZE if hasattr(cfg.MODEL, 'MEMORY_BANK_SIZE') else 0
    if mem_size > 0:
        memory_bank = FeatureMemoryBank(size=mem_size, dim=clip_dim, device=device)
        logger.info(f'Memory bank initialized: size={mem_size}, dim={clip_dim}')
    else:
        memory_bank = None
        logger.info('Memory bank disabled (MEMORY_BANK_SIZE=0)')
    
    # --- Self-Distillation: load pre-computed teacher features ---
    teacher_features_dict = None
    teacher_path = cfg.MODEL.TEACHER_FEATURES_PATH if hasattr(cfg.MODEL, 'TEACHER_FEATURES_PATH') else ''
    if teacher_path and os.path.isfile(teacher_path):
        teacher_features_dict = torch.load(teacher_path, map_location='cpu')
        logger.info(f'Self-distillation enabled: loaded {len(teacher_features_dict)} teacher features from {teacher_path}')
    elif teacher_path:
        logger.warning(f'Teacher features path not found: {teacher_path}')
    
    model.eval()
    #evaluator.reset()
    epoch=0
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff.reset()
        evaluator_general.reset()

    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff.reset()
        evaluator_same.reset()
    else:
        evaluator.reset()

    pretrainModelTest=False
    if pretrainModelTest:
        logger.info("Test the model at the beginning of training")
        if cfg.DATA.DATASET == 'prcc':
            logger.info("Clothes changing setting")
            rank1= test(cfg, model, evaluator_diff, val_loader, logger, device,epoch, train_writer)
            logger.info("Standard setting")
            test(cfg, model, evaluator_same, val_loader_same, logger, device, epoch,  train_writer,test=True)
        elif cfg.DATA.DATASET == 'ltcc':
            logger.info("Clothes changing setting")
            rank1 = test(cfg, model, evaluator_diff, val_loader, logger, device,epoch,train_writer, ltcc=True)
            logger.info("Standard setting")
            test(cfg, model, evaluator_general, val_loader, logger, device, epoch, train_writer,test=True)
        else:
            rank1= test(cfg, model, evaluator, val_loader, logger, device,epoch,train_writer)   
    

    for epoch in range(cfg.TRAIN.START_EPOCH, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()


        if cfg.DATA.DATASET == 'ltcc':
            evaluator_diff.reset()
            evaluator_general.reset()

        elif cfg.DATA.DATASET == 'prcc':
            evaluator_diff.reset()
            evaluator_same.reset()
        else:
            evaluator.reset()
        
        scheduler.step(epoch)
        model.train()
      
        for idx, data in enumerate(train_loader):
            step=idx+len(train_loader)*epoch
            
            samples=data['image']
            targets=data['pid']
            camids=data['camid']
            clothes=data['clothes_id']
            caption_feature=data['caption_feature'] if 'caption_feature' in data else None
            
            # Look up teacher features for self-distillation
            teacher_feats = None
            if teacher_features_dict is not None:
                img_paths = data['image_path']
                try:
                    teacher_feats = torch.stack([teacher_features_dict[p] for p in img_paths])
                    teacher_feats = teacher_feats.to(local_rank if torch.cuda.is_available() else 'cpu')
                except KeyError as e:
                    pass  # Skip distillation for this batch if paths don't match
                   
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
           
            with amp.autocast(enabled=True):
              
                score, feat = model(samples, cam_label=camids)
                

            loss = loss_fn(score, feat, targets, camids,caption_feature,clothes_ids=clothes,train_writer=train_writer,step=step,memory_bank=memory_bank,teacher_features=teacher_feats)
            
            train_writer.add_scalar('loss/total', loss.item(), step)

            current_lr=scheduler._get_lr(epoch)[0]
            train_writer.add_scalar('lr',current_lr , step)
            
            scaler.scale(loss).backward()
            

            scaler.step(optimizer)
            scaler.update()
            if 'center' in cfg.MODEL.LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == targets).float().mean()
            elif isinstance(score, dict):
                if isinstance(score['cls_score'], list):
                    acc = (score['cls_score'][0].max(1)[1] == targets).float().mean()
                else:
                    acc = (score['cls_score'].max(1)[1] == targets).float().mean()
            else:
                acc = (score.max(1)[1] == targets).float().mean()

            loss_meter.update(loss.item(), samples.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (idx + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (idx + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, current_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (idx + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % eval_period == 0 :#and cfg.MODEL.DIST_TRAIN and dist.get_rank() == 0:
            model.eval()
            if cfg.DATA.DATASET == 'prcc':
                logger.info("Clothes changing setting")
                rank1, mAP = test(cfg, model, evaluator_diff, val_loader, logger, device,epoch, train_writer)
                logger.info("Standard setting")
                test(cfg, model, evaluator_same, val_loader_same, logger, device, epoch,  train_writer,test=True)
            elif cfg.DATA.DATASET == 'ltcc':
                logger.info("Clothes changing setting")
                rank1, mAP = test(cfg, model, evaluator_diff, val_loader, logger, device,epoch,train_writer, ltcc=True)
                logger.info("Standard setting")
                test(cfg, model, evaluator_general, val_loader, logger, device, epoch, train_writer,test=True)
            else:
                rank1, mAP = test(cfg, model, evaluator, val_loader, logger, device,epoch,train_writer)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch
                logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))
                save_model = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(save_model, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_rank1.pth'))
                logger.info("Saved best Rank-1 model")
            
            if mAP > best_mAP:
                best_mAP = mAP
                best_mAP_epoch = epoch
                logger.info("==> Best mAP {:.1%}, achieved at epoch {}".format(best_mAP, best_mAP_epoch))
                save_model = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(save_model, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_mAP.pth'))
                logger.info("Saved best mAP model")

    if cfg.MODEL.DIST_TRAIN:
        if dist.get_rank() == 0:
            logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    total_time = time.time() - start_train_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def do_inference(cfg,
                 model,
                 dataset,
                 val_loader = None,
                 val_loader_same = None):
    logger = logging.getLogger("DIFFER.test")
    logger.info("Enter inferencing")

    logger.info("transreid inferencing")
    device = "cuda"
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # ltcc
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # prcc
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,ignore_cam=True)
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff.reset()
        evaluator_general.reset()

    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff.reset()
        evaluator_same.reset()
    else:
        evaluator.reset()
    model.to(device)
    model.eval()
    if cfg.DATA.DATASET == 'prcc':
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True,clothChanging=True)
        logger.info("Standard setting")
        test(cfg, model, evaluator_same, val_loader_same, logger, device, test=True)
    elif cfg.DATA.DATASET == 'ltcc':
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True,ltcc=True,clothChanging=True)
        logger.info("Standard setting")
        test(cfg, model, evaluator_general, val_loader, logger, device, test=True)
    else:
        test(cfg, model, evaluator, val_loader, logger, device, test=True,clothChanging=True)

def test(cfg, model, evaluator, val_loader, logger, device, epoch=None, test_writer=None,test=False,ltcc=False,clothChanging=False):
        
    for n_iter, data in enumerate(val_loader):
        if n_iter%100==0:
            logger.info(f"processing batch {n_iter}/{len(val_loader)}")
    
        imgs=data['image']
        pids=data['pid']
        camids=data['camid']
        clothes_id=data['clothes_id']
        img_paths=data['image_path']
        
        with torch.no_grad():
            
            imgs = imgs.to(device) 
         
            camids0=camids.to(device) 
           
            feat = model(imgs, cam_label=camids0)
            
            if ltcc:
                evaluator.update((feat, pids, camids, clothes_id,img_paths))
            else:
                evaluator.update((feat, pids, camids,clothes_id,img_paths))
                
   
    cmc, mAP, _, _, _, queryFeature, galleryFeature,imgpath = evaluator.compute()
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("mAP-:{:.1%}".format(mAP))
        
   
    rank1 = cmc[0] 
    if test :
        torch.cuda.empty_cache()
        return rank1, mAP
   
    
    logger.info("Validation Results - Epoch: {}".format(epoch))
    #rank1 = cmc[0]
    
    #if clothChanging:
    test_writer.add_scalar('rank1', rank1, epoch)
    test_writer.add_scalar('mAP', mAP, epoch)
    torch.cuda.empty_cache()
    return rank1, mAP
