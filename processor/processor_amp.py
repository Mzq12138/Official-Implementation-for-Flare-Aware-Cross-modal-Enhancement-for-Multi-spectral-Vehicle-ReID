import logging,pdb
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval,R1_mAP
from torch.cuda import amp
import torch.distributed as dist
import numpy
import torchvision.transforms as transforms
from PIL import Image
from loss import multiModalMarginLossNew
from loss import cdc
from loss import hetero_loss


from torch.nn.functional import kl_div

def KL_loss (a, b):
    
    return min(kl_div(a.softmax(dim=-1).log(), b.softmax(dim=-1)), kl_div(b.softmax(dim=-1).log(), a.softmax(dim=-1)))


def MC_loss (scoreR, scoreN,scoreT):
    
    return KL_loss(scoreR,scoreN) + KL_loss(scoreT,scoreN)

import torch.nn.functional as F


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.image_train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
       
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter1 = AverageMeter()
    acc_meter2 = AverageMeter()
    acc_meter3 = AverageMeter()
    
    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    #evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # image_train
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        acc_meter1.reset()
        acc_meter2.reset()
        acc_meter3.reset()
      
        
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img1,img2,img3, vid, target_cam,_, flare_label) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
           
           
            img1 = img1.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            flare_label = flare_label.to(device)

            with amp.autocast(enabled=True):
                
                mode1,mode2,mode3, RFeat, scoreR_forlabel, NFeat,scoreN_forlabel = model(img1,img2,img3, target, flare_label = flare_label)
                loss1 = loss_fn(mode1[0], mode1[1], target, target_cam)
                loss2 = loss_fn(mode2[0], mode2[1], target, target_cam)
                loss3 = loss_fn(mode3[0], mode3[1], target, target_cam)

                mcloss = 0
                kl_loss = 0
               
                loss =  loss1 + loss2 + loss3
                
                if model.use_mfmp:
                    #MFMP loss
                    kl_loss =  0.8 * KL_loss(scoreR_forlabel, scoreN_forlabel)
                    loss += kl_loss
                if model.use_mcloss:
                    mcloss = MC_loss(mode1[0][0],mode2[0][0],mode3[0][0])
                    loss += mcloss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(mode1[0][0], list):
                acc = (mode1[0][0].max(1)[1] == target).float().mean()
                acc1 = (mode2[0][0].max(1)[1] == target).float().mean()
                acc2 = (mode3[0][0].max(1)[1] == target).float().mean()

            else:
                acc = (mode1[0][0].max(1)[1] == target).float().mean()
                acc1 = (mode2[0][0].max(1)[1] == target).float().mean()
                acc2 = (mode3[0][0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img1.shape[0])
            acc_meter.update(acc, 1)
            acc_meter1.update(acc1, 1)
            acc_meter2.update(acc2, 1)
             
            torch.cuda.synchronize()
            
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f},Acc1: {:.3f},Acc2: {:.3f}, Base Lr: {:.2e},biloss:{:.3f},icloss:{:.3f}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, total_loss:{:.3f}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, acc_meter1.avg, acc_meter2.avg, scheduler._get_lr(epoch)[0],kl_loss,mcloss,loss1,loss2,loss3,loss))
            

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img1,img2,img3, vid, camid, camids,viewids, img_paths,flare_label ) in enumerate(val_loader):
                        with torch.no_grad():
                            img1 = img1.to(device)
                            img2 = img2.to(device)
                            img3 = img3.to(device)
                            camids = camids.to(device)
                            flare_label = flare_label.to(device)
                            feat = model(img1,img2,img3, target, flare_label=flare_label)
                            if cfg.DATASETS.NAMES == "MSVR310":
                                evaluator.update((feat, vid, camid, viewids, img_paths))
                            else:
                                evaluator.update((feat, vid, camid,img_paths)) 
                            # evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    if mAP >= best_index['mAP']:
                        best_index['mAP'] = mAP
                        best_index['Rank-1'] = cmc[0]
                        best_index['Rank-5'] = cmc[4]
                        best_index['Rank-10'] = cmc[9]
                        torch.save(model.state_dict(),
                                os.path.join(cfg.SAVE_DIR, cfg.MODEL.NAME + 'best.pth'))
                    logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
                    logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
                    logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
                    logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img1,img2,img3, vid, camid, camids,viewids, img_paths,flare_label ) in enumerate(val_loader):
                    with torch.no_grad():
                        

                        img1 = img1.to(device)
                        img2 = img2.to(device)
                        img3 = img3.to(device)
                        camids = camids.to(device)
                        flare_label = flare_label.to(device)
                        feat = model(img1,img2,img3, target, flare_label=flare_label)
                        if cfg.DATASETS.NAMES == "MSVR310":
                            evaluator.update((feat, vid, camid, viewids, img_paths))
                        else:
                            evaluator.update((feat, vid, camid,img_paths)) 
                        # evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    torch.save(model.state_dict(),
                               os.path.join(cfg.SAVE_DIR, cfg.MODEL.NAME + 'best.pth'))
                logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
                logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
                logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
                logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,cfg=cfg)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    
    for n_iter, (img1,img2,img3, vid, camid, camids,viewids, img_paths,flare_label ) in enumerate(val_loader):
        with torch.no_grad():
            img1 = img1.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            target = vid.to(device)
            camids = camids.to(device)
            flare_label = flare_label.to(device)
            feat = model(img1,img2,img3,target, flare_label=flare_label)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, vid, camid, viewids, _))
            else:
                evaluator.update((feat, vid, camid,img_paths)) 

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

