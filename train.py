from utils.logger import setup_logger
from datasets import make_dataloader

from model import FACENet
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train_amp
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import pdb, datetime

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="/data/sunyongqi/codes/Facenetgai/TransReID-main/configs/MSV863/vit_transreid_stride_mmbaseline.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    # parser.add_argument("--gpu", default='0', type=int)
    parser.add_argument("--gpu", default='0', type=int)
    parser.add_argument("--IC_param", default='0.8', type=float)
    args = parser.parse_args()


    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    print(args.IC_param)
    now=datetime.datetime.now()
    strtime = now.strftime('%Y_%m_%d_%H_%M_%S')
    # pdb.set_trace()
    output_dir = cfg.OUTPUT_DIR + strtime
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cfg.SAVE_DIR = output_dir
    cfg.IC_param = args.IC_param
    cfg.freeze()

    
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
    
    

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://',rank=0,world_size=1)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num = make_dataloader(cfg)

    if cfg.MODEL.FCE or cfg.MODEL.ICLOSS:
        print('Using facenet based on baseline_mm model with vit cross_att...')
        model = FACENet(cfg, num_class=num_classes, camera_num=camera_num, view_num = 0)
        
    else:
        print('Using baseline_mm model...')
        model = FACENet(cfg, num_class=num_classes, camera_num=camera_num, view_num = 0)
        

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)
    # pdb.set_trace()
    do_train_amp(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )
    
    

