import argparse
import os
import datetime
import logging
import time
import numpy as np
from collections import OrderedDict

import torch
import torch.utils
import torch.distributed
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.backends

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.utils.loss import PixelCLLoss

import warnings

warnings.filterwarnings('ignore')


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
        return stripped_state_dict


def train(cfg, local_rank, distributed):
    logger = logging.getLogger("ICCV2021.trainer")

    # create network
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)

    if local_rank == 0:
        print(classifier)

    # batch size: half for source and half for target
    batch_size = cfg.SOLVER.BATCH_SIZE // 2
    if distributed:
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size()) // 2
        if not cfg.MODEL.FREEZE_BN:
            feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg2
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()

    # init optimizer
    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()

    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    # load checkpoint
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        feature_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(
            checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(feature_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(
            checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    # init data loader
    src_train_data = build_dataset(cfg, mode='train', is_source=True)
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False)
    if distributed:
        src_train_sampler = torch.utils.data.distributed.DistributedSampler(src_train_data)
        tgt_train_sampler = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
    else:
        src_train_sampler = None
        tgt_train_sampler = None
    src_train_loader = torch.utils.data.DataLoader(src_train_data, batch_size=batch_size,
                                                   shuffle=(src_train_sampler is None), num_workers=4,
                                                   pin_memory=True, sampler=src_train_sampler, drop_last=True)
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size,
                                                   shuffle=(tgt_train_sampler is None), num_workers=4,
                                                   pin_memory=True, sampler=tgt_train_sampler, drop_last=True)

    iteration = 0
    start_training_time = time.time()
    end = time.time()
    output_dir = cfg.OUTPUT_DIR
    save_to_disk = local_rank == 0
    max_iters = cfg.SOLVER.MAX_ITER
    meters = MetricLogger(delimiter="  ")

    # init loss
    seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
    cl_criterion = PixelCLLoss(cfg)

    # init prototype
    logger.info("Loading semantic prototype from {}.".format(cfg.CENTROID_DIR))
    semantic_prototype = torch.load(os.path.join(cfg.CENTROID_DIR, 'semantic_prototype.pth'))
    logger.info(">>>>>>>>>>>>>>>> Loaded source centroids >>>>>>>>>>>>>>>>")

    feature_extractor.train()
    classifier.train()

    logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
    for i, ((src_input, src_label, src_name), (tgt_input, tgt_pseudo_label, _)) in enumerate(
            zip(src_train_loader, tgt_train_loader)):
        data_time = time.time() - end

        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters,
                                          power=cfg.SOLVER.LR_POWER)
        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr * 10

        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()
        src_input = src_input.cuda(non_blocking=True)
        src_label = src_label.cuda(non_blocking=True).long()
        tgt_input = tgt_input.cuda(non_blocking=True)
        tgt_pseudo_label = tgt_pseudo_label.cuda(non_blocking=True).long()
        semantic_prototype = semantic_prototype.cuda(non_blocking=True)

        src_size = src_input.shape[-2:]

        src_feat = feature_extractor(src_input)
        src_seg = classifier(src_feat)  # forward ``size`` is None
        src_pred = F.interpolate(src_seg, size=src_size, mode='bilinear', align_corners=True)
        loss_seg = seg_criterion(src_pred.div(1.8), src_label)
        loss_seg.backward(retain_graph=True)

        tgt_feat = feature_extractor(tgt_input)
        tgt_seg = classifier(tgt_feat)  # forward ``size`` is None
        tgt_mask = F.interpolate(tgt_pseudo_label.unsqueeze(0).float(), (tgt_seg.size(2), tgt_seg.size(3)),
                                 mode='nearest').squeeze(0).long()

        # get source mask
        src_seg_softmax = F.softmax(src_seg.detach().clone(), dim=1)
        _, src_seg_label = torch.max(src_seg_softmax, dim=1)
        src_mask = F.interpolate(src_label.unsqueeze(0).float(), (src_seg_label.size(1), src_seg_label.size(2)),
                                 mode='nearest').squeeze(0).long()
        src_mask[src_mask != src_seg_label] = 255
        N, C1, H1, W1 = src_feat.size()
        assert C1 == 2048 or C1 == 1024
        # normalization before compute mean
        src_feat_detach = F.normalize(src_feat.detach().clone(), p=2, dim=1)
        src_feat_detach = src_feat_detach.transpose(1, 2).transpose(2, 3).contiguous()
        # update semantic prototypes
        current_prototype = torch.zeros_like(semantic_prototype).cuda(non_blocking=True)
        assert current_prototype.size() == semantic_prototype.size()
        for k in src_mask.unique():
            if k != 255:
                src_k_mask = (src_mask == k)
                src_k_feat = src_feat_detach[src_k_mask.view(N, H1, W1, 1).repeat(1, 1, 1, C1)].view(-1, C1)
                current_prototype[:, [k]] = torch.mean(src_k_feat, dim=0).view(-1, 1)
        semantic_prototype = cfg.SOLVER.ALPHA * semantic_prototype + (1 - cfg.SOLVER.ALPHA) * current_prototype

        # contrastive loss
        loss_cl = cfg.SOLVER.CL_LOSS * cl_criterion(semantic_prototype=semantic_prototype,
                                                    src_feat=src_feat,
                                                    src_mask=src_mask,
                                                    tgt_feat=tgt_feat,
                                                    tgt_mask=tgt_mask)
        loss_cl.backward()

        optimizer_fea.step()
        optimizer_cls.step()

        meters.update(loss_seg=loss_seg.item())
        meters.update(loss_cl=loss_cl.item())

        iteration = iteration + 1

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.2f}"
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer_fea.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                )
            )

        if (iteration == cfg.SOLVER.MAX_ITER or iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0) and save_to_disk:
            filename = os.path.join(output_dir, "model_iter{:06d}.pth".format(iteration))
            torch.save({'iteration': iteration,
                        'feature_extractor': feature_extractor.state_dict(),
                        'classifier': classifier.state_dict(),
                        'optimizer_fea': optimizer_fea.state_dict(),
                        'optimizer_cls': optimizer_cls.state_dict(),
                        }, filename)

        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (cfg.SOLVER.MAX_ITER)
        )
    )

    return feature_extractor, classifier


def main():
    parser = argparse.ArgumentParser(description="Pytorch Domain Adaptive Semantic Segmentation Training")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("ICCV2021", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, args.local_rank, args.distributed)


if __name__ == '__main__':
    main()
