import argparse
import os
import logging
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.distributed
import torch.backends.cudnn

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.utils.misc import mkdir
from core.utils.logger import setup_logger


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def get_source_centroids(cfg):
    logger = logging.getLogger("ICCV2021.trainer")
    logger.info("Start training")

    feature_extractor = build_feature_extractor(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)

    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    feature_extractor.eval()
    classifier.eval()

    torch.cuda.empty_cache()

    src_train_data = build_dataset(cfg, mode='train', is_source=True, epochwise=True)
    src_train_loader = torch.utils.data.DataLoader(src_train_data,
                                                   batch_size=cfg.SOLVER.BATCH_SIZE_VAL,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   pin_memory=True,
                                                   drop_last=False)

    # the k-th column is the representation of k-th semantic prototype
    objective_feat = {k: None for k in range(cfg.MODEL.NUM_CLASSES)}

    with torch.no_grad():
        for batch in tqdm(src_train_loader):
            src_input, src_label, _ = batch
            src_input = src_input.cuda(non_blocking=True)
            src_label = src_label.cuda(non_blocking=True).long()

            src_feat = feature_extractor(src_input)
            src_seg = classifier(src_feat)

            src_seg_softmax = F.softmax(src_seg, dim=1)
            _, src_seg_label = torch.max(src_seg_softmax, dim=1)

            # source mask
            N, C1, H1, W1 = src_feat.size()
            src_mask = F.interpolate(src_label.unsqueeze(0).float(), size=(H1, W1), mode='nearest').squeeze(0).long()
            src_mask[src_mask != src_seg_label] = 255  # combine the predicted label

            # normalization before compute centroid
            src_feat = F.normalize(src_feat, p=2, dim=1)
            src_feat = src_feat.transpose(1, 2).transpose(2, 3).contiguous()
            for k in range(cfg.MODEL.NUM_CLASSES):
                if k in src_mask:
                    src_k_mask = (src_mask == k)
                    if src_k_mask.sum() > 0:
                        src_k_feat = src_feat[src_k_mask.view(N, H1, W1, 1).repeat(1, 1, 1, C1)].view(-1, C1)
                        src_k_feat_centroids = torch.mean(src_k_feat, dim=0).view(-1, 1)
                        if objective_feat[k] is None:
                            objective_feat[k] = src_k_feat_centroids
                        else:
                            objective_feat[k] = torch.cat((objective_feat[k], src_k_feat_centroids), dim=1)

    semantic_prototype = torch.zeros((C1, cfg.MODEL.NUM_CLASSES))
    for k in range(cfg.MODEL.NUM_CLASSES):
        semantic_prototype[:, [k]] = torch.mean(objective_feat[k], dim=1).view(-1, 1).cpu()
    logger.info('Semantic prototype finised!')
    return semantic_prototype


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Testing")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("ICCV2021", output_dir, 0)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    semantic_prototype = get_source_centroids(cfg)
    torch.save(semantic_prototype, os.path.join(output_dir, 'semantic_prototype.pth'))


if __name__ == "__main__":
    main()
