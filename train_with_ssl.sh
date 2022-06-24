export NGPUS=4

#### GTA -> Cityscapes (ResNet-101)
##  warm-up
python -m torch.distributed.launch --nproc_per_node=$NGPUS warmup.py -cfg configs/deeplabv2_r101_adv.yaml OUTPUT_DIR results/r101_g2c_adv/ SOLVER.BATCH_SIZE 8
CUDA_VISIBLE_DEVICES=1 python test.py -cfg configs/deeplabv2_r101_adv.yaml resume results/r101_g2c_adv/ OUTPUT_DIR results/r101_g2c_adv/ SOLVER.BATCH_SIZE 8
## get semantic prototypes & target pseudo label
CUDA_VISIBLE_DEVICES=1 python semantic_prototype.py -cfg configs/deeplabv2_r101_prepare.yaml resume results/r101_g2c_adv/model_best.pth OUTPUT_DIR results/r101_g2c_adv/ TEST.BATCH_SIZE 1
CUDA_VISIBLE_DEVICES=1 python pseudo_label.py -cfg configs/deeplabv2_r101_prepare.yaml resume results/r101_g2c_adv/model_best.pth OUTPUT_DIR results/r101_g2c_adv/ SOLVER.BATCH_SIZE 8
## train with the proposed method
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_ours.py -cfg configs/deeplabv2_r101_cl.yaml resume results/r101_g2c_adv/model_best.pth OUTPUT_DIR results/r101_g2c_ours/ CENTROID_DIR results/r101_g2c_adv/ LABEL_DIR results/r101_g2c_adv/inference/cityscapes_train SOLVER.CPA_LOSS 1.0 SOLVER.ALPHA 0.1 SOLVER.BATCH_SIZE 8
CUDA_VISIBLE_DEVICES=1 python test.py -cfg configs/deeplabv2_r101_cl.yaml resume results/r101_g2c_ours/ OUTPUT_DIR results/r101_g2c_ours/ SOLVER.BATCH_SIZE 8
CUDA_VISIBLE_DEVICES=1 python pseudo_label.py -cfg configs/deeplabv2_r101_prepare.yaml resume results/r101_g2c_ours/model_best.pth OUTPUT_DIR results/r101_g2c_ours/ SOLVER.BATCH_SIZE 8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_ssl.py -cfg configs/deeplabv2_r101_tgt_self_distill.yaml LABEL_DIR results/r101_g2c_ours/inference/cityscapes_train OUTPUT_DIR results/r101_g2c_ours_ssl/ resume results/r101_g2c_ours/model_best.pth SOLVER.BATCH_SIZE 8
