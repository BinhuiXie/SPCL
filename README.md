# SPCL
A New Framework for Domain Adaptive Semantic Segmentation via Semantic Prototype-based Contrastive Learning by [Binhui Xie](https://binhuixie.github.io), [Mingjia Li](https://github.com/KiwiXR), [Shuang Li](https://shuangli.xyz).

## Update
**2021/11/25: arXiv version of [SPCL](https://arxiv.org/abs/2111.12358) is available.**

**2022/06/24: Code is released.**

If you find it useful for your research, please cite 
```
@article{xie2021spcl,
  title={SPCL: A New Framework for Domain Adaptive Semantic Segmentation via Semantic Prototype-based Contrastive},
  author={Binhui Xie, Mingjia Li, Shuang Li},
  journal={arXiv preprint arXiv:2111.12358},
  year={2021}
}

```

### Prerequisites
- Python 3.6
- torch 1.7.1
- torchvision 0.8.2
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.1
### Step-by-step installation

```bash
conda create --name spcl -y python=3.6
conda activate spcl

# this installs the right pip and dependencies for the fresh python
conda install -y ipython pip

pip install torch==1.7.1 torchvision==0.8.2 ninja yacs cython matplotlib tqdm opencv-python imageio mmcv
```

### Getting started

- Download [The GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/)

- Download [The SYNTHIA Dataset](http://synthia-dataset.net/download/808/)

- Download [The Synscapes Dataset](https://7dlabs.com/synscapes-overview)

- Download [The Cityscapes Dataset](https://www.cityscapes-dataset.com/)

- Symlink the required dataset
```bash
ln -s /path_to_gta5_dataset datasets/gta5
ln -s /path_to_synthia_dataset datasets/synthia
ln -s /path_to_synscapes_dataset datasets/synscapes
ln -s /path_to_cityscapes_dataset datasets/cityscapes
```

- Generate the label statics file for GTA5 and SYNTHIA Datasets by running 
```bash
python datasets/generate_gta5_label_info.py -d datasets/gta5 -o datasets/gta5/
python datasets/generate_synthia_label_info.py -d datasets/synthia -o datasets/synthia/
```

The data folder should be structured as follows:
```
├── datasets/
│   ├── cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
│   ├── gta5/
|   |   ├── images/
|   |   ├── labels/
|   |   ├── gtav_label_info.p
│   ├── synthia/
|   |   ├── RAND_CITYSCAPES/
|   |   ├── synthia_label_info.p
│   ├── synscapes/
|   |   ├── img/rgb-2k
|   |   ├── img/class
│   └── 			
...
```

### Train
We provide the training script using 4 Tesla V100 GPUs.
```bash
bash train_with_ssl.sh
```

### Evaluate

Tip: For those who are interested in how performance change during the process of adversarial training, test.py also accepts directory as the input and the results will be stored in a csv file.

```bash
python test.py -cfg configs/deeplabv2_r101_tgt_ssl.yaml resume results/r101_g2c_ours_ssl/ OUTPUT_DIR results/r101_g2c_ours_ssl/ SOLVER.BATCH_SIZE 8
```


## Acknowledgments

This project is based on the following open-source projects: [FADA](https://github.com/JDAI-CV/FADA) and [SDCA](https://github.com/BIT-DA/SDCA). We thank authors for making the source code publically available. 
