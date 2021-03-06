import os
from .cityscapes import cityscapesDataSet
from .cityscapes_ssl import cityscapesSoftDataSet
from .synthia import synthiaDataSet
from .gta5 import GTA5DataSet
from .synscapes import SynscapesDataSet


class DatasetCatalog(object):
    DATASET_DIR = "datasets"
    DATASETS = {
        "gta5_train": {
            "data_dir": "gta5",
            "data_list": "gta5_train_list.txt"
        },
        "synthia_train": {
            "data_dir": "synthia",
            "data_list": "synthia_train_list.txt"
        },
        "synscapes_train": {
            "data_dir": "synscapes",
            "data_list": "synscapes_train_list.txt"
        },
        "cityscapes_train": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt"
        },
        "cityscapes_train_soft": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt",
            "label_dir": "cityscapes/soft_labels/inference/cityscapes_train"
        },
        "cityscapes_val": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_val_list.txt"
        },
    }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None, cfg=None):
        if "gta5" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return GTA5DataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                               split=mode, transform=transform)
        elif "synthia" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return synthiaDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                  split=mode, transform=transform)
        elif "synscapes" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return SynscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                    split=mode, transform=transform)
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if 'soft' in name:
                if cfg.LABEL_DIR:
                    args['label_dir'] = cfg.LABEL_DIR
                    print("Loading Cityscapes_train soft label from{}".format(args['label_dir']))
                else:
                    args['label_dir'] = os.path.join(data_dir, attrs["label_dir"])
                return cityscapesSoftDataSet(args["root"], args["data_list"], args['label_dir'],
                                             max_iters=max_iters, num_classes=num_classes,
                                             split=mode, transform=transform)
            return cityscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        raise RuntimeError("Dataset not available: {}".format(name))
