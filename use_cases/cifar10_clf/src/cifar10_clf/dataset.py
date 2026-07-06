from pathlib import Path
from typing import Callable, Optional
import mlflow.data.schema
import mlflow.types
import numpy as np
from torchvision.datasets import CIFAR10, ImageFolder
from PIL.Image import Image
import mlflow


class LazyCIFAR10Combined(ImageFolder, mlflow.data.Dataset):

    @staticmethod
    def from_original_cifar_10(
        path_root_raw: Path,
        path_root_processed: Path,
        **dataset_kwargs,
    ):
        ds_train = CIFAR10(root=str(path_root_raw), train=True, download=True)
        cls_idx_to_cls_name = {v: k for k, v in ds_train.class_to_idx.items()}
        img: Image
        for i in range(len(ds_train)):
            img, cls_idx = ds_train[i]
            cls_name = cls_idx_to_cls_name[cls_idx]
            path_cls = path_root_processed / cls_name
            path_cls.mkdir(exist_ok=True)
            with open(path_cls / f"{i}.png", "wb") as fp:
                img.save(fp)
        n_train = len(ds_train)
        ds_test = CIFAR10(root=str(path_root_raw), train=False, download=False)
        for i in range(len(ds_test)):
            img, cls_idx = ds_test[i]
            cls_name = cls_idx_to_cls_name[cls_idx]
            path_cls = path_root_processed / cls_name
            with open(path_cls / f"{n_train+i}.png", "wb") as fp:
                img.save(fp)

        return LazyCIFAR10Combined(root=str(path_root_processed), **dataset_kwargs)

    def extra_repr(self) -> str:
        return ""
