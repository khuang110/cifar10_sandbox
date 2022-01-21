import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typing import Any, Callable, Tuple, Optional

from PIL import Image


class CIFar10(Dataset):
    def __init__(
        self,
        train_set,
        transform: Optional[Callable] = None,
        y_transform: Optional[Callable] = None,
    ):
        self._X = train_set[0]
        self._X = np.vstack(self._X).reshape(-1, 3, 32, 32)
        self._X = self._X.transpose((0, 2, 3, 1))  # convert to HWC
        self._y = train_set[1]
        self._transform = transform
        self._y_transform = y_transform

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self._X[idx, :], self._y[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self._transform is not None:
            img = self._transform(img)

        if self._y_transform is not None:
            target = self._y_transform(target)

        return img, target
