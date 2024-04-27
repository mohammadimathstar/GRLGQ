import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from typing import Tuple
from torch import Tensor
from torch.utils.data import random_split


import copy
from abc import ABC
from typing import Any


class BaseTransform(ABC):
    """
    An abstract base class for writing transforms.
    Transforms are a general way to modify and customize.
    """
    def __call__(self, data: Any) -> Any:
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(data))

    def forward(self, data: Any) -> Any:
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class SVDFeatureReduction(BaseTransform):
    r"""Dimensionality reduction of node features via Singular Value
    Decomposition (SVD) (functional name: :obj:`svd_feature_reduction`).

    Args:
        out_channels (int): The dimensionlity of node features after
            reduction.
    """
    def __init__(self, imgsize: Tuple, out_channels: int):
        self.out_channels = out_channels
        self.imgsize = imgsize

    def forward(self, images: Tensor) -> Tensor:
        images_flatten = images.view(-1, self.imgsize[0] * self.imgsize[1]).T
        U, _, _ = torch.linalg.svd(images_flatten)
        return U[:, :self.out_channels]


def split_train_test(dataset, test_rate: float = 0.2, seed: int = 42):
	gen = torch.Generator().manual_seed(seed)
	d = random_split(dataset, [1 - test_rate, test_rate], generator=gen)
	return d


class DataSet(Dataset):
	def __init__(self, data_folder: str, img_size: Tuple, dim: int):
		self.data_folder = data_folder
		self.img_size = img_size
		self.dim_of_subspace = dim

		self.cls_folders = sorted([
			os.path.join(self.data_folder, f) for f in os.listdir(self.data_folder) if os.path.isdir(os.path.join(self.data_folder, f))
		])

		self.img_set_folders = sorted([
			os.path.join(c, f)
			for c in self.cls_folders
			for f in os.listdir(c)
			if os.path.isdir(os.path.join(c, f))
		])

		self.cls = [f.split("/")[-1] for f in self.cls_folders]
		self.cls2idx = {c: i for i, c in enumerate(self.cls)}
		self.idx2cls = {i: c for i, c in enumerate(self.cls)}

		self.transform_to_tensor = transforms.ToTensor()

		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Resize(size=self.img_size),
			transforms.Grayscale(),
		])
		self.svd = SVDFeatureReduction(imgsize=self.img_size, out_channels=dim)

	def __len__(self):
		return len(self.img_set_folders)

	def __getitem__(self, index):

		images = torch.stack([
			self.transform(
				cv2.imread(os.path.join(self.img_set_folders[index], f))
			)[0]
			for f in os.listdir(self.img_set_folders[index]) if os.path.isfile(
				os.path.join(self.img_set_folders[index], f)
			)
		], dim=0)

		subspace = self.svd(images)

		return subspace, self.cls2idx[self.img_set_folders[index].split("/")[-2]]



# if __name__ == '__main__':
# 	dataset_train = DataSet('./data/ETH-80/', img_size=(20, 20), dim=10)
# 	# train_loader = DataLoader(dataset_train, batch_size=5, shuffle=True)
# 	# for x, y in train_loader:
# 	# 	print(x.shape, y)
# 	# 	break
#
# 	p = split_train_test(dataset_train, test_rate=0.5, seed=42)
# 	pp = DataLoader(p[0], batch_size=20, shuffle=True)
# 	# ppp = DataLoader(p[1], batch_size=20, shuffle=True)
# 	for x, y in pp:
# 		print(x.shape, y)
# 		break
# 	for x, y in pp:
# 		print(x.shape, y)
