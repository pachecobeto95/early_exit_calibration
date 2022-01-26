import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
import os, sys, time, math, glob
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from PIL import Image
import torch
import numpy as np
from torch import Tensor
from typing import Callable, Any, Optional, List, Type, Union
import torch.nn.init as init
import functools
from tqdm import tqdm

DIR_NAME = os.path.dirname(__file__)
dataset_root_path = os.path.join(DIR_NAME, "datasets", "caltech256")
dataset_path = os.path.join(dataset_root_path, "256_ObjectCategories")
indices_path = os.path.join(dataset_root_path, "indices")
test_idx_path = os.path.join(indices_path, "test_idx_caltech256_id_1.npy")
test_idx = np.load(test_idx_path, allow_pickle=True)
test_idx = np.array(list(test_idx.tolist()))

save_path = os.path.join(dataset_root_path, "test_dataset")

transformations_test = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor()])

data_set = datasets.ImageFolder(dataset_path, transform=transformations_test)
#test_loader = generate_test_dataset(data_set, train_idx_path, val_idx_path)
test_data = torch.utils.data.Subset(data_set, indices=test_idx)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=4)


if (not os.path.exists(save_path)):
	os.makedirs(save_path)

for i, (data, target) in tqdm(enumerate(test_loader, 1)):
  save_image(data, os.path.join(save_path, "%s_%s.jpg"%(i, target.item())))

