import torchvision.transforms as transforms
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import os, sys, time, math, os
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import random_split
from PIL import Image
import torch
import numpy as np


def load_test_caltech_256(input_dim, dataset_path, split_ratio, savePath_idx, model_id=1):
	# This method loads the Caltech-256 dataset.


	#To normalize the input images data.
	mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
	std =  [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	# Note that we do not apply data augmentation in the test dataset.
	transformations_test = transforms.Compose([
		transforms.Resize(input_dim), 
		transforms.ToTensor(), 
		#transforms.Normalize(mean = mean, std = std),
		])


	torch.manual_seed(self.seed)
	np.random.seed(seed=self.seed)

	train_set = datasets.ImageFolder(dataset_path)

	val_set = datasets.ImageFolder(dataset_path, transform=transformations_test)
    
	test_set = datasets.ImageFolder(dataset_path, transform=transformations_test)

	if (os.path.exists(os.path.join(savePath_idx, "test_idx_%s_id_%s.npy"%(dataset_name, model_id)))):
		test_idx = np.load(os.path.join(savePath_idx, "test_idx_%s_id_%s.npy"%(dataset_name, model_id)), allow_pickle=True)
		test_idx = np.array(list(test_idx.tolist()))
	else:
		_, test_idx = self.get_indices(train_set, split_ratio)


	test_data = torch.utils.data.Subset(test_set, indices=test_idx)

	test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=4)

	return test_loader 


