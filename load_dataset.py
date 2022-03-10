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



def get_indices(train_set, split_ratio):
	nr_samples = len(train_set)
	indices = list(range(nr_samples))

	train_size = nr_samples - int(np.floor(split_ratio * nr_samples))

	np.random.shuffle(indices)

	train_idx, test_idx = indices[:train_size], indices[train_size:]

	return train_idx, test_idx

def load_test_caltech_256(input_dim, dataset_path, split_ratio, savePath_idx, model_id, seed=42):
	# This method loads the Caltech-256 dataset.


	#To normalize the input images data.
	mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
	std =  [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	# Note that we do not apply data augmentation in the test dataset.
	transformations_test = transforms.Compose([
		transforms.Resize(input_dim), 
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])


	torch.manual_seed(seed)
	np.random.seed(seed=seed)

	train_set = datasets.ImageFolder(dataset_path)

	val_set = datasets.ImageFolder(dataset_path, transform=transformations_test)
    
	test_set = datasets.ImageFolder(dataset_path, transform=transformations_test)

	test_idx_path = os.path.join(savePath_idx, "test_idx_caltech256_id_%s.npy"%(model_id))
	val_idx_path = os.path.join(savePath_idx, "validation_idx_caltech256_id_%s.npy"%(model_id))

	val_idx = np.load(val_idx_path, allow_pickle=True)
	val_idx = np.array(list(val_idx.tolist()))
	

	if (os.path.exists(test_idx_path)):
		test_idx = np.load(test_idx_path, allow_pickle=True)
		test_idx = np.array(list(test_idx.tolist()))
	else:
		_, test_idx = get_indices(train_set, split_ratio)

	#nr_samples = len(train_set)
	#indices = list(range(nr_samples))

	#train_size = nr_samples - int(np.floor(split_ratio * nr_samples))

	#np.random.shuffle(indices)

	#_, test_idx2 = indices[:train_size], indices[train_size:]

	#print((np.array(test_idx2)==test_idx).all())
	test_data = torch.utils.data.Subset(test_set, indices=test_idx)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=4)

	val_data = torch.utils.data.Subset(val_set, indices=val_idx)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=4)


	return val_loader 


