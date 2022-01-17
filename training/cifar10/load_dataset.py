import torchvision.transforms as transforms
import torchvision, torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
import os, sys, time, math, os, ssl
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np


def loadCifar10(root_path, indices_path, model_id, batch_size_train, batch_size_test, input_size, crop_size, split_rate=0.1, seed=42):
	ssl._create_default_https_context = ssl._create_unverified_context

	np.random.seed(seed)
	torch.manual_seed(seed)

	mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

	transform_train = transforms.Compose([
		transforms.Resize(input_size),
		transforms.RandomCrop(crop_size, padding = 4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)])
    
	transform_test = transforms.Compose([
		transforms.Resize(input_size),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)])

	trainset = CIFAR10(root_path, transform=transform_train, train=True, download=True)

	indices = np.arange(len(trainset))

	# This line defines the size of training dataset.
	train_size = int(len(indices) - int(split_rate*len(indices)))

	np.random.shuffle(indices)
	train_idx, val_idx = indices[:train_size], indices[train_size:]


	np.save('train_id_%s.npy'%(model_id), train_idx)
	np.save('val_id_%s.npy'%(model_id), val_idx)

	train_data = torch.utils.data.Subset(trainset, indices=train_idx)
	val_data = torch.utils.data.Subset(trainset, indices=val_idx)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True, num_workers=4)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size_test, shuffle=False, num_workers=4)

	testset = CIFAR10(root_path, transform=transform_test, train=False, download=True)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle = False, num_workers=4)

	return train_loader, val_loader, testloader

def loadCifar100(root_path, indices_path, model_id, batch_size_train, batch_size_test, input_size, crop_size, split_rate=0.1, seed=42):
	ssl._create_default_https_context = ssl._create_unverified_context

	np.random.seed(seed)
	torch.manual_seed(seed)

	mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

	transform_train = transforms.Compose([
		transforms.Resize(input_size),
		transforms.RandomCrop(crop_size, padding = 4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)])

	transform_test = transforms.Compose([
		transforms.Resize(input_size),
		transforms.RandomCrop(crop_size, padding = 4),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)])

	trainset = CIFAR100(root_path, transform=transform_train, train=True, download=True)

	indices = np.arange(len(trainset))

	# This line defines the size of training dataset.
	train_size = int(len(indices) - int(split_rate*len(indices)))

	np.random.shuffle(indices)
	train_idx, val_idx = indices[:train_size], indices[train_size:]


	np.save('train_indices_cifar100_id_%s.npy'%(model_id), train_idx)
	np.save('val_indices_cifar100_id_%s.npy'%(model_id), val_idx)

	train_data = torch.utils.data.Subset(trainset, indices=train_idx)
	val_data = torch.utils.data.Subset(trainset, indices=val_idx)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True, num_workers=4)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size_test, shuffle=False, num_workers=4)

	testset = CIFAR100(root_path, transform=transform_train, train=False, download=True)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=4)

	return train_loader, val_loader, testloader
