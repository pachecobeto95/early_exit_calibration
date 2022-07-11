import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
import os, sys, time, math, os
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import random_split
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import product
import pandas as pd
import torchvision.models as models
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch import Tensor
import functools, os
from tqdm import tqdm
from networks.mobilenet import MobileNetV2_2
from utils import create_dir
from load_dataset import loadCifar10, loadCifar100
import argparse


def loadCifar10(batch_size, input_size, crop_size, split_rate, seed=42):
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

	trainset = CIFAR10(".", transform=transform_train, train=True, download=True)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

	testset = CIFAR10(".", transform=transform_test, train=False, download=True)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle = False, num_workers=4)

	return train_loader, testloader


if (__name__ == "__main__"):
	parser = argparse.ArgumentParser(description='Training the Early-Exit MobileNetV2.')

	parser.add_argument("--lr", type=float, default=0.045, help='Learning Rate (default: 0.045)')
	parser.add_argument('--weight_decay', type=float, default= 0.00004, help='Weight Decay (default: 0.00004)')
	parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (default: 0.9)')
	parser.add_argument('--lr_decay', type=float, default=0.98, help='Learning Rate Decay (default: 0.98)')
	parser.add_argument('--batch_size', type=int, default=512, help='Batch Size (default: 512)')
	parser.add_argument('--seed', type=int, default=42, help='Seed (default: 42)')
	parser.add_argument('--split_rate', type=float, default=0.2, help='Split rate of the dataset (default: 0.2)')
	parser.add_argument('--patience', type=int, default=10, help='Patience (default: 10)')
	parser.add_argument('--n_epochs', type=int, default=300, help='Number of epochs (default: 300)')
	parser.add_argument('--dataset_name', type=str, default="cifar10", 
		choices=["cifar10", "cifar100"], help='Pretrained (default: True)')
	parser.add_argument('--lr_scheduler', type=str, default="stepRL", 
		choices=["stepRL", "plateau", "cossine"], help='Learning Rate Scheduler (default: stepRL)')
	parser.add_argument('--n_branches', type=int, default=5, help='Number of side branches (default: 5)')
	parser.add_argument('--distribution', type=str, default="linear", help='Distribution of Branches (default: 1)')
	parser.add_argument('--exit_type', type=str, default="bnpool", 
		choices=["bnpool", "plain"], help='Exit Block Type (default: bnpool)')
	parser.add_argument('--loss_weight_type', type=str, default="crescent", 
		choices=["crescent", "decrescent", "equal"], help='Loss Weight (default: decrescent)')

	args = parser.parse_args()

	root_path = os.path.dirname(__file__)
	n_classes = 10 if(args.dataset_name == "cifar10") else 100
	input_size, crop_size = 32, 32
	input_shape = (3, input_size, input_size)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	count, epoch = 0, 0
	best_val_loss = np.inf
	n_exits = args.n_branches + 1
	model_name = "mobilenet"

	criterion = nn.CrossEntropyLoss()
	loss_dict = {"crescent": np.linspace(0.15, 1, n_exits), "decrescent": np.linspace(1, 0.15, n_exits), 
	"equal": np.ones(n_exits)}

	loss_weights = loss_dict[args.loss_weight_type]

	#model = Early_Exit_DNN(model_name, n_classes, args.n_branches, input_shape, 
	#	args.exit_type, device, distribution=args.distribution)
	
	#model = model.to(device)

	train_loader, val_loader, test_loader = loadCifar10(args.batch_size, input_size, crop_size, args.split_rate, seed=args.seed)















