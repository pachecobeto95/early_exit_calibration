import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
import os, sys, time, math, torch, argparse, functools
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import random_split
from PIL import Image
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
from tqdm import tqdm
from networks.mobilenet import MobileNetV2_2
from utils import create_dir_temperature, get_model_arch
from load_dataset import loadCifar10, loadCifar100
from calibration_dnn import MainModelCalibration

if (__name__ == "__main__"):
	parser = argparse.ArgumentParser(description='Calibrating the backbone of a MobileNetV2')
	parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate (default: 0.01)')
	parser.add_argument('--max_iter', type=int, default= 1000, help='Max Iter (default: 1000)')
	parser.add_argument('--model_id', type=int, default=1, help='Model ID (default: 1)')
	parser.add_argument('--pretrained', dest='pretrained', action='store_false', default=True, help='Pretrained (default:True)')
	parser.add_argument('--dataset_name', type=str, default="cifar10", 
		choices=["cifar10", "cifar100"], help='Dataset Name (default: cifar10)')
	parser.add_argument('--seed', type=int, default=42, help='Seed (default: 42)')
	parser.add_argument('--model_name', type=str, default="mobilenet", 
		choices=["mobilenet", "vgg16", "resnet18", "resnet152"], help='Model Name (default: mobilenet)')
	parser.add_argument('--batch_size', type=int, default=128, help='Batch Size (default: 128)')
	parser.add_argument('--split_rate', type=float, default=0.1, help='Split rate of the dataset (default: 0.1)')


	args = parser.parse_args()
	root_path = os.path.dirname(__file__)

	dataset_path = os.path.join(root_path, "dataset")
	network_dir_path = os.path.join(root_path, args.model_name) 
	model_dir_path = os.path.join(network_dir_path, "models")
	history_dir_path = os.path.join(network_dir_path, "history")
	temp_dir_path = os.path.join(network_dir_path, "temperature")
	indices_dir_path = os.path.join(root_path, "indices")
	
	mode = "ft" if(args.pretrained) else "scratch"
	input_size = 224 if (args.pretrained) else 32
	crop_size = 224 if (args.pretrained) else 32
	n_classes = 10 if (args.dataset_name == "cifar10") else 100
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model_path = os.path.join(model_dir_path, "%s_main_%s_id_%s_%s.pth"%(args.model_name, args.dataset_name, args.model_id, mode))
	save_temp_path = os.path.join(temp_dir_path, "temp_%s_main_%s_id_%s_%s.pth"%(args.model_name, args.dataset_name, args.model_id, mode))
	
	create_dir_temperature(temp_dir_path)

	model = get_model_arch(args.pretrained, args.model_name, n_classes, device).to(device)

	if(args.dataset_name=="cifar10"):
		train_loader, val_loader, test_loader = loadCifar10(dataset_path, indices_dir_path, args.model_id, 
			args.batch_size, input_size, crop_size, split_rate=args.split_rate, seed=args.seed)

	else:
		train_loader, val_loader, test_loader = loadCifar100(dataset_path, indices_dir_path, args.model_id, 
			args.batch_size, input_size, crop_size, split_rate=args.split_rate, seed=args.seed)


	scaled_model = MainModelCalibration(model, device, model_path, save_temp_path, args.lr, args.max_iter)
	scaled_model.set_temperature(val_loader)

