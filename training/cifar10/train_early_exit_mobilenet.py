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
from train import trainEvalEarlyExit
from early_exit_dnns import Early_Exit_DNN

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Training the Early Exit of a MobileNetV2. B-Mobilenet')
	parser.add_argument('--lr_backbone', type=float, default=0.045, help='Learning Rate (default: 0.045)')
	parser.add_argument('--lr_branches', type=float, default=1.5e-4, help='Learning Rate (default: 1.5e-4)')
	parser.add_argument('--weight_decay', type=float, default= 0.00004, help='Weight Decay (default: 0.00004)')
	parser.add_argument('--opt', type=str, default= "SGD", 
		choices=["SGD", "RMSProp", "Adam"], help='Optimizer (default: SGD)')
	parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (default: 0.9)')
	parser.add_argument('--lr_decay', type=float, default=0.98, help='Learning Rate Decay (default: 0.98)')
	parser.add_argument('--batch_size', type=int, default=128, help='Batch Size (default: 96)')
	parser.add_argument('--seed', type=int, default=42, help='Seed (default: 42)')
	parser.add_argument('--split_rate', type=float, default=0.1, help='Split rate of the dataset (default: 0.1)')
	parser.add_argument('--patience', type=int, default=10, help='Patience (default: 10)')
	parser.add_argument('--n_epochs', type=int, default=300, help='Number of epochs (default: 300)')
	parser.add_argument('--model_id', type=int, default=1, help='Model ID (default: 1)')
	parser.add_argument('--pretrained', dest='pretrained', action='store_false', default=True, help='Pretrained (default:True)')
	parser.add_argument('--dataset_name', type=str, default="cifar10", 
		choices=["cifar10", "cifar100"], help='Pretrained (default: True)')
	parser.add_argument('--lr_scheduler', type=str, default="stepRL", 
		choices=["stepRL", "plateau", "cossine"], help='Learning Rate Scheduler (default: stepRL)')
	parser.add_argument('--n_branches', type=int, default=5, help='Number of side branches (default: 5)')
	parser.add_argument('--backbone_pretrained', dest='backbone_pretrained', 
		action='store_false', default=True, help='Pretrained (default:True)')
	parser.add_argument('--distribution', type=str, default="linear", help='Distribution of Branches (default: 1)')
	parser.add_argument('--exit_type', type=str, default="bnpool", 
		choices=["bnpool", "plain"], help='Exit Block Type (default: bnpool)')
	parser.add_argument('--loss_weight_type', type=str, default="crescent", 
		choices=["crescent", "decrescent", "equal"], help='Loss Weight (default: decrescent)')

	args = parser.parse_args()

	root_path = os.path.dirname(__file__)
	dataset_path = os.path.join(root_path, "dataset")
	model_dir_path = os.path.join(root_path, "mobilenet", "models")
	history_dir_path = os.path.join(root_path, "mobilenet", "history")
	mode = "ft" if(args.pretrained) else "scratch"
	mode_backbone = "backbone" if(args.backbone_pretrained) else ""

	backbone_model_path = os.path.join(model_dir_path, "mobilenet_main_%s_id_%s_%s.pth"%(args.dataset_name, args.model_id, mode))
	early_exit_model_path = os.path.join(model_dir_path, "b_mobilenet_early_exit_%s_id_%s_%s.pth"%(args.dataset_name, args.model_id, mode))
	history_path = os.path.join(history_dir_path, "history_b_mobilenet_early_exit_%s_id_%s_%s.csv"%(args.dataset_name, args.model_id, mode))
	indices_dir_path = os.path.join(root_path, "indices")

	model_name = "mobilenet"
	n_classes = 10 if(args.dataset_name == "cifar10") else 100
	input_size = 224 if (args.pretrained) else 32
	crop_size = 224 if (args.pretrained) else 32
	input_shape = (3, input_size, input_size)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	count, epoch = 0, 0
	best_val_loss = np.inf
	df = pd.DataFrame()
	n_exits = args.n_branches + 1


	criterion = nn.CrossEntropyLoss()
	loss_dict = {"crescent": np.linspace(0.15, 1, n_exits), "decrescent": np.linspace(1, 0.3, n_exits), 
	"equal": np.ones(n_exits)}

	loss_weights = loss_dict[args.loss_weight_type]

	early_exit_dnn = Early_Exit_DNN(model_name, n_classes, args.pretrained, args.backbone_pretrained, 
		backbone_model_path, args.n_branches, input_shape, args.exit_type, device, distribution=args.distribution)
	early_exit_dnn = early_exit_dnn.to(device)

	if(args.dataset_name=="cifar10"):
		train_loader, val_loader, test_loader = loadCifar10(dataset_path, indices_dir_path, args.model_id, 
			args.batch_size, input_size, crop_size, split_rate=args.split_rate, seed=args.seed)
	else:
		train_loader, val_loader, test_loader = loadCifar100(dataset_path, indices_dir_path, args.model_id, 
			args.batch_size, input_size, crop_size, split_rate=args.split_rate, seed=args.seed)

	
	if(args.opt == "SGD"):
		if(args.pretrained or args.backbone_pretrained):
			optimizer = optim.SGD([{'params': early_exit_dnn.stages.parameters(), 'lr': args.lr_backbone}, 
				{'params': early_exit_dnn.exits.parameters(), 'lr': args.lr_branches},
				{'params': early_exit_dnn.classifier.parameters(), 'lr': args.lr_backbone}], 
				momentum=args.momentum, weight_decay=args.weight_decay,
				nesterov=True)
		else:
			optimizer = optim.SGD(early_exit_dnn.parameters(), lr=args.lr_branches, 
				momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

	else:
		if(args.pretrained or args.backbone_pretrained):
			optimizer = optim.Adam([{'params': early_exit_dnn.stages.parameters(), 'lr': args.lr_backbone}, 
				{'params': early_exit_dnn.exits.parameters(), 'lr': args.lr_branches},
				{'params': early_exit_dnn.classifier.parameters(), 'lr': args.lr_backbone}], weight_decay=args.weight_decay)
		else:
			optimizer = optim.Adam(early_exit_dnn.parameters(), lr=args.lr_branches, weight_decay=args.weight_decay)


	if(args.lr_scheduler == "stepRL"):
		scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay, verbose=True)
	elif(args.lr_scheduler == "plateau"):
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay, 
			patience=int(args.patience/2), verbose=True)
	else:
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, verbose=True)

	while (count <= args.patience):
		epoch += 1
		print("Current Epoch: %s"%(epoch))

		result = {}
		result_train = trainEvalEarlyExit(early_exit_dnn, train_loader, criterion, optimizer, args.n_branches, 
			epoch, device, loss_weights, train=True)
		
		result_val = trainEvalEarlyExit(early_exit_dnn, val_loader, criterion, optimizer, args.n_branches, 
			epoch, device, loss_weights, train=False)
		
		scheduler.step()
		result.update(result_train), result.update(result_val) 

		df = df.append(pd.Series(result), ignore_index=True)
		df.to_csv(history_path)

		if (result["val_loss"] < best_val_loss):
			best_val_loss = result["val_loss"]
			count = 0
			save_dict = {"model_state_dict": early_exit_dnn.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
			"epoch": epoch, "val_loss": result["val_loss"]}
    
			torch.save(save_dict, early_exit_model_path)

		else:
			print("Current Patience: %s"%(count))
			count += 1

	print("Stop! Patience is finished")
	trainEvalModel(early_exit_dnn, test_loader, criterion, optimizer, train=False)