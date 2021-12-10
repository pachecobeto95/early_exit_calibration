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
from load_dataset import loadCifar10, loadCifar100
import argparse
from networks.resnet import resnet152
from utils import verify_stop_condition, create_dir
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def trainEvalModel(model, dataLoader, criterion, optimizer, train):
	if(train):
		model.train()
	else:
		model.eval()


	acc_list, loss_list = [], []
	softmax = nn.Softmax(dim=1)
	for (data, target) in tqdm(dataLoader):
		data, target = data.to(device), target.to(device)

		if (train):
			optimizer.zero_grad()
			output = model(data).squeeze()
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()

		else:
			with torch.no_grad():
				output = model(data).squeeze()
				loss = criterion(output, target)

		_, infered_class = torch.max(softmax(output), 1)
		acc_list.append(100*infered_class.eq(target.view_as(infered_class)).sum().item()/target.size(0))
		loss_list.append(loss.item())

	avg_acc = np.mean(acc_list)
	avg_loss = np.mean(loss_list)

	print("%s Loss: %s Loss"%('Train' if train else 'Eval', np.mean(avg_loss)))
	print("%s Acc: %s Acc"%('Train' if train else 'Eval', np.mean(acc_list)))

	mode = "train" if(train) else "val"
	return {"%s_loss"%(mode): avg_loss, "%s_acc"%(mode): avg_acc}


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Training the backbone of a MobileNetV2')
	parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate (default: 0.0001)')
	parser.add_argument('--weight_decay', type=float, default= 0.0001, help='Weight Decay (default: 0.0001)')
	parser.add_argument('--opt', type=str, default= "SGD", help='Optimizer (default: RMSProp)')
	parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (default: 0.9)')
	parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning Rate Decay (default: 0.98)')
	parser.add_argument('--batch_size', type=int, default=128, help='Batch Size (default: 128)')
	parser.add_argument('--seed', type=int, default=42, help='Seed (default: 42)')
	parser.add_argument('--split_rate', type=float, default=0.1, help='Split rate of the dataset (default: 0.1)')
	parser.add_argument('--patience', type=int, default=10, help='Patience (default: 10)')
	parser.add_argument('--n_epochs', type=int, default=500, help='Number of epochs (default: 500)')
	parser.add_argument('--model_id', type=int, default=1, help='Model ID (default: 1)')
	parser.add_argument('--pretrained', dest='pretrained', action='store_false', default=True, help='Pretrained (default:True)')
	parser.add_argument('--dataset_name', type=str, default="cifar10", 
		choices=["cifar10", "cifar100"], help='Pretrained (default: True)')
	parser.add_argument('--lr_scheduler', type=str, default="stepRL", 
		choices=["stepRL", "plateau", "cossine"], help='Learning Rate Scheduler (default: stepRL)')
	parser.add_argument('-warm', type=int, default=1, help='Warm up training phase (default: 1)')


	args = parser.parse_args()

	root_path = os.path.dirname(__file__)
	dataset_path = os.path.join(root_path, "dataset")
	model_dir_path = os.path.join(root_path, "resnet152", "models")
	history_dir_path = os.path.join(root_path, "resnet152", "history")
	mode = "ft" if(args.pretrained) else "scratch"

	model_path = os.path.join(model_dir_path, "resnet152_%s_main_id_%s_%s.pth"%(args.dataset_name, args.model_id, mode))
	history_path = os.path.join(history_dir_path, "history_%s_resnet152_main_id_%s_%s.csv"%(args.dataset_name, args.model_id, mode))
	
	indices_dir_path = os.path.join(root_path, "indices")

	create_dir(model_dir_path, history_dir_path)

	n_classes = 10 if(args.dataset_name == "cifar10") else 100
	input_size = 224 if (args.pretrained) else 32
	crop_size = 224 if (args.pretrained) else 32
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	count, epoch = 0, 0
	best_val_loss = np.inf
	df = pd.DataFrame()

	if(args.pretrained):
		model = models.resnet152(pretrained=args.pretrained)
		model.fc = nn.Linear(model.fc.in_features, n_classes)
	else:
		model = resnet152(n_classes)

	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	
	if(args.dataset_name=="cifar10"):
		train_loader, val_loader, test_loader = loadCifar10(dataset_path, indices_dir_path, args.model_id, 
			args.batch_size, input_size, crop_size, split_rate=args.split_rate, seed=args.seed)
	else:
		train_loader, val_loader, test_loader = loadCifar100(dataset_path, indices_dir_path, args.model_id, 
			args.batch_size, input_size, crop_size, split_rate=args.split_rate, seed=args.seed)

	if(args.opt == "RMSProp"):
		optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, 
			alpha=args.momentum, weight_decay=args.weight_decay, momentum=args.momentum)
	
	elif(args.opt == "SGD"):
		optimizer = optim.SGD(model.parameters(), lr=args.lr, 
			momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
	else:
		optimizer = optim.Adam(model.parameters(), lr=args.lr,
			weight_decay=args.weight_decay)

	if(args.lr_scheduler == "stepRL"):
		scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 180], gamma=args.lr_decay, 
			last_epoch=-1, verbose=True)

	elif(args.lr_scheduler == "plateau"):
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay, 
			patience=int(args.patience/2), verbose=True)
	else:
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, verbose=True)

	while (epoch <= args.n_epochs):
		epoch += 1
		print("Current Epoch: %s"%(epoch))

		result = {}
		result_train = trainEvalModel(model, train_loader, criterion, optimizer, train=True)

		if epoch <= args.warm+1:
			for g in optimizer.param_groups:
				g['lr'] = 0.1
		
		result_val = trainEvalModel(model, val_loader, criterion, optimizer, train=False)
		scheduler.step()
		result.update(result_train), result.update(result_val) 

		df = df.append(pd.Series(result), ignore_index=True)
		df.to_csv(history_path)

		if (result["val_loss"] < best_val_loss):
			best_val_loss = result["val_loss"]
			count = 0
			save_dict = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
			"epoch": epoch, "val_loss": result["val_loss"]}
    
			torch.save(save_dict, model_path)

		else:
			print("Current Patience: %s"%(count))
			count += 1

	print("Stop! Patience is finished")
	trainEvalModel(model, test_loader, criterion, optimizer, train=False)
