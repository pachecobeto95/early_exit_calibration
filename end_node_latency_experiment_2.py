import os, config, time
import requests, sys, json, os
import numpy as np
from PIL import Image
import pandas as pd
import argparse
#from utils import LoadDataset
from requests.exceptions import HTTPError, ConnectTimeout
from glob import glob
#import torch
from load_dataset import load_test_caltech_256


def load_dataset(args, dataset_path, savePath_idx):
	if(args.dataset_name=="caltech256"):
		return load_test_caltech_256(args.input_dim, dataset_path, args.split_ratio, savePath_idx)

	elif(args.dataset_name=="cifar100"):
		sys.exit()

def main(args):
	#Number of side branches that exists in the early-exit DNNs
	#nr_branches_model_list = np.arange(config.nr_min_branches, config.nr_max_branches+1)

	nr_branches_model = args.n_branches
	p_tar_list = np.arange(0.5, 0.95, 0.05)
	dataset_path = dataset_path[args.dataset_name]
	
	root_save_path = os.path.dirname(__file__)

	save_indices_path = config.save_indices_path if(args.dataset_name) else None


	#This line defines the number of side branches processed at the cloud
	nr_branch_edge = np.arange(2, nr_branches_model+1)
	print("Sending Confs")
	sendModelConf(config.urlConfModelEdge, nr_branches_model, args.dataset_name, args.model_name)
	sendModelConf(config.urlConfModelCloud, nr_branches_model, args.dataset_name, args.model_name)
	print("Finish Confs")

	test_loader = load_dataset(args, dataset_path, save_indices_path)





if (__name__ == "__main__"):
	# Input Arguments. Hyperparameters configuration
	parser = argparse.ArgumentParser(description="Evaluating early-exits DNNs perfomance")
	parser.add_argument('--n_branches', type=int, default=config.nr_max_branches,
		choices=list(range(config.nr_min_branches, config.nr_max_branches+1)), 
		help='Number of branches in the early-exit DNNs model (default: %s)'%(config.nr_max_branches))

	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech256", "cifar10", "cifar100"], 
		help='Dataset name (default: Caltech-256)')

	parser.add_argument('--model_name', type=str, default=config.model_name, choices=["mobilenet", "resnet18", "resnet50", "vgg16"], 
		help='DNN model name (default: MobileNet)')

	parser.add_argument('--input_dim', type=int, default=224, choices=[224, 32], 
		help='Input Dim')

	parser.add_argument('--split_ratio', type=float, default=0.2, help='Split Ratio')

	args = parser.parse_args()


	main(args)
