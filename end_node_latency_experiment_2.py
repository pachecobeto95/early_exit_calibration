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
from torchvision.utils import save_image
import logging


def load_dataset(args, dataset_path, savePath_idx):
	if(args.dataset_name=="caltech256"):
		return load_test_caltech_256(config.input_dim, dataset_path, args.split_ratio, savePath_idx, 
			config.model_id_dict[args.model_name])

	elif(args.dataset_name=="cifar100"):
		sys.exit()

def sendImage(img_path, idx, url, target, p_tar, nr_branch_edge, warmUp=False):

	#my_img = {'img': open(img_path, 'rb')}

	data_dict = {"p_tar": p_tar, "nr_branch": int(nr_branch_edge), "target": target.item(), "warmUp": warmUp, "id": idx}

	files = [
	('img', (img_path, open(img_path, 'rb'), 'application/octet')),
	('data', ('data', json.dumps(data_dict), 'application/json')),]

	try:
		r = requests.post(url, files=files, timeout=1000)
		r.raise_for_status()
	
	except HTTPError as http_err:
		raise SystemExit(http_err)

	except ConnectTimeout as timeout_err:
		print("Url: ¨%s, Timeout error: %s"%(url, timeout_err))

def sendData(url, data):
	try:
		r = requests.post(url, json=data, timeout=1000)
		r.raise_for_status()
	
	except HTTPError as http_err:
		raise SystemExit(http_err)
	except requests.Timeout:
		logging.warning("Timeout")
		pass
	#except ConnectTimeout as timeout_err:
	#	print("Timeout error: ", timeout_err)

def sendModelConf(url, n_branches, dataset_name, model_name):
	
	pretrained_str = "ft" if (model_name=="") else "scratch"
	
	data_dict = {"n_branches": n_branches, "dataset_name": dataset_name, "model_name": model_name, 
	"n_classes": config.models_params[dataset_name]["n_classes"], 
	"input_shape": config.models_params[dataset_name]["input_shape"],
	"model_id": config.model_id_dict[model_name],
	"pretrained": pretrained_str}

	sendData(url, data_dict)

def sendConfigExp(url, target, p_tar, nr_branch_edge):

	data_dict = {"target": target.item(), "p_tar": p_tar, "nr_branch": int(nr_branch_edge)}
	sendData(url, data_dict)

def warmUpDnnInference(test_loader):
	
	for i, (data, target) in enumerate(test_loader, 1):
		if(i >= config.warmUpSize):
			break

		filepath = os.path.join(config.save_img_dir_path, "%s_%s.jpg"%(target.item(), i))	
		save_image(data, filepath)
		sendConfigExp(config.url_cloud_config_exp, target, 1, 5)
		sendImage(filepath, i, config.url_edge_no_calib, target, 1, 5, warmUp=True)
		#sendImage(filepath, config.url_edge_overall_calib, target, 1, 5, warmUp=True)
		#sendImage(filepath, config.url_edge_branches_calib, target, 1, 5, warmUp=True)


def inferenceTimeExperiment(test_loader, p_tar_list, nr_branch_edge_list, logPath):
	if (not os.path.exists(config.save_img_dir_path)):
		os.makedirs(config.save_img_dir_path)

	test_set_size = len(test_loader)

	warmUpDnnInference(test_loader)
	logPathOpen = open(logPath, "a")

	for i, (data, target) in enumerate(test_loader, 1):
		#print("Image: %s/%s"%(i, test_set_size), file=logPathOpen)
		#print("Image: %s/%s"%(i, test_set_size))
		logging.debug("Image: %s/%s"%(i, test_set_size))

		filepath = os.path.join(config.save_img_dir_path, "%s_%s.jpg"%(target.item(), i))
		save_image(data, filepath)

		for nr_branch_edge in nr_branch_edge_list:

			# For a given number of branches processed in edge, this loop changes the threshold p_tar configuration.
			for p_tar in p_tar_list:
				#sendConfigExp(config.url_edge_config_exp, target, p_tar, nr_branch_edge)
				sendConfigExp(config.url_cloud_config_exp, target, p_tar, nr_branch_edge)
				sendImage(filepath, i, config.url_edge_no_calib, target, p_tar, nr_branch_edge)
				sendImage(filepath, i, config.url_edge_overall_calib, target, p_tar, nr_branch_edge)
				sendImage(filepath, i, config.url_edge_branches_calib, target, p_tar, nr_branch_edge)


def main(args):
	#Number of side branches that exists in the early-exit DNNs
	#nr_branches_model_list = np.arange(config.nr_min_branches, config.nr_max_branches+1)

	p_tar_list = [0.83, 0.86]
	dataset_path = config.models_params[args.dataset_name]["dataset_path"]

	logPath = "./logTest_%s_%s.log"%(args.model_name, args.dataset_name)

	logging.basicConfig(level=logging.DEBUG, filename=logPath, filemode="a+", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
	
	root_save_path = os.path.dirname(__file__)

	save_indices_path = config.models_params[args.dataset_name]["indices"]

	#This line defines the number of side branches processed at the edge
	nr_branch_edge = np.arange(3, config.nr_branch_model+1)

	#print("Sending Confs")
	logging.debug("Sending Confs")

	sendModelConf(config.urlConfModelEdge, config.nr_branch_model, args.dataset_name, args.model_name)
	sendModelConf(config.urlConfModelCloud, config.nr_branch_model, args.dataset_name, args.model_name)
	
	#print("Finish Confs")
	logging.debug("Finish Confs")

	test_loader = load_dataset(args, dataset_path, save_indices_path)
	inferenceTimeExperiment(test_loader, p_tar_list, nr_branch_edge, logPath)





if (__name__ == "__main__"):
	# Input Arguments. Hyperparameters configuration
	parser = argparse.ArgumentParser(description="Evaluating early-exits DNNs perfomance")

	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech256", "cifar10", "cifar100"], 
		help='Dataset name (default: Caltech-256)')

	parser.add_argument('--model_name', type=str, default=config.model_name, 
		choices=["mobilenet", "resnet18", "resnet152", "vgg16"], help='DNN model name (default: MobileNet)')

	parser.add_argument('--split_ratio', type=float, default=0.2, help='Split Ratio')

	args = parser.parse_args()


	main(args)
