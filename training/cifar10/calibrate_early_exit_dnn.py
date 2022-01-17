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
from utils import create_dir, save_all_results_ee_calibration, testEarlyExitInference
from load_dataset import loadCifar10, loadCifar100
from calibration_dnn import MainModelCalibration, calibratingEEModels
from train import testMainModel
from early_exit_dnns import Early_Exit_DNN


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
	parser.add_argument('--batch_size_train', type=int, default=128, help='Batch Size (default: 128)')
	parser.add_argument('--batch_size_test', type=int, default=1, help='Batch Size (default: 1)')	
	parser.add_argument('--split_rate', type=float, default=0.1, help='Split rate of the dataset (default: 0.1)')
	parser.add_argument('--loss_weight_type', type=str, default="crescent", 
		choices=["crescent", "decrescent", "equal"], help='Loss Weight (default: decrescent)')
	parser.add_argument('--backbone_pretrained', dest='backbone_pretrained', 
		action='store_false', default=True, help='Pretrained (default:True)')
	parser.add_argument('--n_branches', type=int, default=5, help='Number of side branches (default: 5)')
	parser.add_argument('--distribution', type=str, default="linear", help='Distribution of Branches (default: 1)')
	parser.add_argument('--exit_type', type=str, default="bnpool", 
		choices=["bnpool", "plain"], help='Exit Block Type (default: bnpool)')



	args = parser.parse_args()
	root_path = os.path.dirname(__file__)

	dataset_path = os.path.join(root_path, "dataset")
	network_dir_path = os.path.join(root_path, args.model_name) 
	model_dir_path = os.path.join(network_dir_path, "models")
	history_dir_path = os.path.join(network_dir_path, "history")
	temp_dir_path = os.path.join(network_dir_path, "temperature")
	result_dir_path = os.path.join(network_dir_path, "results")
	indices_dir_path = os.path.join(root_path, "indices")
	
	mode = "ft" if(args.pretrained) else "scratch"
	input_size = 224 if (args.pretrained) else 32
	crop_size = 224 if (args.pretrained) else 32
	n_classes = 10 if (args.dataset_name == "cifar10") else 100
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	input_shape = (3, input_size, input_size)

	threshold_list = [0.8, 0.9]

	backbone_model_path = os.path.join(model_dir_path, "%s_main_%s_id_%s_%s.pth"%(args.model_name, args.dataset_name, args.model_id, mode))
	model_path = os.path.join(model_dir_path, "b_%s_early_exit_%s_id_%s_%s_%s.pth"%(args.model_name, args.dataset_name, args.model_id, mode, args.loss_weight_type))
	save_overall_temp_path = os.path.join(temp_dir_path, "overall_temperature_%s_early_exit_%s_id_%s_%s.csv"%(args.model_name, args.dataset_name, args.model_id, mode))
	save_branches_temp_path = os.path.join(temp_dir_path, "branches_temperature_%s_early_exit_%s_id_%s_%s.csv"%(args.model_name, args.dataset_name, args.model_id, mode))
	
	result_no_calib_path = os.path.join(result_dir_path, "no_calib_results_%s_early_exit_%s_id_%s_%s.csv"%(args.model_name, args.dataset_name, args.model_id, mode))
	overall_result_calib_path = os.path.join(result_dir_path, "overall_calib_results_%s_early_exit_%s_id_%s_%s.csv"%(args.model_name, args.dataset_name, args.model_id, mode))
	branches_result_calib_path = os.path.join(result_dir_path, "branches_calib_results_%s_early_exit_%s_id_%s_%s.csv"%(args.model_name, args.dataset_name, args.model_id, mode))
	all_result_calib_path = os.path.join(result_dir_path, "all_samples_calib_results_%s_early_exit_%s_id_%s_%s.csv"%(args.model_name, args.dataset_name, args.model_id, mode))
	
	create_dir(temp_dir_path)
	create_dir(result_dir_path)

	early_exit_dnn = Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.backbone_pretrained, 
		backbone_model_path, args.n_branches, input_shape, args.exit_type, device, distribution=args.distribution)
	early_exit_dnn = early_exit_dnn.to(device)

	print(model_path)
	early_exit_dnn.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])


	resultPathDict = {"no_calib": result_no_calib_path, 
	"overall_calib":overall_result_calib_path, "branches_calib":branches_result_calib_path, 
	"all_samples_calib": all_result_calib_path}

	temperaturePath = {"overall_calib": save_overall_temp_path, 
	"branches_calib": save_branches_temp_path, "all_samples_calib":save_branches_temp_path}

	if(args.dataset_name=="cifar10"):
		train_loader, val_loader, test_loader = loadCifar10(dataset_path, indices_dir_path, args.model_id, 
			args.batch_size_train, args.batch_size_test, input_size, crop_size, split_rate=args.split_rate, seed=args.seed)

	else:
		train_loader, val_loader, test_loader = loadCifar100(dataset_path, indices_dir_path, args.model_id, 
			args.batch_size_train, args.batch_size_test, input_size, crop_size, split_rate=args.split_rate, seed=args.seed)


	for threshold in threshold_list:
		print("Ptar: %s"%(threshold))
		no_calib_result = testEarlyExitInference(early_exit_dnn, early_exit_dnn.n_branches, test_loader, 
			threshold, device, model_type="no_calib")

		scaled_models_dict = calibratingEEModels(early_exit_dnn, val_loader, threshold, device, model_path, temperaturePath, args)


		overall_result = testEarlyExitInference(scaled_models_dict["calib_overall"], early_exit_dnn.n_branches, test_loader, 
			threshold, device, model_type="calib_overall")

		branches_result = testEarlyExitInference(scaled_models_dict["calib_branches"], early_exit_dnn.n_branches, 
			test_loader, threshold, device, model_type="calib_branches")

		all_samples_result = testEarlyExitInference(scaled_models_dict["calib_branches_all_samples"], 
			early_exit_dnn.n_branches, test_loader, threshold, device, model_type="calib_branches_all_samples")

		save_all_results_ee_calibration(no_calib_result, overall_result, branches_result, 
			all_samples_result, resultPathDict)
