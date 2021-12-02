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
from pthflops import count_ops
from torch import Tensor
from typing import Callable, Any, Optional, List, Type, Union
import torch.nn.init as init
import functools
from tqdm import tqdm
from load_dataset import LoadDataset
from early_exit_dnn import Early_Exit_DNN
from calibration_early_exit_dnn import ModelOverallCalibration


torch.multiprocessing.set_sharing_strategy('file_system')


def evalEarlyExitInference(model, n_branches, test_loader, p_tar, device):
	df_result = pd.DataFrame()

	n_exits = n_branches + 1
	conf_branches_list, infered_class_branches_list, target_list = [], [], []
	correct_list, exit_branch_list, id_list = [], [], []

	model.eval()

	with torch.no_grad():
		for i, (data, target) in tqdm(enumerate(test_loader, 1)):
      
			data, target = data.to(device), target.to(device)

			conf_branches, infered_class_branches = model.forwardOverall(data)


			conf_branches_list.append([conf.item() for conf in conf_branches])
			infered_class_branches_list.append([inf_class.item() for inf_class in infered_class_branches])    
			correct_list.append([infered_class_branches[i].eq(target.view_as(infered_class_branches[i])).sum().item() for i in range(n_exits)])
			id_list.append(i)
			target_list.append(target.item())

			del data, target
			torch.cuda.empty_cache()

	conf_branches_list = np.array(conf_branches_list)
	infered_class_branches_list = np.array(infered_class_branches_list)
	correct_list = np.array(correct_list)
  
	print("Acc:")
	print([sum(correct_list[:, i])/len(correct_list[:, i]) for i in range(n_exits)])

	results = {"p_tar": [round(p_tar, 2)]*len(target_list), "target": target_list, "id": id_list}
	for i in range(n_exits):
		results.update({"conf_branch_%s"%(i+1): conf_branches_list[:, i],
			"infered_class_branches_%s"%(i+1): infered_class_branches_list[:, i],
			"correct_branch_%s"%(i+1): correct_list[:, i]})

	return results


def saveResults(result, save_path):
	df_result = pd.read_csv(save_path) if (os.path.exists(save_path)) else pd.DataFrame()
	df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
	df_result = df_result.append(df)
	df_result.to_csv(save_path)


def expOverallCalibration(model, val_loader, test_loader, device, p_tar_list, model_path, result_path, saveTempPath):
	for p_tar in p_tar_list:

		print("P_tar: %s"%(round(p_tar, 2)))
		scaled_model = ModelOverallCalibration(model, device, model_path, saveTempPath)
		scaled_model.set_temperature(val_loader, p_tar)

		result = evalEarlyExitInference(model, model.n_branches, test_loader, p_tar, device)
		saveResults(result, result_path)

 

input_dim = 224
batch_size_train = 64
batch_size_test = 1
model_id = 1
split_ratio = 0.2
n_classes = 258
pretrained = False
n_branches = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_shape = (3, input_dim, input_dim)

distribution = "linear"
exit_type = "bnpool"
dataset_name = "caltech256"
model_name = "resnet152"


root_save_path = "."

save_indices_path = os.path.join(".", "caltech256", "indices")


dataset_path = os.path.join(root_save_path, "datasets", dataset_name, "256_ObjectCategories")

model_path = os.path.join(root_save_path, "appEdge", "api", "services", "models",
	dataset_name, model_name, "models", 
	"ee_%s_branches_%s_id_%s.pth"%(model_name, n_branches, model_id))

save_path = os.path.join(root_save_path, "appEdge", "api", "services", "models", dataset_name, model_name)

result_path =  os.path.join(save_path, "results", "calib_overall_exp_data_%s.csv"%(model_id))

saveTempOverallPath = os.path.join(root_save_path, "appEdge", "api", "services", "models",
  dataset_name, model_name, "temperature", "temp_overall_id_%s.csv"%(model_id))


dataset = LoadDataset(input_dim, batch_size_train, batch_size_test, model_id)
train_loader, val_loader, test_loader = dataset.caltech_256(dataset_path, split_ratio, dataset_name, save_indices_path)

early_exit_dnn = Early_Exit_DNN(model_name, n_classes, pretrained, n_branches, input_shape, exit_type, device, distribution=distribution)
early_exit_dnn = early_exit_dnn.to(device)



p_tar_list = [0.8, 0.85, 0.9, 0.95]

expOverallCalibration(early_exit_dnn, val_loader, test_loader, device, p_tar_list, model_path, result_path, saveTempOverallPath)



