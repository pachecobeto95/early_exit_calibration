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

result_path =  os.path.join(save_path, "results", "calib_overall_exp_data_%s_alert.csv"%(model_id))

saveTempOverallPath = os.path.join(root_save_path, "appEdge", "api", "services", "models",
  dataset_name, model_name, "temperature", "temp_overall_id_%s.csv"%(model_id))


dataset = LoadDataset(input_dim, batch_size_train, batch_size_test, model_id)
train_loader, val_loader, test_loader = dataset.caltech_256(dataset_path, split_ratio, dataset_name, save_indices_path)

early_exit_dnn = Early_Exit_DNN(model_name, n_classes, pretrained, n_branches, input_shape, exit_type, device, distribution=distribution)
early_exit_dnn = early_exit_dnn.to(device)



p_tar_list = [0.9]

for p_tar in p_tar_list:
	scaled_model = ModelOverallCalibration(early_exit_dnn, device, model_path)
	scaled_model.set_temperature(val_loader, p_tar)
