import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.mobilenet import MobileNetV2_2
from networks.resnet import resnet18, resnet152
from networks.vgg import vgg16_bn
import torchvision.models as models
import pandas as pd
import numpy as np

def create_dirs(history_path, model_path):
	if not os.path.exists(model_path):
		os.makedirs(model_path)
		os.makedirs(history_path)

def create_dir(dir_path):
	if (not os.path.exists(dir_path)):
		os.makedirs(dir_path)

def verify_stop_condition(count, epoch, args):
	stop_condition = count <= args.patience if(args.pretrained) else epoch <= args.n_epochs
	return stop_condition


def save_calibration_main_results(result, save_path):
	df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
	df.to_csv(save_path)


def get_model_arch(pretrained, model_name, n_classes, device):
	if (pretrained):
		mobilenet = models.mobilenet_v2()
		mobilenet.classifier[1] = nn.Linear(1280, n_classes)

		vgg16 = models.vgg16_bn()
		vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, n_classes)

		model_resnet18 = models.resnet18()
		model_resnet18.fc = nn.Linear(model_resnet18.fc.in_features, n_classes)

		model_resnet152 = models.resnet18()
		model_resnet152.fc = nn.Linear(model_resnet152.fc.in_features, n_classes)

		dict_model = {"mobilenet": mobilenet, "vgg16": vgg16, 
		"resnet18": model_resnet18, "resnet152": model_resnet152}
	else:
		dict_model = {"mobilenet": MobileNetV2_2(n_classes, device), "vgg16": vgg16_bn(n_classes), 
		"resnet18": resnet18(n_classes), "resnet152": resnet152(n_classes)}

	return dict_model[model_name]
