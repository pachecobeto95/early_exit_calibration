import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.mobilenet import MobileNetV2_2
from networks.resnet import resnet18
from networks.vgg import vgg16_bn
import torchvision.models as models

def create_dir(history_path, model_path):
	if not os.path.exists(model_path):
		os.makedirs(model_path)
		os.makedirs(history_path)

def create_dir_temperature(temp_dir_path):
	if (not os.path.exists(temp_dir_path)):
		os.makedirs(temp_dir_path)

def verify_stop_condition(count, epoch, args):
	stop_condition = count <= args.patience if(args.pretrained) else epoch <= args.n_epochs
	return stop_condition


def get_model_arch(pretrained, model_name, n_classes, device):
	if (pretrained):
		mobilenet = models.mobilenet_v2()
		mobilenet.classifier[1] = nn.Linear(1280, n_classes)

		dict_model = {"mobilenet": mobilenet, "vgg16": models.vgg16_bn(), 
		"resnet18": models.resnet18(), "resnet152": models.resnet152()}
	else:
		dict_model = {"mobilenet": MobileNetV2_2(n_classes, device), "vgg16": vgg16_bn(n_classes), 
		"resnet18": resnet18(), "resnet152": models.resnet152()}

	return dict_model[model_name]
