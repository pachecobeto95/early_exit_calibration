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

		vgg16 = models.vgg16_bn()
		vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, n_classes)

		model_resnet18 = models.resnet18()
		model_resnet18.fc = nn.Linear(model_resnet18.fc.in_features, n_classes)

		model_resnet152 = models.resnet18()
		model_resnet152.fc = nn.Linear(model_resnet152.fc.in_features, n_classes)

		dict_model = {"mobilenet": mobilenet, "vgg16": vgg16(), 
		"resnet18": model_resnet18, "resnet152": model_resnet152()}
	else:
		dict_model = {"mobilenet": MobileNetV2_2(n_classes, device), "vgg16": vgg16_bn(n_classes), 
		"resnet18": resnet18(n_classes), "resnet152": model_resnet152()}

	return dict_model[model_name]


class WarmUpLR(_LRScheduler):
	"""warmup_training learning rate scheduler
	Args:
	optimizer: optimzier(e.g. SGD)
	total_iters: totoal_iters of warmup phase
	"""
	def __init__(self, optimizer, total_iters, last_epoch=-1):
		self.total_iters = total_iters
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]




