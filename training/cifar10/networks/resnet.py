import torchvision
import os, sys, time, math
from torchvision import transforms, utils, datasets
from torch.utils.data import random_split
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import torchvision.models as models
import functools

import torch.nn as nn
import torch

class BasicBlock(nn.Module):
	"""Basic Block for resnet 18 and resnet 34
	"""

	#BasicBlock and BottleNeck block
	#have different output size
	#we use class attribute expansion
	#to distinct
	expansion = 1

	def __init__(self, in_channels, out_channels, stride=1):
		super(BasicBlock, self).__init__()

		#residual function
		self.residual_function = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels * BasicBlock.expansion))

		#shortcut
		self.shortcut = nn.Sequential()

		#the shortcut output dimension is not the same with residual function
		#use 1*1 convolution to match the dimension
		if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels * BasicBlock.expansion)
				)

	def forward(self, x):
		return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Resnet(nn.Module):
	def __init__(self,  block, num_block, n_classes):
		super(Resnet, self).__init__()

		self.in_channels = 64

		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.layer1 = self._make_layer(block, 64, num_block[0], 1)
		self.layer2 = self._make_layer(block, 128, num_block[1], 2)
		self.layer3 = self._make_layer(block, 256, num_block[2], 2)
		self.layer4 = self._make_layer(block, 512, num_block[3], 2)
		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, n_classes)

	def _make_layer(self, block, out_channels, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_channels, out_channels, stride))
			self.in_channels = out_channels * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		output = self.features(x)
		output = self.avg_pool(output)
		output = output.view(output.size(0), -1)
		output = self.fc(output)
		return output

def resnet18(n_classes):
	""" return a ResNet 18 object"""
	return Resnet(BasicBlock, [2, 2, 2, 2], n_classes)

