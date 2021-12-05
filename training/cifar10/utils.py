import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_dir(history_path, model_path):
	if not os.path.exists(model_path):
		
		os.makedirs(model_path)
		os.makedirs(history_path)

class BaseBlock(nn.Module):
	alpha = 1

	def __init__(self, input_channel, output_channel, t = 6, downsample = False):

		super(BaseBlock, self).__init__()
		self.stride = 2 if downsample else 1
		self.downsample = downsample
		self.shortcut = (not downsample) and (input_channel == output_channel) 

		# apply alpha
		input_channel = int(self.alpha * input_channel)
		output_channel = int(self.alpha * output_channel)

		# for main path:
		c  = t * input_channel
		# 1x1   point wise conv
		self.conv1 = nn.Conv2d(input_channel, c, kernel_size = 1, bias = False)
		self.bn1 = nn.BatchNorm2d(c)
		# 3x3   depth wise conv
		self.conv2 = nn.Conv2d(c, c, kernel_size = 3, stride = self.stride, padding = 1, groups = c, bias = False)
		self.bn2 = nn.BatchNorm2d(c)
		# 1x1   point wise conv
		self.conv3 = nn.Conv2d(c, output_channel, kernel_size = 1, bias = False)
		self.bn3 = nn.BatchNorm2d(output_channel)

def forward(self, inputs):
		# main path
		x = F.relu6(self.bn1(self.conv1(inputs)), inplace = True)
		x = F.relu6(self.bn2(self.conv2(x)), inplace = True)
		x = self.bn3(self.conv3(x))
		# shortcut path
		x = x + inputs if self.shortcut else x
		return x


class MobileNetV2(nn.Module):
	def __init__(self, n_classes, alpha = 1):
		super(MobileNetV2, self).__init__()
		self.n_classes = n_classes

		# first conv layer 
		#self.conv0 = nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False)
		#self.bn0 = nn.BatchNorm2d(int(32*alpha))

		# build bottlenecks
		BaseBlock.alpha = alpha
		self.bottlenecks = nn.Sequential(
			nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False),
			nn.BatchNorm2d(int(32*alpha)),
			nn.ReLU6(inplace = True),
			BaseBlock(32, 16, t = 1, downsample = False),
			BaseBlock(16, 24, downsample = False),
			BaseBlock(24, 24),
			BaseBlock(24, 32, downsample = False),
			BaseBlock(32, 32),
			BaseBlock(32, 32),
			BaseBlock(32, 64, downsample = True),
			BaseBlock(64, 64),
			BaseBlock(64, 64),
			BaseBlock(64, 64),
			BaseBlock(64, 96, downsample = False),
			BaseBlock(96, 96),
			BaseBlock(96, 96),
			BaseBlock(96, 160, downsample = True),
			BaseBlock(160, 160),
			BaseBlock(160, 160),
			BaseBlock(160, 320, downsample = False),
			nn.Conv2d(int(320*alpha), 1280, kernel_size = 1, bias = False),
			nn.BatchNorm2d(1280),
			)

		self.classifier = nn.Linear(1280, n_classes)

		# weights init
		self.weights_init()


	def weights_init(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))

			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
			#m.bias.data.zero_()


	def forward(self, x):

		x = self.bottlenecks(x)
		x = x.view(x.shape[0], -1)
		x = self.classifier(x)

		return x
