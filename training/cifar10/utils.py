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
			nn.ReLU6(inplace = True),
			nn.AdaptiveAvgPool2d(1))

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



class InvertedResidual(nn.Module):
	def __init__(self, in_channels, out_channels, t=6, s=1):
		"""
		Initialization of inverted residual block
		:param in_channels: number of input channels
		:param out_channels: number of output channels
		:param t: the expansion factor of block
		:param s: stride of the first convolution
		"""
		super(InvertedResidual, self).__init__()

		self.in_ = in_channels
		self.out_ = out_channels
		self.t = t
		self.s = s
		self.inverted_residual_block()

	def inverted_residual_block(self):

		block = []
		block.append(nn.Conv2d(self.in_, self.in_*self.t, 1, bias=False))
		block.append(nn.BatchNorm2d(self.in_*self.t))
		block.append(nn.ReLU6())

		# conv 3*3 depthwise
		block.append(nn.Conv2d(self.in_*self.t, self.in_*self.t, 3,
			stride=self.s, padding=1, groups=self.in_*self.t, bias=False))
		block.append(nn.BatchNorm2d(self.in_*self.t))
		block.append(nn.ReLU6())

		# conv 1*1 linear
		block.append(nn.Conv2d(self.in_*self.t, self.out_, 1, bias=False))
		block.append(nn.BatchNorm2d(self.out_))

		self.block = nn.Sequential(*block)

		# if use conv residual connection
		if self.in_ != self.out_ and self.s != 2:
			self.res_conv = nn.Sequential(nn.Conv2d(self.in_, self.out_, 1, bias=False),
				nn.BatchNorm2d(self.out_))
		else:
			self.res_conv = None

	def forward(self, x):
		if self.s == 1:
			if self.res_conv is None:
				out = x + self.block(x)
			else:
				out = self.res_conv(x) + self.block(x)
		else:
			out = self.block(x)

		return out


def get_inverted_residual_block_arr(in_, out_, t=6, s=1, n=1):
	block = []
	block.append(InvertedResidual(in_, out_, t, s=s))
	for i in range(n-1):
		block.append(InvertedResidual(out_, out_, t, 1))
	return block

class MobileNetV2_2(nn.Module):
	def __init__(self, n_classes, device, alpha = 1):
		super(MobileNetV2_2, self).__init__()

		t = [1, 1, 6, 6, 6, 6, 6, 6]  # expansion factor t
		s = [1, 1, 1, 2, 2, 1, 2, 1, 1]  # stride of each conv stage
		n = [1, 1, 2, 3, 4, 3, 3, 1, 1]  # number of repeat time
		c = [32, 16, 24, 32, 64, 96, 160, 320, 1280]  # output channel of each conv stage
		down_sample_rate = 32  # product of strides above
		dropout_prob = 0.2

		block = []

		block.append(nn.Sequential(nn.Conv2d(3, c[0], 3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(c[0]),
			nn.Dropout2d(dropout_prob, inplace=True),
			nn.ReLU6()))

		for i in range(7):
			block.extend(get_inverted_residual_block_arr(c[i], c[i+1],
				t=t[i+1], s=s[i+1],
				n=n[i+1]))


		block.append(nn.Sequential(nn.AvgPool2d(image_size//down_sample_rate),
			nn.Dropout2d(dropout_prob, inplace=True),
			nn.Conv2d(c[-1], n_classes, 1, bias=True)))

		self.network = nn.Sequential(*block).to(device)

		# initialize
		self.initialize()

	def initialize(self):
		"""Initializes the model parameters"""
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		return self.network(x)



