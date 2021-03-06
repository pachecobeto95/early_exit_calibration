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
from torch import Tensor
import functools, os
from tqdm import tqdm
from networks.mobilenet import MobileNetV2_2
from utils import create_dir
from load_dataset import loadCifar10, loadCifar100
import argparse, ssl
from torchvision.datasets import CIFAR10, CIFAR100
from pthflops import count_ops
from ptflops import get_model_complexity_info


class EarlyExitBlock(nn.Module):
  """
  This EarlyExitBlock allows the model to terminate early when it is confident for classification.
  """
  def __init__(self, input_shape, n_classes, exit_type, device):
    super(EarlyExitBlock, self).__init__()
    self.input_shape = input_shape

    _, channel, width, height = input_shape
    self.expansion = width * height if exit_type == 'plain' else 1

    self.layers = nn.ModuleList()

    if (exit_type == 'bnpool'):
      self.layers.append(nn.BatchNorm2d(channel))

    if (exit_type != 'plain'):
      self.layers.append(nn.AdaptiveAvgPool2d(1))
    
    #This line defines the data shape that fully-connected layer receives.
    current_channel, current_width, current_height = self.get_current_data_shape()

    self.layers = self.layers.to(device)

    #This line builds the fully-connected layer
    self.classifier = nn.Sequential(nn.Linear(current_channel*current_width*current_height, n_classes)).to(device)

    self.softmax_layer = nn.Softmax(dim=1)


  def get_current_data_shape(self):
    _, channel, width, height = self.input_shape
    temp_layers = nn.Sequential(*self.layers)

    input_tensor = torch.rand(1, channel, width, height)
    _, output_channel, output_width, output_height = temp_layers(input_tensor).shape
    return output_channel, output_width, output_height
        
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    x = x.view(x.size(0), -1)
    output = self.classifier(x)
    return output


class BaseBlock(nn.Module):
    alpha = 1

    def __init__(self, input_channel, output_channel, t = 6, downsample = False):
        """
            t:  expansion factor, t*input_channel is channel of expansion layer
            alpha:  width multiplier, to get thinner models
            rho:    resolution multiplier, to get reduced representation
        """ 
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
    def __init__(self, output_size, alpha = 1):
        super(MobileNetV2, self).__init__()
        self.output_size = output_size

        # first conv layer 
        self.conv0 = nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(int(32*alpha))

        # build bottlenecks
        BaseBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
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
            BaseBlock(160, 320, downsample = False))

        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(int(320*alpha), 1280, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, output_size)

        # weights init
        self.weights_init()


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs):

        # first conv layer
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace = True)
        # assert x.shape[1:] == torch.Size([32, 32, 32])

        # bottlenecks
        x = self.bottlenecks(x)
        # assert x.shape[1:] == torch.Size([320, 8, 8])

        # last conv layer
        x = F.relu6(self.bn1(self.conv1(x)), inplace = True)
        # assert x.shape[1:] == torch.Size([1280,8,8])

        # global pooling and fc (in place of conv 1x1 in paper)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x



class Early_Exit_DNN(nn.Module):
  def __init__(self, model_name: str, n_classes: int, 
               n_branches: int, input_shape:tuple, exit_type: str, device, distribution="linear"):
    super(Early_Exit_DNN, self).__init__()

    """
    This classes builds an early-exit DNNs architectures
    Args:

    model_name: model name 
    n_classes: number of classes in a classification problem, according to the dataset
    pretrained: 
    n_branches: number of branches (early exits) inserted into middle layers
    input_shape: shape of the input image
    exit_type: type of the exits
    distribution: distribution method of the early exit blocks.
    device: indicates if the model will processed in the cpu or in gpu
    
    Note: the term "backbone model" refers to a regular DNN model, considering no early exits.

    """
    self.model_name = model_name
    self.n_classes = n_classes
    self.n_branches = n_branches
    self.input_shape = input_shape
    self.exit_type = exit_type
    self.distribution = distribution
    self.device = device
    self.channel, self.width, self.height = input_shape

    build_early_exit_dnn = self.select_dnn_architecture_model()
    build_early_exit_dnn()

  def select_dnn_architecture_model(self):
  
    architecture_dnn_model_dict = {"mobilenet": self.early_exit_MobileNet}

    return architecture_dnn_model_dict.get(self.model_name, self.invalid_model)
    
  def linear_distribution(self, i):
    """
    This method defines the Flops to insert an early exits, according to a linear distribution.
    """
    flop_margin = 1.0 / (self.n_branches+1)
    return self.total_flops * flop_margin * (i+1)

  def verifies_nr_exits(self, backbone_model):
    """
    This method verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    """
    
    total_layers = len(list(backbone_model.children()))
    if (self.n_branches >= total_layers):
      raise Exception("The number of early exits is greater than number of layers in the DNN backbone model.")

  def countFlops(self, model):
    """
    This method counts the numper of Flops in a given full DNN model or intermediate DNN model.
    """
    inputs = torch.rand(1, self.channel, self.width, self.height).to(self.device)
    flops, all_data = count_ops(model, inputs, print_readable=False, verbose=False)
    return flops

  def where_insert_early_exits(self):
    """
    This method defines where insert the early exits, according to the dsitribution method selected.
    Args:

    total_flops: Flops of the backbone (full) DNN model.
    """
    threshold_flop_list = []

    for i in range(self.n_branches):
      threshold_flop_list.append(self.linear_distribution(i))

    return threshold_flop_list

  def invalid_model(self):
    raise Exception("This DNN model has not implemented yet.")

  def is_suitable_for_exit(self):
    """
    This method answers the following question. Is the position to place an early exit?
    """
    intermediate_model = nn.Sequential(*(list(self.stages)+list(self.layers)))
    x = torch.rand(1, self.channel, self.width, self.height).to(self.device)
    current_flop, _ = count_ops(intermediate_model, x, verbose=False, print_readable=False)
    return self.stage_id < self.n_branches and current_flop >= self.threshold_flop_list[self.stage_id]

  def add_exit_block(self):
    """
    This method adds an early exit in the suitable position.
    """
    input_tensor = torch.rand(1, self.channel, self.width, self.height)

    self.stages.append(nn.Sequential(*self.layers))
    x = torch.rand(1, self.channel, self.width, self.height).to(self.device)
    feature_shape = nn.Sequential(*self.stages)(x).shape
    self.exits.append(EarlyExitBlock(feature_shape, self.n_classes, self.exit_type, self.device).to(self.device))#.to(self.device))
    self.layers = nn.ModuleList()
    self.stage_id += 1    

    
  def early_exit_MobileNet(self):
    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    n_blocks = 18
    last_channel = 1280
    
    # Loads the backbone model. In other words, Mobilenet architecture provided by Pytorch.
    backbone_model = MobileNetV2(self.n_classes).to(self.device)


    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    self.layers.append(backbone_model.conv0)
    self.layers.append(backbone_model.bn0)
    
    #self.layers.append(backbone_model.network[0])


    for i in range(17):
      self.layers.append(backbone_model.bottlenecks[i])
      if (self.is_suitable_for_exit()):
        self.add_exit_block()

    self.layers.append(backbone_model.conv1) 
    self.layers.append(backbone_model.bn1) 

    self.stages.append(nn.Sequential(*self.layers))
    self.classifier = backbone_model.fc
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    
    output_list, conf_list, class_list  = [], [], []

    for i, exitBlock in enumerate(self.exits):
      
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      output_list.append(output_branch)

      #Confidence is the maximum probability of belongs one of the predefined classes and inference_class is the argmax
      conf, infered_class = torch.max(self.softmax(output_branch), 1)
      conf_list.append(conf)
      class_list.append(infered_class)

    #print(x.shape)
    x = self.stages[-1](x)
    #print(x.shape)
    x = F.adaptive_avg_pool2d(x, 1)
    #print(x.shape)
    x = torch.flatten(x, 1)
    #print(x.shape)
    #sys.exit()
    output = self.classifier(x)
    infered_conf, infered_class = torch.max(self.softmax(output), 1)
    output_list.append(output)
    conf_list.append(infered_conf)
    class_list.append(infered_class)

    return output_list, conf_list, class_list




def loadCifar10(batch_size, input_size, crop_size, split_rate, seed=42):
	ssl._create_default_https_context = ssl._create_unverified_context

	np.random.seed(seed)
	torch.manual_seed(seed)

	mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

	transform_train = transforms.Compose([
		transforms.Resize(input_size),
		transforms.RandomCrop(crop_size, padding = 4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)])
    
	transform_test = transforms.Compose([
		transforms.Resize(input_size),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)])

	trainset = CIFAR10(".", transform=transform_train, train=True, download=True)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

	testset = CIFAR10(".", transform=transform_test, train=False, download=True)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle = False, num_workers=4)

	return train_loader, testloader

def trainEvalEarlyExit(model, dataLoader, criterion, optimizer, n_branches, epoch, device, loss_weights, train):
	if(train):
		model.train()
	else:
		model.eval()

	mode = "train" if(train) else "val"
	running_loss = []
	n_exits = n_branches + 1
	acc_dict = {i: [] for i in range(1, (n_exits)+1)}


	for (data, target) in tqdm(dataLoader):
		data, target = data.to(device), target.to(device)

		if (train):
			optimizer.zero_grad()
			output_list, conf_list, class_list = model(data)
			loss = 0
			for j, (output, inf_class, weight) in enumerate(zip(output_list, class_list, loss_weights), 1):
				loss = loss + weight*criterion(output, target)
				acc_dict[j].append(100*inf_class.eq(target.view_as(inf_class)).sum().item()/target.size(0))

			loss.backward()
			optimizer.step()
		
		else:
			with torch.no_grad():
				output_list, conf_list, class_list = model(data)
				loss = 0
				for j, (output, inf_class, weight) in enumerate(zip(output_list, class_list, loss_weights), 1):
					loss += weight*criterion(output, target)
					acc_dict[j].append(100*inf_class.eq(target.view_as(inf_class)).sum().item()/target.size(0))


		running_loss.append(float(loss.item()))

		# clear variables
		del data, target, output_list, conf_list, class_list
		torch.cuda.empty_cache()

	loss = round(np.average(running_loss), 4)
	print("Epoch: %s"%(epoch))
	print("%s Loss: %s"%(mode, loss))

	result_dict = {"epoch":epoch, "%s_loss"%(mode): loss}
	for key, value in acc_dict.items():
		result_dict.update({"%s_acc_branch_%s"%(mode, key): round(np.average(acc_dict[key]), 4)})    
		print("%s Acc Branch %s: %s"%(mode, key, result_dict["%s_acc_branch_%s"%(mode, key)]))
  
	return result_dict


if (__name__ == "__main__"):
	parser = argparse.ArgumentParser(description='Training the Early-Exit MobileNetV2.')

	parser.add_argument("--lr", type=float, default=0.045, help='Learning Rate (default: 0.045)')
	parser.add_argument('--weight_decay', type=float, default= 0.00004, help='Weight Decay (default: 0.00004)')
	parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (default: 0.9)')
	parser.add_argument('--lr_decay', type=float, default=0.98, help='Learning Rate Decay (default: 0.98)')
	parser.add_argument('--batch_size', type=int, default=128, help='Batch Size (default: 512)')
	parser.add_argument('--seed', type=int, default=42, help='Seed (default: 42)')
	parser.add_argument('--split_rate', type=float, default=0.2, help='Split rate of the dataset (default: 0.2)')
	parser.add_argument('--patience', type=int, default=10, help='Patience (default: 10)')
	parser.add_argument('--n_epochs', type=int, default=300, help='Number of epochs (default: 300)')
	parser.add_argument('--dataset_name', type=str, default="cifar10", 
		choices=["cifar10", "cifar100"], help='Pretrained (default: True)')
	parser.add_argument('--lr_scheduler', type=str, default="stepRL", 
		choices=["stepRL", "plateau", "cossine"], help='Learning Rate Scheduler (default: stepRL)')
	parser.add_argument('--n_branches', type=int, default=5, help='Number of side branches (default: 5)')
	parser.add_argument('--distribution', type=str, default="linear", help='Distribution of Branches (default: 1)')
	parser.add_argument('--exit_type', type=str, default="bnpool", 
		choices=["bnpool", "plain"], help='Exit Block Type (default: bnpool)')
	parser.add_argument('--loss_weight_type', type=str, default="crescent", 
		choices=["crescent", "decrescent", "equal"], help='Loss Weight (default: decrescent)')

	args = parser.parse_args()

	root_path = os.path.dirname(__file__)
	n_classes = 10 if(args.dataset_name == "cifar10") else 100
	input_size, crop_size = 32, 32
	input_shape = (3, input_size, input_size)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	count, epoch = 0, 0
	best_val_loss = np.inf
	n_exits = args.n_branches + 1
	model_name = "mobilenet"

	criterion = nn.CrossEntropyLoss()
	loss_dict = {"crescent": np.linspace(0.15, 1, n_exits), "decrescent": np.linspace(1, 0.15, n_exits), 
	"equal": np.ones(n_exits)}

	loss_weights = loss_dict[args.loss_weight_type]

	model = Early_Exit_DNN(model_name, n_classes, args.n_branches, input_shape, args.exit_type, device, distribution=args.distribution)
	model = model.to(device)

	train_loader, test_loader = loadCifar10(args.batch_size, input_size, crop_size, args.split_rate, seed=args.seed)

	optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay,
		nesterov=True)

	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay, patience=int(args.patience/2), verbose=True)

	while (count <= args.patience):
		epoch += 1
		print("Current Epoch: %s"%(epoch))

		result = {}
		result_train = trainEvalEarlyExit(model, train_loader, criterion, optimizer, args.n_branches, 
			epoch, device, loss_weights, train=True)
		
		result_val = trainEvalEarlyExit(model, test_loader, criterion, optimizer, args.n_branches, 
			epoch, device, loss_weights, train=False)
		
		#scheduler.step()
		result.update(result_train), result.update(result_val) 

		#df = df.append(pd.Series(result), ignore_index=True)
		#df.to_csv(history_path)

		if (result["val_loss"] < best_val_loss):
			best_val_loss = result["val_loss"]
			count = 0
			save_dict = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
			"epoch": epoch, "val_loss": result["val_loss"]}
    
			torch.save(save_dict, "./mobilenet_2.pth")

		else:
			print("Current Patience: %s"%(count))
			count += 1

	print("Stop! Patience is finished")
	#trainEvalEarlyExit(early_exit_dnn, test_loader, criterion, optimizer, args.n_branches, 
	#	epoch, device, loss_weights, train=False)











