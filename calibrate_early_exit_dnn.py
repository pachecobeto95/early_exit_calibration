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
torch.multiprocessing.set_sharing_strategy('file_system')

def create_save_dir(dataset_save_path, model_name, save_indices_path):
  if (not os.path.exists(dataset_save_path)):
    os.makedirs(dataset_save_path)
    os.makedirs(os.path.join(dataset_save_path, model_name))
    os.makedirs(save_indices_path)
  else:
    if (not os.path.exists(save_indices_path)):
      os.makedirs(save_indices_path)


class LoadDataset():
  def __init__(self, input_dim, batch_size_train, batch_size_test, model_id, seed=42):
    self.input_dim = input_dim
    self.batch_size_train = batch_size_train
    self.batch_size_test = batch_size_test
    self.seed = seed
    self.model_id = model_id

    #To normalize the input images data.
    mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std =  [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    # Note that we apply data augmentation in the training dataset.
    self.transformations_train = transforms.Compose([transforms.Resize((256, 256)),
                                                     transforms.CenterCrop((224, 224)), 
                                                     transforms.RandomChoice([
                                                                              transforms.ColorJitter(brightness=(0.80, 1.20)),
                                                                              transforms.RandomGrayscale(p = 0.25)]),
                                                     transforms.RandomHorizontalFlip(p = 0.25),
                                                     transforms.RandomRotation(25),
                                                     transforms.ToTensor(), 
                                                     transforms.Normalize(mean = mean, std = std),
                                                     ])

    # Note that we do not apply data augmentation in the test dataset.
    self.transformations_test = transforms.Compose([
                                                     transforms.Resize((256, 256)),
                                                     transforms.CenterCrop((224, 224)),
                                                     transforms.ToTensor(), 
                                                     transforms.Normalize(mean = mean, std = std),
                                                     ])

  def cifar_100(self, root_path, split_ratio, savePath_idx):
    """
    root_path: the path where the dataset is downloaded.
    split_ratio: validation dataset ratio
    """
    # This method loads Cifar-100 dataset
    root = "cifar_100"
    torch.manual_seed(self.seed)

    # This downloads the training and test Cifar-100 datasets and also applies transformation  in the data.
    train_set = datasets.CIFAR100(root=root_path, train=True, download=True, transform=self.transformations_train)
    test_set = datasets.CIFAR100(root=root_path, train=False, download=True, transform=self.transformations_test)

    classes_list = train_set.classes

    # This line defines the size of validation dataset.
    val_size = int(split_ratio*len(train_set))

    # This line defines the size of training dataset.
    train_size = int(len(train_set) - val_size)

    if (os.path.exists(os.path.join(savePath_idx, "training_idx_cifar100_id_%s.npy"%(self.model_id)))):
      train_idx = np.load(os.path.join(savePath_idx, "training_idx_cifar100_id_%s.npy"%(self.model_id)))
      val_idx = np.load(os.path.join(savePath_idx, "validation_idx_cifar100_id_%s.npy"%(self.model_id)))

    else:
      train_idx, val_idx = self.get_indices(train_set, split_ratio)
      #train_data = torch.utils.data.Subset(train_set, indices=train_idx)
      #val_data = torch.utils.data.Subset(train_set, indices=val_idx)

      np.save(os.path.join(savePath_idx, "training_idx_cifar100_id_%s.npy"%(self.model_id)), train_idx)
      np.save(os.path.join(savePath_idx, "validation_idx_cifar100_id_%s.npy"%(self.model_id)), val_idx)

    train_data = torch.utils.data.Subset(train_set, indices=train_idx)
    val_data = torch.utils.data.Subset(train_set, indices=val_idx)

    #This block creates data loaders for training, validation and test datasets.
    train_loader = DataLoader(train_data, self.batch_size_train, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, self.batch_size_test, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, self.batch_size_test, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
  
  def get_indices(self, dataset, split_ratio):
    nr_samples = len(dataset)
    indices = list(range(nr_samples))
    
    train_size = nr_samples - int(np.floor(split_ratio * nr_samples))

    np.random.shuffle(indices)

    train_idx, test_idx = indices[:train_size], indices[train_size:]

    return train_idx, test_idx

  def caltech_256(self, dataset_path, split_ratio, dataset_name, savePath_idx):
    # This method loads the Caltech-256 dataset.

    torch.manual_seed(self.seed)
    np.random.seed(seed=self.seed)

    # This block receives the dataset path and applies the transformation data. 
    train_set = datasets.ImageFolder(dataset_path, transform=self.transformations_train)
    self.train_set = train_set

    val_set = datasets.ImageFolder(dataset_path, transform=self.transformations_test)
    self.val_set = val_set
    
    test_set = datasets.ImageFolder(dataset_path, transform=self.transformations_test)

    if (os.path.exists(os.path.join(savePath_idx, "training_idx_%s_id_%s.npy"%(dataset_name, self.model_id)))):
      train_idx = np.load(os.path.join(savePath_idx, "training_idx_%s_id_%s.npy"%(dataset_name, self.model_id)))
      val_idx = np.load(os.path.join(savePath_idx, "validation_idx_%s_id_%s.npy"%(dataset_name, self.model_id)))
      self.val_idx = val_idx
      test_idx = np.load(os.path.join(savePath_idx, "test_idx_%s_id_%s.npy"%(dataset_name, self.model_id)), allow_pickle=True)
      test_idx = np.array(list(test_idx.tolist()))

    else:

      # This line get the indices of the samples which belong to the training dataset and test dataset. 
      train_idx, test_idx = self.get_indices(train_set, split_ratio)

      # This line mounts the training and test dataset, selecting the samples according indices. 
      train_data = torch.utils.data.Subset(train_set, indices=train_idx)
      ##essa linha parecia estar faltando. copiei da versão anterior##

      # This line gets the indices to split the train dataset into training dataset and validation dataset.
      train_idx, val_idx = self.get_indices(train_data, split_ratio)

      #np.save(os.path.join(savePath_idx, "training_idx_%s_id_%s.npy"%(dataset_name, self.model_id)), train_idx)
      #np.save(os.path.join(savePath_idx, "validation_idx_%s_id_%s.npy"%(dataset_name, self.model_id)), val_idx)

    # This line mounts the training and test dataset, selecting the samples according indices. 
    train_data = torch.utils.data.Subset(train_set, indices=train_idx)
    val_data = torch.utils.data.Subset(val_set, indices=val_idx)
    test_data = torch.utils.data.Subset(test_set, indices=test_idx)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size_train, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size_test, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size_test, num_workers=4)

    return train_loader, val_loader, val_loader 

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

    self.layers = self.layers#.to(device)

    #This line builds the fully-connected layer
    self.classifier = nn.Sequential(nn.Linear(current_channel*current_width*current_height, n_classes))#.to(device)

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
    #confidence = self.softmax_layer()
    return output

def conv1x1(in_planes, out_planes, stride=1):
  """1x1 convolution"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
  expansion: int = 1

  def __init__(self, inplanes: int, planes: int, stride: int = 1,
               downsample: Optional[nn.Module] = None, groups: int = 1, base_width: int = 16, 
               dilation: int = 1,
               norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
    super(BasicBlock, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    if groups != 1 or base_width != 64:
      raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
      raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv3x3(inplanes, planes)

    self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x: Tensor) -> Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    print(out.shape)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class ResNet(nn.Module):

  def __init__(self, block, layers, num_classes,zero_init_residual: bool = False, 
               groups: int = 1,
               width_per_group: int = 64,
               replace_stride_with_dilation: Optional[List[bool]] = None,
               norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
    super(ResNet, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    self._norm_layer = norm_layer

    self.inplanes = 16
    self.dilation = 1
    if replace_stride_with_dilation is None:
      # each element in the tuple indicates if we should replace
      # the 2x2 stride with a dilated convolution instead
      replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
      raise ValueError("replace_stride_with_dilation should be None "
      "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    
    self.groups = groups
    self.base_width = width_per_group

    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)    
    self.bn1 = norm_layer(16)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)    
    self.fc = nn.Linear(64, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, Bottleneck):
          nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        elif isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

  def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes * block.expansion, stride),
          norm_layer(planes * block.expansion),
          )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes, groups=self.groups,
                          base_width=self.base_width, dilation=self.dilation,
                          norm_layer=norm_layer))

    return nn.Sequential(*layers)

  def forward(self, x: Tensor) -> Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    x = F.avg_pool2d(x, x.size()[3])
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x

class DownSample(nn.Module):
  def __init__(self, downsample, feat):
    super().__init__()
    self.downsample = downsample
    self.feat = feat
  
  def forward(self, x):
    identity = self.downsample(self.feat)
    return x.add(identity)

class Early_Exit_DNN(nn.Module):
  def __init__(self, model_name: str, n_classes: int, 
               pretrained: bool, n_branches: int, input_shape:tuple, 
               exit_type: str, device, distribution="linear"):
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
    self.pretrained = pretrained
    self.n_branches = n_branches
    self.input_shape = input_shape
    self.exit_type = exit_type
    self.distribution = distribution
    self.device = device
    self.channel, self.width, self.height = input_shape

    build_early_exit_dnn = self.select_dnn_architecture_model()
    build_early_exit_dnn()

  def select_dnn_architecture_model(self):
    """
    This method selects the backbone to insert the early exits.
    """

    architecture_dnn_model_dict = {"alexnet": self.early_exit_alexnet,
                                   "mobilenet": self.early_exit_mobilenet,
                                   "resnet18": self.early_exit_resnet18,
                                   "resnet50": self.early_exit_resnet50_2,
                                   "vgg16": self.early_exit_vgg16, 
                                   "inceptionV3": self.early_exit_inceptionV3,
                                   "resnet152": self.early_exit_resnet152}

    return architecture_dnn_model_dict.get(self.model_name, self.invalid_model)

  def select_distribution_method(self):
    """
    This method selects the distribution method to insert early exits into the middle layers.
    """
    distribution_method_dict = {"linear":self.linear_distribution,
                                "pareto":self.paretto_distribution,
                                "fibonacci":self.fibo_distribution}
    return distribution_method_dict.get(self.distribution, self.invalid_distribution)
    
  def linear_distribution(self, i):
    """
    This method defines the Flops to insert an early exits, according to a linear distribution.
    """
    flop_margin = 1.0 / (self.n_branches+1)
    return self.total_flops * flop_margin * (i+1)

  def paretto_distribution(self, i):
    """
    This method defines the Flops to insert an early exits, according to a pareto distribution.
    """
    return self.total_flops * (1 - (0.8**(i+1)))

  def fibo_distribution(self, i):
    """
    This method defines the Flops to insert an early exits, according to a fibonacci distribution.
    """
    gold_rate = 1.61803398875
    return total_flops * (gold_rate**(i - self.num_ee))

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
    input = torch.rand(1, self.channel, self.width, self.height)#.to(self.device)
    flops, all_data = count_ops(model, input, print_readable=False, verbose=False)
    return flops

  def where_insert_early_exits(self):
    """
    This method defines where insert the early exits, according to the dsitribution method selected.
    Args:

    total_flops: Flops of the backbone (full) DNN model.
    """
    threshold_flop_list = []
    distribution_method = self.select_distribution_method()

    for i in range(self.n_branches):
      threshold_flop_list.append(distribution_method(i))

    return threshold_flop_list

  def invalid_model(self):
    raise Exception("This DNN model has not implemented yet.")
  def invalid_distribution(self):
    raise Exception("This early-exit distribution has not implemented yet.")

  def is_suitable_for_exit(self):
    """
    This method answers the following question. Is the position to place an early exit?
    """
    intermediate_model = nn.Sequential(*(list(self.stages)+list(self.layers)))
    x = torch.rand(1, 3, 224, 224)#.to(self.device)
    current_flop, _ = count_ops(intermediate_model, x, verbose=False, print_readable=False)
    return self.stage_id < self.n_branches and current_flop >= self.threshold_flop_list[self.stage_id]

  def add_exit_block(self):
    """
    This method adds an early exit in the suitable position.
    """
    input_tensor = torch.rand(1, self.channel, self.width, self.height)

    self.stages.append(nn.Sequential(*self.layers))
    x = torch.rand(1, 3, 224, 224)#.to(self.device)
    feature_shape = nn.Sequential(*self.stages)(x).shape
    self.exits.append(EarlyExitBlock(feature_shape, self.n_classes, self.exit_type, self.device))#.to(self.device))
    self.layers = nn.ModuleList()
    self.stage_id += 1    

  def set_device(self):
    """
    This method sets the device that will run the DNN model.
    """

    self.stages.to(self.device)
    self.exits.to(self.device)
    self.layers.to(self.device)
    self.classifier.to(self.device)

  def set_device_resnet50(self):
    self.stages.to(self.device)
    self.exits.to(self.device)
    self.layers.to(self.device)
    self.classifier.to(self.device)

  def early_exit_alexnet(self):
    """
    This method inserts early exits into a Alexnet model
    """

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    # Loads the backbone model. In other words, Alexnet architecture provided by Pytorch.
    backbone_model = models.alexnet(self.pretrained)

    # It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    self.verifies_nr_exit_alexnet(backbone_model.features)
    
    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    for layer in backbone_model.features:
      self.layers.append(layer)
      if (isinstance(layer, nn.ReLU)) and (self.is_suitable_for_exit()):
        self.add_exit_block()

    
    
    self.layers.append(nn.AdaptiveAvgPool2d(output_size=(6, 6)))
    self.stages.append(nn.Sequential(*self.layers))

    
    self.classifier = backbone_model.classifier
    self.classifier[6] = nn.Linear(in_features=4096, out_features=self.n_classes, bias=True)
    self.softmax = nn.Softmax(dim=1)
    self.set_device()

  def verifies_nr_exit_alexnet(self, backbone_model):
    """
    This method verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    In AlexNet, we consider a convolutional block composed by: Convolutional layer, ReLU and he Max-pooling layer.
    Hence, we consider that it makes no sense to insert side branches between these layers or only after the convolutional layer.
    """

    count_relu_layer = 0
    for layer in backbone_model:
      if (isinstance(layer, nn.ReLU)):
        count_relu_layer += 1

    if (count_relu_layer > self.n_branches):
      raise Exception("The number of early exits is greater than number of layers in the DNN backbone model.")

  def early_exit_resnet152(self):
    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    self.inplanes = 64

    n_blocks = 4

    backbone_model = models.resnet152(self.pretrained)

    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    self.threshold_flop_list = self.where_insert_early_exits()

    building_first_layer = ["conv1", "bn1", "relu", "maxpool"]
    for layer in building_first_layer:
      self.layers.append(getattr(backbone_model, layer))

    if (self.is_suitable_for_exit()):
      self.add_exit_block()

    for i in range(1, n_blocks+1):
      
      block_layer = getattr(backbone_model, "layer%s"%(i))

      for l in block_layer:
        self.layers.append(l)

        if (self.is_suitable_for_exit()):
          self.add_exit_block()
    
    self.layers.append(nn.AdaptiveAvgPool2d(1))
    self.classifier = nn.Sequential(nn.Linear(2048, self.n_classes))
    self.stages.append(nn.Sequential(*self.layers))
    self.softmax = nn.Softmax(dim=1)
    self.set_device()


  def early_exit_resnet18(self):
    """
    This method inserts early exits into a Resnet18 model
    """

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    self.inplanes = 64

    n_blocks = 4

    backbone_model = models.resnet18(self.pretrained)

    # It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    self.verifies_nr_exits(backbone_model)

    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    building_first_layer = ["conv1", "bn1", "relu", "maxpool"]
    for layer in building_first_layer:
      self.layers.append(getattr(backbone_model, layer))

    if (self.is_suitable_for_exit()):
      self.add_exit_block()

    for i in range(1, n_blocks+1):
      
      block_layer = getattr(backbone_model, "layer%s"%(i))

      for l in block_layer:
        self.layers.append(l)

        if (self.is_suitable_for_exit()):
          self.add_exit_block()
    
    self.layers.append(nn.AdaptiveAvgPool2d(1))
    self.classifier = nn.Sequential(nn.Linear(512, self.n_classes))
    self.stages.append(nn.Sequential(*self.layers))
    self.softmax = nn.Softmax(dim=1)
    self.set_device()


  def early_exit_resnet50_2(self):

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    backbone_model = models.resnet50(pretrained=True)

    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    x = torch.rand(1, 3, 224, 224)
    first_layers_list = ["conv1", "bn1", "relu", "maxpool"]
    for first_layer in first_layers_list:
      self.layers.append(getattr(backbone_model, first_layer))

    data = nn.Sequential(*self.layers)(x)

    n_layers = 4

    for n in range(1, n_layers+1):
      backbone_block = getattr(backbone_model, "layer%s"%(n))
      n_blocks = len(backbone_block)
      
      for j in range(n_blocks):
        bottleneck_layers = backbone_block[j]
        self.layers.append(bottleneck_layers)
        
        if (self.is_suitable_for_exit()):
          self.add_exit_block()

    self.layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    self.classifier = nn.Sequential(nn.Linear(2048, self.n_classes))
    self.stages.append(nn.Sequential(*self.layers))
    self.softmax = nn.Softmax(dim=1)


  def early_exit_resnet50(self):

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    backbone_model = models.resnet50(pretrained=True)

    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    x = torch.rand(1, 3, 224, 224)
    first_layers_list = ["conv1", "bn1", "relu", "maxpool"]
    for first_layer in first_layers_list:
      self.layers.append(getattr(backbone_model, first_layer))

    data = nn.Sequential(*self.layers)(x)

    bottleneck_list = ["conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "relu", "downsample"]

    bottleneck_short_list = bottleneck_list[:-1]
    n_layers = 4

    for n in range(1, n_layers+1):
      backbone_block = getattr(backbone_model, "layer%s"%(n))
      n_blocks = len(backbone_block)
      
      for j in range(n_blocks):
        bottleneck_layers = backbone_block[j]
        bottleneck_layers_list = bottleneck_list if (j==0) else bottleneck_short_list

        for layer in bottleneck_layers_list:
          temp_layer = getattr(bottleneck_layers, layer)
          if (layer == "downsample"):
            #pass
            self.layers.append(DownSample(temp_layer, data))
          else:
            self.layers.append(temp_layer)

          if (self.is_suitable_for_exit()):
            self.add_exit_block()
      
      data = backbone_block(data)

    self.layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    self.classifier = nn.Sequential(nn.Linear(2048, self.n_classes))
    self.stages.append(nn.Sequential(*self.layers))
    self.softmax = nn.Softmax(dim=1)
    #self.set_device_resnet50()


  def early_exit_vgg16(self):
    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0


    backbone_model = models.vgg16_bn(self.pretrained)
    backbone_model_features = backbone_model.features
    
    self.total_flops = self.countFlops(backbone_model)
    self.threshold_flop_list = self.where_insert_early_exits()

    for layer in backbone_model_features:
      self.layers.append(layer)
      if (self.is_suitable_for_exit()):
        self.add_exit_block()


    self.layers.append(backbone_model.avgpool)
    self.stages.append(nn.Sequential(*self.layers))
    self.classifier = backbone_model.classifier
    self.classifier[0] = nn.Linear(in_features=25088, out_features=4096)
    self.classifier[3] = nn.Linear(in_features=4096, out_features=4096)
    self.classifier[6] = nn.Linear(in_features=4096, out_features=self.n_classes)
    self.set_device()
    self.softmax = nn.Softmax(dim=1)


  def early_exit_inceptionV3(self):
    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    backbone_model = models.inception_v3(self.pretrained)
    self.total_flops = self.countFlops(backbone_model)
    self.threshold_flop_list = self.where_insert_early_exits()

    architecture_layer_dict = ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3",
                              "maxpool1", "Conv2d_3b_1x1", "Conv2d_4a_3x3", "maxpool2",
                              "Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d",
                              "Mixed_6e", "Mixed_7a", "Mixed_7b", "Mixed_7c", "avgpool", "dropout"]

    for block in architecture_layer_dict:
      layer_list.append(getattr(inception, block))
      if (self.is_suitable_for_exit()):
        self.add_exit_block()


    self.stages.append(nn.Sequential(*self.layers))
    self.classifier = backbone_model.fc
    self.set_device()
    self.softmax = nn.Softmax(dim=1)


  def early_exit_resnet56(self):

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    self.in_planes = 16
    n_layers = 3
    num_blocks =  [9, 9, 9]
    basic_block_list = ["conv1", "bn1", "relu", "conv2", "bn2"]

    backbone_model = ResNet(BasicBlock, num_blocks, num_classes=self.n_classes)
    
    self.total_flops = self.countFlops(backbone_model)
    self.threshold_flop_list = self.where_insert_early_exits()

    for i in range(1, n_layers + 1):
      intermediate_block_layer = getattr(backbone_model, "layer%s"%(i))

      for k in range(0, num_blocks[i-1]):

        basic_block = intermediate_block_layer[k]
        for layer in basic_block_list:
          self.layers.append(getattr(basic_block, layer))
          if (self.is_suitable_for_exit()):
            self.add_exit_block()

    

  def early_exit_mobilenet(self):
    """
    This method inserts early exits into a Mobilenet V2 model
    """

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    last_channel = 1280
    
    # Loads the backbone model. In other words, Mobilenet architecture provided by Pytorch.
    backbone_model = models.mobilenet_v2(self.pretrained)

    # It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    self.verifies_nr_exits(backbone_model.features)
    
    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    for i, layer in enumerate(backbone_model.features.children()):
      
      self.layers.append(layer)    
      if (self.is_suitable_for_exit()):
        self.add_exit_block()

    self.layers.append(nn.AdaptiveAvgPool2d(1))
    self.stages.append(nn.Sequential(*self.layers))
    

    self.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(last_channel, self.n_classes),)

    self.set_device()
    self.softmax = nn.Softmax(dim=1)

  def forwardTrain(self, x):
    """
    This method is used to train the early-exit DNN model
    """
    
    output_list, conf_list, class_list  = [], [], []

    for i, exitBlock in enumerate(self.exits):
      
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      output_list.append(output_branch)

      #Confidence is the maximum probability of belongs one of the predefined classes and inference_class is the argmax
      conf, infered_class = torch.max(self.softmax(output_branch), 1)
      conf_list.append(conf)
      class_list.append(infered_class)

    x = self.stages[-1](x)

    x = torch.flatten(x, 1)

    output = self.classifier(x)
    infered_conf, infered_class = torch.max(self.softmax(output), 1)
    output_list.append(output)
    conf_list.append(infered_conf)
    class_list.append(infered_class)

    return output_list, conf_list, class_list

  def temperature_scale_overall(self, logits, temp_overall):
    if (temp_overall is None):
      return torch.zeros(logits.shape).to(self.device)
    else:    
      return torch.div(logits, temp_overall)


  def temperature_scale_branches(self, logits, temp, exit_branch):
    if(temp[exit_branch] is None):
      return torch.zeros(logits.shape).to(self.device)
    else:
      return torch.div(logits, temp[exit_branch])

  def forwardAllExits(self, x):  

    output_list, conf_list, infered_class_list = [], [], []

    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)

      output_branch = exitBlock(x)
      conf, infered_class = torch.max(self.softmax(output_branch), 1)

      output_list.append(output_branch), infered_class_list.append(infered_class)
      conf_list.append(conf)

    x = self.stages[-1](x)
    
    x = torch.flatten(x, 1)

    output = self.classifier(x)    
    output_list.append(output)

    conf, infered_class = torch.max(self.softmax(output), 1)

    infered_class_list.append(infered_class), conf_list.append(conf)
   
    return output_list, conf_list, infered_class_list


  def forwardOverallCalibration(self, x, temp_overall):
    output_list, conf_list, class_list = [], [], []
    n_exits = self.n_branches + 1

    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      
      output_branch = self.temperature_scale_overall(output_branch, temp_overall)

      conf_branch, infered_class_branch = torch.max(self.softmax(output_branch), 1)

      output_list.append(output_branch)
      conf_list.append(conf_branch), class_list.append(infered_class_branch)
      
    x = self.stages[-1](x)
    
    x = torch.flatten(x, 1)

    output = self.classifier(x)
    output = self.temperature_scale_overall(output, temp_overall)
    output_list.append(output)

    conf, infered_class = torch.max(self.softmax(output), 1)
    conf_list.append(conf), class_list.append(infered_class)

    return output_list, conf_list, class_list

  def forwardBranchesCalibration(self, x, temp_branches):
    output_list, conf_list, class_list = [], [], []
    n_exits = self.n_branches + 1

    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      output_branch = self.temperature_scale_branches(output_branch, temp_branches, i)

      conf_branch, infered_class_branch = torch.max(self.softmax(output_branch), 1)

      output_list.append(output_branch)
      conf_list.append(conf_branch), class_list.append(infered_class_branch)
      
    x = self.stages[-1](x)
    
    x = torch.flatten(x, 1)

    output = self.classifier(x)
    output = self.temperature_scale_branches(output, temp_branches, -1)
    output_list.append(output)

    conf, infered_class = torch.max(self.softmax(output), 1)
    conf_list.append(conf), class_list.append(infered_class)

    return output_list, conf_list, class_list

  def forwardAllSamplesCalibration(self, x, temp_all_samples):
    output_list, conf_list, class_list = [], [], []
    n_exits = self.n_branches + 1

    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      output_branch = self.temperature_scale_branches(output_branch, temp_all_samples, i)

      conf_branch, infered_class_branch = torch.max(self.softmax(output_branch), 1)

      output_list.append(output_branch)
      conf_list.append(conf_branch), class_list.append(infered_class_branch)
      
    x = self.stages[-1](x)
    
    x = torch.flatten(x, 1)

    output = self.classifier(x)
    output = self.temperature_scale_branches(output, temp_all_samples, -1)
    output_list.append(output)

    conf, infered_class = torch.max(self.softmax(output), 1)
    conf_list.append(conf), class_list.append(infered_class)

    return output_list, conf_list, class_list


  def forwardEval(self, x, p_tar):
    """
    This method is used to train the early-exit DNN model
    """
    output_list, conf_list, class_list  = [], [], []

    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)

      output_branch = exitBlock(x)
      conf, infered_class = torch.max(self.softmax(output_branch), 1)

      # Note that if confidence value is greater than a p_tar value, we terminate the dnn inference and returns the output
      if (conf.item() >= p_tar):
        return output_branch, conf, infered_class, i+1

      else:
        output_list.append(output_branch)
        conf_list.append(conf)
        class_list.append(infered_class)

    x = self.stages[-1](x)
    
    x = torch.flatten(x, 1)

    output = self.classifier(x)
    conf, infered_class = torch.max(self.softmax(output), 1)
    
    # Note that if confidence value is greater than a p_tar value, we terminate the dnn inference and returns the output
    # This also happens in the last exit
    if (conf.item() >= p_tar):
      return output, conf, infered_class, self.n_branches 
    else:

      # If any exit can reach the p_tar value, the output is give by the more confidence output.
      # If evaluation, it returns max(output), max(conf) and the number of the early exit.

      conf_list.append(conf)
      class_list.append(infered_class)
      output_list.append(output)
      conf_temp_list = [conf.item() for conf in conf_list]
      max_conf = np.argmax(conf_temp_list)
      return output_list[max_conf], conf_list[max_conf], class_list[max_conf], self.n_branches



  def forward(self, x, p_tar=0.5, training=True):
    """
    This implementation supposes that, during training, this method can receive a batch containing multiple images.
    However, during evaluation, this method supposes an only image.
    """
    if (training):
      return self.forwardTrain(x)
    else:
      return self.forwardEval(x, p_tar)


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class BranchesModelWithTemperature(nn.Module):
  def __init__(self, model, n_branches, device, lr=0.01, max_iter=50):
    super(BranchesModelWithTemperature, self).__init__()
    """
    This method calibrates a early-exit DNN. The calibration goal is to turn the classification confidencer closer to the real model's accuracy.
    In this work, we apply the calibration method called Temperature Scaling.
    The paper below explains in detail: https://arxiv.org/pdf/1706.04599.pdf

    Here, we follow two approaches:
    * we find a temperature parameter for each side branch
    * we find a temperature parameter for the entire early-exit DNN model.

    """
    self.model = model            #this receives the architecture model. It is important to notice this models has already trained. 
    self.n_branches = n_branches  #the number of side branches or early exits.
    self.n_exits = self.n_branches + 1 
    self.device = device               
    self.lr = lr                  # defines the learning rate of the calibration process.
    self.max_iter = max_iter      #defines the number of iteractions to train the calibration process
    
    # This line initiates a parameters list of the temperature 
    self.temperature_branches = [nn.Parameter(1.5*torch.ones(1).to(self.device)) for i in range(self.n_exits)]
    self.softmax = nn.Softmax(dim=1)
    
    # This line initiates a single temperature parameter for the entire early-exit DNN model
    self.temperature_overall = nn.Parameter(1.5*torch.ones(1).to(self.device))
    self.temperature = nn.Parameter((torch.ones(1) * 1.5).to(self.device))

  def temperature_scale(self, logits):

    temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / temperature

  def forwardAllSamplesCalibration(self, x):
    return self.model.forwardAllSamplesCalibration(x, self.temperature_branches)

  def forwardBranchesCalibration(self, x):
    return self.model.forwardBranchesCalibration(x, self.temperature_branches)

  def forwardOverallCalibration(self, x):
     return self.model.forwardOverallCalibration(x, self.temperature_overall)
  
  def temperature_scale_overall(self, logits):
    return torch.div(logits, self.temperature_overall)

  def temperature_scale_branches(self, logits):
    return torch.div(logits, self.temperature_branch)
  
  def save_temperature_branches(self, error_measure_dict, save_branches_path):


    df = pd.read_csv(save_branches_path) if (os.path.exists(save_branches_path)) else pd.DataFrame()

            
    df = df.append(pd.Series(error_measure_dict), ignore_index=True)
    df.to_csv(save_branches_path)

  def save_temperature_overall(self, error_measure_dict, save_overall_path):
    """
    This method saves the temperature in an csv file in self.save_path
    This saves: 
    p_tar: which means the threshold
    before_temperature_nll: the error before the calibration  
    after_temperature_nll: the error after the calibration
    temperature parameter:
                 
    """

    df = pd.read_csv(save_overall_path) if (os.path.exists(save_overall_path)) else pd.DataFrame()
    
    df = df.append(pd.Series(error_measure_dict), ignore_index=True)
    df.to_csv(save_overall_path)

  def calibrate_overall2(self, val_loader, p_tar, save_overall_path):
    """
    This method calibrates the entire model. In other words, this method finds a singles temperature parameter 
    for the entire early-exit DNN model
    """
 

    nll_criterion = nn.CrossEntropyLoss().to(self.device)
    ece = ECE()

    optimizer = optim.LBFGS([self.temperature_overall], lr=self.lr, max_iter=self.max_iter)

    logits_list, labels_list = [], []

    self.model.eval()
    with torch.no_grad():
      for (data, target) in tqdm(val_loader):

        data, target = data.to(self.device), target.to(self.device)
        
        logits, conf, infer_class, exit_branch = self.model(data, p_tar, training=False)

        logits_list.append(logits), labels_list.append(target)

    logits_list = torch.cat(logits_list).to(self.device)
    labels_list = torch.cat(labels_list).to(self.device)

    before_temperature_nll = nll_criterion(logits_list, labels_list).item()
    
    before_ece = ece(logits_list, labels_list).item()

    def eval():
      optimizer.zero_grad()
      loss = nll_criterion(self.temperature_scale_overall(logits_list), labels_list)
      loss.backward()
      return loss
    
    optimizer.step(eval)

    after_temperature_nll = nll_criterion(self.temperature_scale_overall(logits_list), labels_list).item()
    after_ece = ece(self.temperature_scale_overall(logits_list), labels_list).item()

    print("Before NLL: %s, After NLL: %s"%(before_temperature_nll, after_temperature_nll))
    print("Before ECE: %s, After ECE: %s"%(before_ece, after_ece))
    print("Temp %s"%(self.temperature_overall.item()))

    error_measure_dict = {"p_tar": p_tar, "before_nll": before_temperature_nll, "after_nll": after_temperature_nll, 
                          "before_ece": before_ece, "after_ece": after_ece, 
                          "temperature": self.temperature_overall.item()}
    
    # This saves the parameter to save the temperature parameter
    self.save_temperature_overall(error_measure_dict, save_overall_path)


  def calibrate_overall(self, val_loader, p_tar, save_overall_path):

    self.cuda()
    nll_criterion = nn.CrossEntropyLoss().to(self.device)
    ece_criterion = _ECELoss().to(self.device)

    # First: collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []
    with torch.no_grad():
      for data, label in val_loader:
        data, label = data.to(self.device), label.to(self.device)

        logits, _, _, exit_branch = self.model(data, p_tar, training=False)


        logits_list.append(logits)
        labels_list.append(label)
    
    logits = torch.cat(logits_list).cuda()
    labels = torch.cat(labels_list).cuda()

    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(logits, labels).item()
    before_temperature_ece = ece_criterion(logits, labels).item()
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

    def eval():
      optimizer.zero_grad()
      loss = nll_criterion(self.temperature_scale(logits), labels)
      loss.backward()
      return loss
    
    optimizer.step(eval)

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
    after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
    print('Optimal temperature: %.3f' % self.temperature.item())
    print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

    return self

  def calibrate_branches_all_samples(self, val_loader, p_tar, save_branches_path):

    nll_criterion = nn.CrossEntropyLoss().to(self.device)
    ece = ECE()

    logits_list = [[] for i in range(self.n_exits)]
    labels_list = [[] for i in range(self.n_exits)]

    before_ece_list, after_ece_list = [], []
    
    before_temperature_nll_list, after_temperature_nll_list = [], []

    temperature_branch_list = []

    error_measure_dict = {"p_tar": p_tar}

    self.model.eval()
    with torch.no_grad():
      for (data, target) in tqdm(val_loader):
          
        data, target = data.to(self.device), target.to(self.device)

        logits, _, _ = self.model.forwardAllExits(data)


        for i in range(self.n_exits):
          logits_list[i].append(logits[i])
          labels_list[i].append(target)

    for i in range(self.n_exits):
      print("Exit: %s"%(i))

      if (len(logits_list[i]) == 0):
        before_temperature_nll_list.append(None), after_temperature_nll_list.append(None)
        before_ece_list.append(None), after_ece_list.append(None)
        temperature_branch_list.append(None)
        continue

      self.temperature_branch = nn.Parameter((torch.ones(1)*1.0).to(self.device))
      optimizer = optim.LBFGS([self.temperature_branch], lr=self.lr, max_iter=self.max_iter)

      logit_branch = torch.cat(logits_list[i]).to(self.device)
      label_branch = torch.cat(labels_list[i]).to(self.device)

      before_temperature_nll = nll_criterion(logit_branch, label_branch).item()
      before_temperature_nll_list.append(before_temperature_nll)

      before_ece = ece(logit_branch, label_branch).item()
      before_ece_list.append(before_ece)

      def eval():
        optimizer.zero_grad()
        loss = nll_criterion(self.temperature_scale_branches(logit_branch), label_branch)
        loss.backward()
        return loss
      
      optimizer.step(eval)

      after_temperature_nll = nll_criterion(self.temperature_scale_branches(logit_branch), label_branch).item()
      after_temperature_nll_list.append(after_temperature_nll)
      
      after_ece = ece(self.temperature_scale_branches(logit_branch), label_branch).item()
      after_ece_list.append(after_ece)

      print("Branch: %s, Before NLL: %s, After NLL: %s"%(i+1, before_temperature_nll, after_temperature_nll))
      print("Branch: %s, Before ECE: %s, After ECE: %s"%(i+1, before_ece, after_ece))

      print("Temp Branch %s: %s"%(i+1, self.temperature_branch.item()))

      self.temperature_branches[i] = self.temperature_branch

    self.temperature_branches = [temp_branch.item() for temp_branch in self.temperature_branches]
    
    for i in range(self.n_exits):

      error_measure_dict.update({"before_nll_branch_%s"%(i+1): before_temperature_nll_list[i], 
                                 "before_ece_branch_%s"%(i+1): before_ece_list[i],
                                 "after_nll_branch_%s"%(i+1): after_temperature_nll_list[i],
                                 "after_ece_branch_%s"%(i+1): after_ece_list[i],
                                 "temperature_branch_%s"%(i+1): self.temperature_branches[i]})

    
    # This saves the parameter to save the temperature parameter for each side branch

    self.save_temperature_branches(error_measure_dict, save_branches_path)


  def calibrate_branches(self, val_loader, dataset, p_tar, save_branches_path, data_augmentation=False):
    """
    This method calibrates for each side branch. In other words, this method finds a temperature parameter 
    for each side branch of the early-exit DNN model.
    """

    nll_criterion = nn.CrossEntropyLoss().to(self.device)
    ece = ECE()
    temperature_branch_list = []

    logits_list = [[] for i in range(self.n_exits)]
    labels_list = [[] for i in range(self.n_exits)]
    idx_sample_exit_list = [[] for i in range(self.n_exits)]
    before_temperature_nll_list, after_temperature_nll_list = [], []
    before_ece_list, after_ece_list = [], []

    error_measure_dict = {"p_tar": p_tar}

    self.model.eval()
    with torch.no_grad():
      for (data, target) in tqdm(val_loader):
          
        data, target = data.to(self.device), target.to(self.device)
        
        logits, _, _, exit_branch = self.model(data, p_tar, training=False)

        logits_list[exit_branch].append(logits)
        labels_list[exit_branch].append(target)


    for i in range(self.n_exits):
      print("Exit: %s"%(i+1))

      if (len(logits_list[i]) == 0):
        before_temperature_nll_list.append(None), after_temperature_nll_list.append(None)
        before_ece_list.append(None), after_ece_list.append(None)
        temperature_branch_list.append(None)
        continue

      self.temperature_branch = nn.Parameter((torch.ones(1)*1.5).to(self.device))
      
      optimizer = optim.LBFGS([self.temperature_branch], lr=self.lr, max_iter=self.max_iter)

      logit_branch = torch.cat(logits_list[i]).to(self.device)
      label_branch = torch.cat(labels_list[i]).to(self.device)

      before_temperature_nll = nll_criterion(logit_branch, label_branch).item()
      before_temperature_nll_list.append(before_temperature_nll)

      before_ece = ece(logit_branch, label_branch).item()
      before_ece_list.append(before_ece)
      weight_list = np.linspace(1, 0.3, self.n_exits)
      def eval():
        optimizer.zero_grad()
        loss = weight_list[i]*nll_criterion(self.temperature_scale_branches(logit_branch), label_branch)
        loss.backward()
        return loss
      
      optimizer.step(eval)

      after_temperature_nll = nll_criterion(self.temperature_scale_branches(logit_branch), label_branch).item()
      after_temperature_nll_list.append(after_temperature_nll)

      after_ece = ece(self.temperature_scale_branches(logit_branch), label_branch).item()
      after_ece_list.append(after_ece)

      
      self.temperature_branches[i] = self.temperature_branch
      #temperature_branch_list.append(self.temperature_branch.item())

      print("Branch: %s, Before NLL: %s, After NLL: %s"%(i+1, before_temperature_nll, after_temperature_nll))
      print("Branch: %s, Before ECE: %s, After ECE: %s"%(i+1, before_ece, after_ece))

      print("Temp Branch %s: %s"%(i+1, self.temperature_branch.item()))

    self.temperature_branches = [temp_branch.item() for temp_branch in self.temperature_branches]

    for i in range(self.n_exits):
      error_measure_dict.update({"before_nll_branch_%s"%(i+1): before_temperature_nll_list[i], 
                                 "before_ece_branch_%s"%(i+1): before_ece_list[i],
                                 "after_nll_branch_%s"%(i+1): after_temperature_nll_list[i],
                                 "after_ece_branch_%s"%(i+1): after_ece_list[i],
                                 "temperature_branch_%s"%(i+1): self.temperature_branches[i]})
    
    # This saves the parameter to save the temperature parameter for each side branch

    self.save_temperature_branches(error_measure_dict, save_branches_path)

    return self

def calibrating_early_exit_dnn(model, val_loader, dataset, p_tar, n_branches, device, saveTempBranchesPath):
  print("Calibrating ...")

  overall_calibrated_model = BranchesModelWithTemperature(model, n_branches, device)
  overall_calibrated_model.calibrate_overall(val_loader, p_tar, saveTempBranchesPath["calib_overall"])
    
  branches_calibrated_model = BranchesModelWithTemperature(model, n_branches, device)
  branches_calibrated_model.calibrate_branches(val_loader, dataset, p_tar, saveTempBranchesPath["calib_branches"])


  branches_calibrated_all_samples = BranchesModelWithTemperature(model, n_branches, device)
  branches_calibrated_all_samples.calibrate_branches_all_samples(val_loader, p_tar, 
                                                                     saveTempBranchesPath["calib_branches_all_samples"])

  calib_models_dict = {"calib_overall": overall_calibrated_model, 
                       "calib_branches": branches_calibrated_model,
                       "calib_all_samples": branches_calibrated_all_samples}
  
  return calib_models_dict

def experiment_early_exit_inference(model, test_loader, p_tar, n_branches, device, model_type="no_calib"):
  df_result = pd.DataFrame()

  n_exits = n_branches + 1
  conf_branches_list, infered_class_branches_list, target_list = [], [], []
  correct_list, exit_branch_list, id_list = [], [], []

  model.eval()

  with torch.no_grad():
    for i, (data, target) in tqdm(enumerate(test_loader, 1)):
      
      #print(model_type)
      data, target = data.to(device), target.to(device)

      if (model_type == "no_calib"):
        _, conf_branches, infered_class_branches = model.forwardAllExits(data)
        #print([conf.item() for conf in conf_branches])

      elif(model_type == "calib_overall"):
        _, conf_branches, infered_class_branches = model.forwardOverallCalibration(data)
        #print([conf.item() for conf in conf_branches])
        #print(model.temperature_overall)

      elif(model_type == "calib_branches"):
        _, conf_branches, infered_class_branches = model.forwardBranchesCalibration(data)
        #print([conf.item() for conf in conf_branches])

        #print(model.temperature_branches)

      else:
        _, conf_branches, infered_class_branches = model.forwardAllSamplesCalibration(data)

      conf_branches_list.append([conf.item() for conf in conf_branches])
      infered_class_branches_list.append([inf_class.item() for inf_class in infered_class_branches])    
      correct_list.append([infered_class_branches[i].eq(target.view_as(infered_class_branches[i])).sum().item() for i in range(n_exits)])
      id_list.append(i)
      target_list.append(target.item())



      del data, target
      torch.cuda.empty_cache()
      #break

  conf_branches_list = np.array(conf_branches_list)
  infered_class_branches_list = np.array(infered_class_branches_list)
  correct_list = np.array(correct_list)
  
  print(model_type)
  print("Acc:")
  print([sum(correct_list[:, i])/len(correct_list[:, i]) for i in range(n_exits)])

  results = {"p_tar": [p_tar]*len(target_list), "target": target_list, "id": id_list}
  for i in range(n_exits):
    results.update({"conf_branch_%s"%(i+1): conf_branches_list[:, i],
                    "infered_class_branches_%s"%(i+1): infered_class_branches_list[:, i],
                    "correct_branch_%s"%(i+1): correct_list[:, i]})

  return results

def save_result(result, save_path):
  df_result = pd.read_csv(save_path) if (os.path.exists(save_path)) else pd.DataFrame()
  df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
  df_result = df_result.append(df)
  df_result.to_csv(save_path)


def save_all_results(no_calib_result, calib_overall_result, calib_branches_result, calib_all_samples_result, saveResultsDict):
  save_result(no_calib_result, saveResultsDict["no_calib"])
  save_result(calib_overall_result, saveResultsDict["calib_overall"])
  save_result(calib_branches_result, saveResultsDict["calib_branches"])
  save_result(calib_all_samples_result, saveResultsDict["calib_branches_all_samples"])


def extract_confidence_data(model, test_loader, val_loader, dataset, p_tar_list, n_branches, device, saveTempBranchesPath, saveResultsDict):

  for p_tar in p_tar_list:
    print("P_tar: %s"%(p_tar))

    
    calib_models_dict = calibrating_early_exit_dnn(model, val_loader, dataset, p_tar, n_branches, device, saveTempBranchesPath)

    no_calib_result = experiment_early_exit_inference(model, test_loader, p_tar, n_branches, device, model_type="no_calib")
    
    calib_overall_result = experiment_early_exit_inference(calib_models_dict["calib_overall"], test_loader, p_tar, n_branches, device, 
                                                           model_type="calib_overall")
    
    calib_branches_result = experiment_early_exit_inference(calib_models_dict["calib_branches"], test_loader, p_tar, n_branches, device, 
                                                            model_type="calib_branches")
    
    calib_all_samples_result = experiment_early_exit_inference(calib_models_dict["calib_all_samples"], test_loader, p_tar, 
                                                               n_branches, device, model_type="all_samples")



    save_all_results(no_calib_result, calib_overall_result, calib_branches_result, 
                     calib_all_samples_result, saveResultsDict)


input_dim = 224
batch_size_train = 64
batch_size_test = 1
model_id = 3
split_ratio = 0.2
n_classes = 258
pretrained = False
n_branches = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_shape = (3, input_dim, input_dim)
learning_rate = 0.005
weight_decay = 0
momentum = 0
steps = 10
p_tar_calib = 0.8

distribution = "linear"
exit_type = "bnpool"
dataset_name = "caltech256"
model_name = "resnet152"

root_save_path = os.path.dirname(__file__)

save_indices_path = os.path.join(root_save_path, "datasets", dataset_name, "indices")

dataset_path = os.path.join(root_save_path, "datasets", dataset_name, "256_ObjectCategories")

model_save_path = os.path.join(root_save_path, "appEdge", "api", "services", "models",
	dataset_name, model_name, "models", 
	"ee_%s_branches_%s_id_%s.pth"%(model_name, n_branches, model_id))

dataset = LoadDataset(input_dim, batch_size_train, batch_size_test, model_id)
train_loader, val_loader, test_loader = dataset.caltech_256(dataset_path, split_ratio, dataset_name, save_indices_path)

early_exit_dnn = Early_Exit_DNN(model_name, n_classes, pretrained, n_branches, input_shape, exit_type, device, distribution=distribution)
early_exit_dnn = early_exit_dnn.to(device)
early_exit_dnn.load_state_dict(torch.load(model_save_path, map_location=device)["model_state_dict"])

result_path = os.path.join(root_save_path, "appEdge", "api", "services", "models",
	dataset_name, model_name, "results")

if (not os.path.exists(result_path)):
  os.makedirs(result_path)

save_no_calib_path =  os.path.join(result_path, "no_calib_exp_data_%s_alert.csv"%(model_id))
save_calib_overall_path =  os.path.join(result_path, "calib_overall_exp_data_%s_alert.csv"%(model_id))
save_calib_branches_path =  os.path.join(result_path, "calib_branches_exp_data_%s_alert.csv"%(model_id))
save_calib_all_samples_path =  os.path.join(result_path, "calib_all_samples_branches_exp_data_%s_alert.csv"%(model_id))


saveResultsDict = {"no_calib": save_no_calib_path, "calib_overall": save_calib_overall_path, 
"calib_branches": save_calib_branches_path,
"calib_branches_all_samples": save_calib_all_samples_path}


saveTempOverallPath = os.path.join(root_save_path, "appEdge", "api", "services", "models",
  dataset_name, model_name, "temperature", "temp_overall_id_%s.csv"%(model_id))

saveTempBranchesPath = os.path.join(root_save_path, "appEdge", "api", "services", "models",
  dataset_name, model_name, "temperature", "temp_branches_id_%s.csv"%(model_id))

saveTempBranchesAllSamplesPath = os.path.join(root_save_path, "appEdge", "api", "services", "models",
  dataset_name, model_name, "temperature", "temp_all_samples_id_%s.csv"%(model_id))

saveTempDict = {"calib_overall": saveTempOverallPath, "calib_branches": saveTempBranchesPath,
                "calib_branches_all_samples": saveTempBranchesAllSamplesPath}


#p_tar_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
p_tar_list = [0.8]
extract_confidence_data(early_exit_dnn, test_loader, val_loader, dataset, p_tar_list, n_branches, device, saveTempDict, saveResultsDict)
