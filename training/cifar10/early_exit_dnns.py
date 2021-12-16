import os, sys, time, math, os
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
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from pthflops import count_ops
from torch import Tensor
from typing import Callable, Any, Optional, List, Type, Union
import torch.nn.init as init
import functools
from tqdm import tqdm
from networks.mobilenet import MobileNetV2_2
from networks.resnet import resnet18, resnet152
from networks.vgg import vgg16_bn


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
               pretrained: bool, backbone_pretrained: bool, backbone_model_path: str, 
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
    self.pretrained = pretrained
    self.n_branches = n_branches
    self.input_shape = input_shape
    self.exit_type = exit_type
    self.distribution = distribution
    self.device = device
    self.channel, self.width, self.height = input_shape
    self.backbone_pretrained = backbone_pretrained
    self.backbone_model_path = backbone_model_path

    build_early_exit_dnn = self.select_dnn_architecture_model()
    build_early_exit_dnn()

  def select_dnn_architecture_model(self):
    """
    This method selects the backbone to insert the early exits.
    """
  
    ee_mobilenet = self.early_exit_mobilenet if (self.pretrained) else self.early_exit_MobileNet

    architecture_dnn_model_dict = {"alexnet": self.early_exit_alexnet,
                                   "mobilenet": ee_mobilenet,
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
    input = torch.rand(1, self.channel, self.width, self.height).to(self.device)
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
    x = torch.rand(1, 3, 224, 224).to(self.device)
    current_flop, _ = count_ops(intermediate_model, x, verbose=False, print_readable=False)
    return self.stage_id < self.n_branches and current_flop >= self.threshold_flop_list[self.stage_id]

  def add_exit_block(self):
    """
    This method adds an early exit in the suitable position.
    """
    input_tensor = torch.rand(1, self.channel, self.width, self.height)

    self.stages.append(nn.Sequential(*self.layers))
    x = torch.rand(1, 3, 224, 224).to(self.device)
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


    if (self.pretrained):
      backbone_model = models.resnet152(self.pretrained)
    else:
      backbone_model = resnet152(self.n_classes)

    if (self.backbone_pretrained):
      backbone_model.load_state_dict(torch.load(self.backbone_model_path, map_location=self.device)["model_state_dict"])

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
    self.classifier = backbone_model.fc if(self.backbone_pretrained) else nn.Sequential(nn.Linear(2048, self.n_classes))
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

    if (self.pretrained):
      backbone_model = models.resnet18(self.pretrained)
      backbone_model.fc = nn.Linear(backbone_model.fc.in_features, self.n_classes)
    else:
      backbone_model = resnet18(self.n_classes).to(self.device)

    if (self.backbone_pretrained):
      backbone_model.load_state_dict(torch.load(self.backbone_model_path, map_location=self.device)["model_state_dict"])


    # It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    self.verifies_nr_exits(backbone_model)

    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    building_first_layer = ["conv1", "bn1", "relu"]
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
    self.classifier = backbone_model.fc if(self.backbone_pretrained) else nn.Sequential(nn.Linear(512, self.n_classes))
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

    if (self.pretrained):
      backbone_model = models.vgg16_bn(self.pretrained)
      backbone_model.classifier[6] = nn.Linear(backbone_model.classifier[6].in_features, self.n_classes)

    else:
      backbone_model = vgg16(self.n_classes)

    if (self.backbone_pretrained):
      backbone_model.load_state_dict(torch.load(self.backbone_model_path, map_location=self.device)["model_state_dict"])



    backbone_model_features = backbone_model.features
    
    self.total_flops = self.countFlops(backbone_model)
    self.threshold_flop_list = self.where_insert_early_exits()

    for layer in backbone_model_features:
      self.layers.append(layer)
      if (self.is_suitable_for_exit()):
        self.add_exit_block()


    self.layers.append(backbone_model.avgpool)
    self.stages.append(nn.Sequential(*self.layers))
    if(self.backbone_pretrained):
      self.classifier = backbone_model.classifier
    else:
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

    
  def early_exit_MobileNet(self):
    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    n_blocks = 18
    last_channel = 1280
    
    # Loads the backbone model. In other words, Mobilenet architecture provided by Pytorch.
    backbone_model = MobileNetV2_2(self.n_classes, self.device).to(self.device)

    # It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    self.verifies_nr_exits(backbone_model.network)

    if(self.backbone_pretrained):
      backbone_model.load_state_dict(torch.load(self.backbone_model_path, map_location=self.device)["model_state_dict"])

    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()


    for block in range(n_blocks+1):
      self.layers.append(backbone_model.network[block])    
      if (self.is_suitable_for_exit()):
        self.add_exit_block()


    self.layers.append(backbone_model.network[-1][0])
    self.layers.append(backbone_model.network[-1][1])
    
    self.stages.append(nn.Sequential(*self.layers))
    
    self.classifier = backbone_model.network[-1][-1].to(self.device) if(self.backbone_pretrained) else Conv2d(1280, self.n_classes, 
      kernel_size=(1, 1), stride=(1, 1)).to(self.device)
    self.set_device()
    self.softmax = nn.Softmax(dim=1)



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
    backbone_model.classifier[1] = nn.Linear(last_channel, self.n_classes)

    # It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    self.verifies_nr_exits(backbone_model.features)

    if(self.backbone_pretrained):
      backbone_model.load_state_dict(torch.load(self.backbone_model_path, map_location=self.device)["model_state_dict"])
    
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

    if (self.backbone_pretrained):
      self.classifier = backbone_model.classifier[1]

    else:    
      self.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(last_channel, self.n_classes))

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

    #if((self.model_name=="mobilenet") and (self.pretrained)):
    #  x = torch.flatten(x, 1)

    #x = torch.flatten(x, 1)
    output = self.classifier(x)
    infered_conf, infered_class = torch.max(self.softmax(output), 1)
    output_list.append(output)
    conf_list.append(infered_conf)
    class_list.append(infered_class)

    return output_list, conf_list, class_list

  def temperature_scale_overall(self, logits, temp_overall):    
    return torch.div(logits, temp_overall)


  def temperature_scale_branches(self, logits, temp, exit_branch):
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
    
    


    if((self.model_name != "mobilenet") or (self.pretrained)):
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
    
    if((self.model_name != "mobilenet") or (self.pretrained)):
      x = torch.flatten(x, 1)


    output = self.classifier(x)
    output = self.temperature_scale_overall(output, temp_overall)
    output_list.append(output)

    conf, infered_class = torch.max(self.softmax(output), 1)
    conf_list.append(conf), class_list.append(infered_class)

    return conf_list, class_list

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
    

    if((self.model_name != "mobilenet") or (self.pretrained)):
      x = torch.flatten(x, 1)

    output = self.classifier(x)
    output = self.temperature_scale_branches(output, temp_branches, -1)
    output_list.append(output)

    conf, infered_class = torch.max(self.softmax(output), 1)
    conf_list.append(conf), class_list.append(infered_class)

    return conf_list, class_list

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
    
    if((self.model_name != "mobilenet") or (self.pretrained)):
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
