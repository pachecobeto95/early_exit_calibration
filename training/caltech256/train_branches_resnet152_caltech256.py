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
import ssl
ssl._create_default_https_context = ssl._create_unverified_context




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
    self.transformations_train = transforms.Compose([transforms.Resize((input_dim, input_dim)),
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
                                                     transforms.Resize(input_dim), 
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
    train_loader = DataLoader(train_data, self.batch_size_train, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, self.batch_size_test, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, self.batch_size_test, num_workers=4, pin_memory=True)

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

    val_set = datasets.ImageFolder(dataset_path, transform=self.transformations_test)
    test_set = datasets.ImageFolder(dataset_path, transform=self.transformations_test)

    if (os.path.exists(os.path.join(savePath_idx, "training_idx_%s_id_%s.npy"%(dataset_name, self.model_id)))):
      
      train_idx = np.load(os.path.join(savePath_idx, "training_idx_%s_id_%s.npy"%(dataset_name, self.model_id)))
      val_idx = np.load(os.path.join(savePath_idx, "validation_idx_%s_id_%s.npy"%(dataset_name, self.model_id)))
      #test_idx = np.load(os.path.join(savePath_idx, "test_idx_%s_id_%s.npy"%(dataset_name, self.model_id)), allow_pickle=True)
      #test_idx = np.array(list(test_idx.tolist()))
    else:
      # This line get the indices of the samples which belong to the training dataset and test dataset. 
      train_idx, test_idx = self.get_indices(train_set, split_ratio)

      # This line mounts the training and test dataset, selecting the samples according indices. 
      train_data = torch.utils.data.Subset(train_set, indices=train_idx)

      # This line gets the indices to split the train dataset into training dataset and validation dataset.
      train_idx, val_idx = self.get_indices(train_data, split_ratio)

      np.save(os.path.join(savePath_idx, "training_idx_%s_id_%s.npy"%(dataset_name, self.model_id)), train_idx)
      np.save(os.path.join(savePath_idx, "validation_idx_%s_id_%s.npy"%(dataset_name, self.model_id)), val_idx)

    # This line mounts the training and test dataset, selecting the samples according indices. 
    train_data = torch.utils.data.Subset(train_set, indices=train_idx)
    val_data = torch.utils.data.Subset(val_set, indices=val_idx)
    #test_data = torch.utils.data.Subset(test_set, indices=test_idx)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size_train, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size_test, num_workers=4)
    #test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size_test, num_workers=4)

    return train_loader, val_loader, 0

  def simple_caltech256(self, dataset_path, split_ratio, dataset_name, savePath_idx):

    train_set = datasets.ImageFolder(dataset_path, transform=self.transformations_train)
    val_set = datasets.ImageFolder(dataset_path, transform=self.transformations_test)
    test_set = datasets.ImageFolder(dataset_path, transform=self.transformations_test)


    train_images_list, val_images_list, test_images_list = [], [], []
    class_list = []
    idx_list = []

    if (os.path.exists(os.path.join(savePath_idx, "training_idx_%s_id_%s.npy"%(dataset_name, self.model_id)))):
      train_images_list = np.load(os.path.join(savePath_idx, "training_idx_%s_id_%s.npy"%(dataset_name, self.model_id)))
      val_images_list = np.load(os.path.join(savePath_idx, "validation_idx_%s_id_%s.npy"%(dataset_name, self.model_id)))
      test_images_list = np.load(os.path.join(savePath_idx, "test_idx_%s_id_%s.npy"%(dataset_name, self.model_id)))

    else:

      for category_path in glob.glob(dataset_path+"/*"):    
        
        image_samples_list = np.array(sorted(glob.glob(os.path.join(category_path, "*"))))
        index_list = list(range(0, len(image_samples_list) - 1))

        train_index = random.sample(index_list, train_qntd)
        idx_list.append(train_index)

        train_images_list.append(image_samples_list[train_index])        
        class_list.append(category_path.split('/')[-1])

      train_images_list = [val for sublist in train_images_list for val in sublist]
      random.shuffle(train_images_list)

      idx_to_class = {i:j for i, j in enumerate(class_list)}
      class_to_idx = {value:key for key,value in idx_to_class.items()}


      for i, category_path in enumerate(glob.glob(dataset_path + "/*")):

        image_samples_list = np.array(sorted(glob.glob(os.path.join(category_path, "*"))))
        val_index_list = np.array(list(range(0, len(image_samples_list) - 1)))[~np.array(idx_list[i])]

        val_indexes = random.sample(list(val_index_list), val_qntd)
        val_images_list.append(image_samples_list[val_indexes])        

        idx_list[i] += val_indexes

      val_images_list = [val for sublist in val_images_list for val in sublist]
      random.shuffle(val_images_list)


      for i, category_path in enumerate(glob.glob(dataset_path+"/*")):
        image_samples_list = np.array(sorted(glob.glob(os.path.join(category_path, "*"))))
        test_index_list = np.array(list(range(0, len(image_samples_list) - 1)))
        tmp_test_index_list = list(set(test_index_list) - set(idx_list[i]))

        test_indexes = random.sample(tmp_test_index_list, test_qntd)
        test_images_list.append(image_samples_list[test_indexes])        

      test_images_list = [val for sublist in test_images_list for val in sublist]
      random.shuffle(test_images_list)


    train_data = torch.utils.data.Subset(train_set, indices=train_images_list)
    val_data = torch.utils.data.Subset(val_set, indices=val_images_list)
    test_data = torch.utils.data.Subset(test_set, indices=test_images_list)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size_train, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size_test, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size_test, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader



  def getDataset(self, root_path, dataset_name, split_ratio, savePath_idx):
    self.dataset_name = dataset_name
    def func_not_found():
      print("No dataset %s is found"%(self.dataset_name))

    func_name = getattr(self, self.dataset_name, func_not_found)
    train_loader, val_loader, test_loader = func_name(root_path, split_ratio, savePath_idx)
    return train_loader, val_loader, test_loader



class EarlyExitBlock(nn.Module):
  """
  This EarlyExitBlock allows the model to terminate early when it is confident for classification.
  """
  def __init__(self, input_shape, pool_size, n_classes, exit_type, device):
    super(EarlyExitBlock, self).__init__()
    self.input_shape = input_shape

    _, channel, width, height = input_shape
    self.expansion = width * height if exit_type == 'plain' else 1

    self.layers = nn.ModuleList()

    if (exit_type == 'bnpool'):
      self.layers.append(nn.BatchNorm2d(channel))

    if (exit_type != 'plain'):
      self.layers.append(nn.AdaptiveAvgPool2d(pool_size))
    
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



def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class DownSample(nn.Module):
  def __init__(self, downsample, feat):
    super().__init__()
    self.downsample = downsample
    self.feat = feat
  
  def forward(self, x):
    identity = self.downsample(self.feat)
    #x = x.clone() + identity
    #return x + identity
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

    self.pool_size = 7 if(self.model_name == "vgg16") else 1
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
    self.exits.append(EarlyExitBlock(feature_shape, self.pool_size, self.n_classes, self.exit_type, self.device))#.to(self.device))
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
    temperature = temp_overall.unsqueeze(1).expand(logits.size(0), logits.size(1)).to(self.device)
    return logits / temperature

  def temperature_scale_branches(self, logits, temp_branches, exit_branch):
    temperature = temp_branches[exit_branch].unsqueeze(1).expand(logits.size(0), logits.size(1)).to(self.device)
    return logits / temperature

  def forward_inference_calib_overall(self, x, p_tar, temp_overall):
    """
    This method is used to experiment of early-exit DNNs with overall calibration.
    """
    output_list, conf_list, class_list  = [], [], []
    n_exits = self.n_branches + 1
    exit_branches = np.zeros(n_exits)
    wasClassified = False

    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      output_branch = self.temperature_scale_overall(output_branch, temp_overall)

      conf_branch, infered_class_branch = torch.max(self.softmax(output_branch), 1)
      conf_list.append(conf_branch.item()), class_list.append(infered_class_branch)

      if (conf_branch.item() >= p_tar):
        exit_branches[i] = 1

        if (not wasClassified):
          actual_exit_branch = i
          actual_conf = conf_branch.item()
          actual_inferred_class = infered_class_branch
          wasClassified = True

    x = self.stages[-1](x)
    
    x = torch.flatten(x, 1)

    output = self.classifier(x)
    output = self.temperature_scale_overall(output, temp_overall)

    conf, infered_class = torch.max(self.softmax(output), 1)
    conf_list.append(conf.item()), class_list.append(infered_class)

    exit_branches[-1] = 1

    if (conf.item() <  p_tar):
      max_conf = np.argmax(conf_list)
      conf_list[-1] = conf_list[max_conf]
      class_list[-1] = class_list[max_conf]

    if (not wasClassified):
      actual_exit_branch = self.n_branches
      actual_conf = conf_list[-1]
      actual_inferred_class = class_list[-1]

    return actual_conf, actual_inferred_class, actual_exit_branch, conf_list, class_list, exit_branches

  def forward_inference_calib_branches(self, x, p_tar, temp_branches):
    """
    This method is used to experiment of early-exit DNNs with calibration in all the branches.
    """

    output_list, conf_list, class_list  = [], [], []
    n_exits = self.n_branches + 1
    exit_branches = np.zeros(n_exits)
    wasClassified = False

    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      output_branch = self.temperature_scale_branches(output_branch, temp_branches, i)

      conf_branch, infered_class_branch = torch.max(self.softmax(output_branch), 1)
      conf_list.append(conf_branch.item()), class_list.append(infered_class_branch)

      if (conf_branch.item() >= p_tar):
        exit_branches[i] = 1

        if (not wasClassified):
          actual_exit_branch = i
          actual_conf = conf_branch.item()
          actual_inferred_class = infered_class_branch
          wasClassified = True

    x = self.stages[-1](x)
    
    x = torch.flatten(x, 1)

    output = self.classifier(x)
    output = self.temperature_scale_branches(output, temp_branches, -1)
    conf, infered_class = torch.max(self.softmax(output), 1)
    conf_list.append(conf.item()), class_list.append(infered_class)

    exit_branches[-1] = 1

    if (conf.item() <  p_tar):
      max_conf = np.argmax(conf_list)
      conf_list[-1] = conf_list[max_conf]
      class_list[-1] = class_list[max_conf]

    if (not wasClassified):
      actual_exit_branch = self.n_branches
      actual_conf = conf_list[-1]
      actual_inferred_class = class_list[-1]

    return actual_conf, actual_inferred_class, actual_exit_branch, conf_list, class_list, exit_branches

  
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
      max_conf = np.argmax(conf_list)
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


def trainBranches(model, train_loader, optimizer, criterion, n_branches, epoch, device, loss_weights):
  running_loss = []
  n_exits = n_branches + 1
  train_acc_dict = {i: [] for i in range(1, (n_exits)+1)}

  model.train()

  for (data, target) in tqdm(train_loader):
    
    data, target = data.to(device), target.to(device)

    output_list, conf_list, class_list = model(data, training=True)

    optimizer.zero_grad()
    loss = 0
    for j, (output, inf_class, weight) in enumerate(zip(output_list, class_list, loss_weights), 1):
      loss += weight*criterion(output, target)
      train_acc_dict[j].append(100*inf_class.eq(target.view_as(inf_class)).sum().item()/target.size(0))

    running_loss.append(float(loss.item()))
    loss.backward()
    optimizer.step()
    

    # clear variables
    del data, target, output_list, conf_list, class_list
    torch.cuda.empty_cache()

  loss = round(np.average(running_loss), 4)
  print("Epoch: %s"%(epoch))
  print("Train Loss: %s"%(loss))

  result_dict = {"epoch":epoch, "train_loss": loss}
  for key, value in train_acc_dict.items():
    result_dict.update({"train_acc_branch_%s"%(key): round(np.average(train_acc_dict[key]), 4)})    
    print("Train Acc Branch %s: %s"%(key, result_dict["train_acc_branch_%s"%(key)]))
  
  return result_dict

def evalBranches(model, val_loader, criterion, n_branches, epoch, device):
  running_loss = []
  val_acc_dict = {i: [] for i in range(1, (n_branches+1)+1)}
  model.eval()

  with torch.no_grad():
    for (data, target) in tqdm(val_loader):
    #for i, (data, target) in enumerate(val_loader, 1):
      #if (i%100 == 0):
        #print("Batch: %s / %s"%(i, len(val_loader)))
      data, target = data.to(device), target.long().to(device)

      output_list, conf_list, class_list = model(data)
      loss = 0
      for j, (output, inf_class, weight) in enumerate(zip(output_list, class_list, loss_weights), 1):
        loss += weight*criterion(output, target)
        val_acc_dict[j].append(100*inf_class.eq(target.view_as(inf_class)).sum().item()/target.size(0))


      running_loss.append(float(loss.item()))    

      # clear variables
      del data, target, output_list, conf_list, class_list
      torch.cuda.empty_cache()

  loss = round(np.average(running_loss), 4)
  print("Epoch: %s"%(epoch))
  print("Val Loss: %s"%(loss))

  result_dict = {"epoch":epoch, "val_loss": loss}
  for key, value in val_acc_dict.items():
    result_dict.update({"val_acc_branch_%s"%(key): round(np.average(val_acc_dict[key]), 4)})    
    print("Val Acc Branch %s: %s"%(key, result_dict["val_acc_branch_%s"%(key)]))
  
  return result_dict

input_dim = 224
batch_size_train = 32
batch_size_test = 1
model_id = 4
split_ratio = 0.2
n_classes = 258
pretrained = True
n_branches = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_shape = (3, input_dim, input_dim)
weight_decay = 0.0001
momentum = 0.9
steps = 10
p_tar_calib = 0.8

distribution = "linear"
exit_type = "bnpool"
dataset_name = "caltech256"
model_name = "resnet152"
root_save_path = "."

dataset_save_path = os.path.join(root_save_path, dataset_name)
save_indices_path = os.path.join(dataset_save_path, "indices")
#create_save_dir(dataset_save_path, model_name, save_indices_path)

dataset_path = "./datasets/caltech256/256_ObjectCategories/"

model_save_path = os.path.join(root_save_path, "ee_%s_branches_%s_id_%s.pth"%(model_name, n_branches, model_id))
history_save_path = os.path.join(root_save_path, "history_%s_branches_%s_id_%s.csv"%(model_name, n_branches, model_id))
save_temp_overall_path = os.path.join(root_save_path, "appEdge", "api", "services", "models", dataset_name,model_name,
 "temperature", "temp_overall_%s_branches_%s_id_%s.csv"%(model_name, n_branches, model_id))

save_temp_branches_path = os.path.join(root_save_path, "appEdge", "api", "services", "models", dataset_name,model_name,
 "temperature", "temp_branches_%s_branches_%s_id_%s.csv"%(model_name, n_branches, model_id))


dataset = LoadDataset(input_dim, batch_size_train, batch_size_test, model_id)
train_loader, val_loader, test_loader = dataset.caltech_256(dataset_path, split_ratio, dataset_name, save_indices_path)


lr = [1.5e-4, 0.01]

early_exit_dnn = Early_Exit_DNN(model_name, n_classes, pretrained, n_branches, input_shape, exit_type, device, distribution=distribution)
early_exit_dnn = early_exit_dnn.to(device)

criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam([{'params': early_exit_dnn.stages.parameters(), 'lr': lr[0]}, 
                      {'params': early_exit_dnn.exits.parameters(), 'lr': lr[1]},
                      {'params': early_exit_dnn.classifier.parameters(), 'lr': lr[0]}], weight_decay=weight_decay)

#optimizer = optim.SGD([{'params': early_exit_dnn.stages.parameters(), 'lr': lr[0]}, 
#                      {'params': early_exit_dnn.exits.parameters(), 'lr': lr[1]},
#                      {'params': early_exit_dnn.classifier.parameters(), 'lr': lr[0]}], 
#                      momentum=momentum, weight_decay=weight_decay,
#                      nesterov=True)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=0, last_epoch=-1, verbose=True)

n_exits = n_branches + 1
#loss_weights = np.linspace(1, 0.3, n_exits)
#loss_weights = np.ones(n_exits)
loss_weights = np.linspace(0.15, 1, n_exits)

epoch = 0
best_val_loss = np.inf
patience = 10
count = 0
df = pd.DataFrame()

while (count < patience):
  epoch+=1
  print("Epoch: %s"%(epoch))
  result = {}
  result.update(trainBranches(early_exit_dnn, train_loader, optimizer, criterion, n_branches, epoch, device, loss_weights))
  result.update(evalBranches(early_exit_dnn, val_loader, criterion, n_branches, epoch, device))
  scheduler.step()

  #df = df.append(pd.Series(result), ignore_index=True)
  #df.to_csv(history_save_path)

  if (result["val_loss"] < best_val_loss):
    best_val_loss = result["val_loss"]
    count = 0
    save_dict = {"model_state_dict": early_exit_dnn.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "epoch": epoch, "val_loss": result["val_loss"]}
    
    for i in range(1, n_exits+1):
      save_dict.update({"val_acc_branch_%s"%(i): result["val_acc_branch_%s"%(i)]})

    torch.save(save_dict, model_save_path)

  else:
    print("Current Patience: %s"%(count))
    count += 1
print("Stop! Patience is finished")
