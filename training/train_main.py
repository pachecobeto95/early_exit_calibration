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
                                                     transforms.RandomHorizontalFlip(p = 0.25),
                                                     transforms.RandomRotation(25),
                                                     transforms.ToTensor(), 
                                                     transforms.Normalize(mean = mean, std = std),
                                                     ])

    # Note that we do not apply data augmentation in the test dataset.
    self.transformations_test = transforms.Compose([
                                                     transforms.Resize(256),
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


def runTrainMain(dataLoader, model, criterion, optimizer, epoch, n_epochs, train):
  if(train):
  	model.train()
  else:
  	model.eval()

  acc_list, loss_list = [], []
  for (data, target) in tqdm(loader):
    data, target = data.to(device), target.to(device)

    if (main):
      optimizer.zero_grad()
      output = model(input)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()

    else:
      with torch.no_grad():
        output = model(input)
        loss = criterion(output, target)

    _, infered_class = torch.max(softmax(output), 1)
    acc_list.append(100*infered_class.eq(target.view_as(infered_class)).sum().item()/target.size(0))
    loss_list.append(loss.item())
  

  avg_acc = np.mean(acc_list)
  avg_loss = np.mean(loss_list)

  print("%s: %s"%('Train' if train else 'Eval', np.mean(acc_list)))
  mode = "train" if(main) else "val"
  return {"%s_loss"%(mode): avg_loss, "%s_acc"%(mode): avg_acc}




input_resize, input_dim = 256, 224
batch_size_train = 32
batch_size_test = 1
model_id = 1
split_ratio = 0.2
n_classes = 258
pretrained = True
n_branches = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_shape = (3, input_dim, input_dim)
lr = 0.01
weight_decay = 0.0001
seed = 42
momentum = 0.9
pretrained = True
n_epochs = 200

dataset_name = "caltech256"
model_name = "resnet110"
root_save_path = "."

dataset_save_path = os.path.join(root_save_path, dataset_name)
save_indices_path = os.path.join(dataset_save_path, "indices")
#create_save_dir(dataset_save_path, model_name, save_indices_path)

dataset_path = "./datasets/caltech256/256_ObjectCategories/"

model_save_path = os.path.join(root_save_path, "%s_id_%s.pth"%(model_name, model_id))
history_save_path = os.path.join(root_save_path, "history_%s_id_%s.csv"%(model_name, model_id))

dataset = LoadDataset(input_dim, batch_size_train, batch_size_test, model_id)
train_loader, val_loader, test_loader = dataset.caltech_256(dataset_path, split_ratio, dataset_name, save_indices_path)


model = models.resnet152(pretrained=pretrained).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)


best_val_loss = np.inf
patience = 10
count = 0
df = pd.DataFrame()
epoch = 0

while (epoch < n_epochs):
  epoch+=1
  print("Epoch: %s"%(epoch))
  result = {}
  train_results = runTrain(dataLoader=train_loader, 
  	model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, n_epochs=n_epochs, train=True)

  valid_results = runTrain(dataLoader=val_loader, 
  	model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, n_epochs=n_epochs, train=False)

  result.update(train_results), result.update(valid_results)

  scheduler.step()

  df = df.append(pd.Series(result), ignore_index=True)
  df.to_csv(history_save_path)

  if (result["val_loss"] < best_val_loss):
    best_val_loss = result["val_loss"]
    count = 0
    save_dict = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "epoch": epoch, "val_loss": result["val_loss"]}
    
    torch.save(save_dict, model_save_path)

  else:
    print("Current Patience: %s"%(count))
    count += 1

print("Stop! Patience is finished")
