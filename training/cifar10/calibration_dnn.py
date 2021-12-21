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



class _ECELoss(nn.Module):

    def __init__(self, n_bins=10):
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
              #print(bin_lower, bin_upper, ece)

        return ece


class MainModelCalibration(nn.Module):
    def __init__(self, model, device, modelPath, saveTempPath, lr, max_iter):
        super(MainModelCalibration, self).__init__()

        self.model = model
        self.saveTempPath = saveTempPath
        self.lr = lr
        self.max_iter = max_iter
        self.device = device
        self.temperature = nn.Parameter((1.5*torch.ones(1)).to(self.device))

        self.model.load_state_dict(torch.load(modelPath, map_location=device)["model_state_dict"])

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)

    def save_temperature(self, result):
        # This function probably should live outside of this class, but whatever
        df = pd.DataFrame()
    
        df = df.append(pd.Series(result), ignore_index=True)
        df.to_csv(self.saveTempPath)
  
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss().to(self.device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        self.model.eval()

        with torch.no_grad():
            for data, label in tqdm(valid_loader):
                data, label = data.to(self.device), label.to(self.device)  
          
                logits = self.model(data)
                logits_list.append(logits), labels_list.append(label)
      
        logits = torch.cat(logits_list).to(self.device)
        labels = torch.cat(labels_list).to(self.device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)

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
        print('After temperature - NLL: %.3f, ECE: %.3f'%(after_temperature_nll, after_temperature_ece))

        result = {"before_nll": before_temperature_nll, "after_nll": after_temperature_nll,
        "before_ece": before_temperature_ece, "after_ece": after_temperature_ece,
        "temperature": self.temperature.item()}

        self.save_temperature(result)
        return self



class ModelOverallCalibration(nn.Module):

  def __init__(self, model, device, modelPath, saveTempPath, lr=0.01, max_iter=1000):
    super(ModelOverallCalibration, self).__init__()
    
    self.model = model
    self.device = device
    self.temperature_overall = nn.Parameter((1.*torch.ones(1)).to(self.device))
    self.lr = lr
    self.max_iter = max_iter
    self.saveTempPath = saveTempPath

    self.model.load_state_dict(torch.load(modelPath, map_location=device)["model_state_dict"])

  def temperature_scale(self, logits):
    temperature = self.temperature_overall.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / temperature

  def forwardOverall(self, x):
    return self.model.forwardOverallCalibration(x, self.temperature_overall)


  def save_temperature(self, result):
    df = pd.DataFrame()
    
    df = df.append(pd.Series(result), ignore_index=True)
    df.to_csv(self.saveTempPath)
  
  def set_temperature(self, valid_loader, p_tar):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
        
    nll_criterion = nn.CrossEntropyLoss().to(self.device)
    ece_criterion = _ECELoss().to(self.device)

    # First: collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []
    self.model.eval()
    with torch.no_grad():
      for data, label in tqdm(valid_loader):
        data, label = data.to(self.device), label.to(self.device)  
          
        logits, confs, _, exit_branch = self.model.forwardEval(data, p_tar)

        logits_list.append(logits), labels_list.append(label)
      
    logits = torch.cat(logits_list).to(self.device)
    labels = torch.cat(labels_list).to(self.device)

    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(logits, labels).item()
    before_temperature_ece = ece_criterion(logits, labels).item()
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([self.temperature_overall], lr=self.lr, max_iter=self.max_iter)

    def eval():
      optimizer.zero_grad()
      loss = nll_criterion(self.temperature_scale(logits), labels)
      loss.backward()
      return loss
    optimizer.step(eval)

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
    after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
    print('Optimal temperature: %.3f' % self.temperature_overall.item())
    print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

    result = {"p_tar": round(p_tar, 2), "before_nll": before_temperature_nll, "after_nll": after_temperature_nll,
    "before_ece": before_temperature_ece, "after_ece": after_temperature_ece,
    "temperature": self.temperature_overall.item()}

    self.save_temperature(result)

    return self


class ModelBranchesCalibrationAlternative(nn.Module):
  def __init__(self, model, device, modelPath, saveTempPath, lr=0.01, max_iter=50):
    super(ModelBranchesCalibrationAlternative, self).__init__()

    self.model = model
    self.device = device
    self.n_exits = model.n_branches + 1

    self.temperature_branches = [nn.Parameter((1.5*torch.ones(1)).to(self.device)) for i in range(self.n_exits)]
    self.lr = lr
    self.max_iter = max_iter
    self.saveTempPath = saveTempPath

    self.model.load_state_dict(torch.load(modelPath, map_location=device)["model_state_dict"])

  def forwardBranchesCalibration(self, x):
    return self.model.forwardBranchesCalibration(x, self.temperature_branches)
  
  def temperature_scale_branches(self, logits):
    temperature = self.temperature_branch.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / temperature

  def save_temperature(self, result):

    df = pd.DataFrame()    
    df = df.append(pd.Series(result), ignore_index=True)
    df.to_csv(self.saveTempPath)
  
  def set_temperature(self, valid_loader, p_tar):

    nll_criterion = nn.CrossEntropyLoss().to(self.device)
    ece = _ECELoss()

    logits_list = [[] for i in range(self.n_exits)]
    labels_list = [[] for i in range(self.n_exits)]
    idx_sample_exit_list = [[] for i in range(self.n_exits)]
    before_temperature_nll_list, after_temperature_nll_list = [], []
    before_ece_list, after_ece_list = [], []
    logits_total_list, target_total_list = [], []

    error_measure_dict = {"p_tar": p_tar}

    self.model.eval()

    with torch.no_grad():
      for i, (data, target) in tqdm(enumerate(valid_loader)):

        data, target = data.to(self.device), target.to(self.device)
        
        logits, _, inf_class, exit_branch = self.model(data, p_tar, training=False)

        logits_list[exit_branch].append(logits)
        labels_list[exit_branch].append(target)
        idx_sample_exit_list[exit_branch].append(i)
        logits_total_list.append(logits), target_total_list.append(target)

    
    indices = []

    for n in range(self.n_exits):
      indices = indices + idx_sample_exit_list[n]
      new_indices = np.setdiff1d(idx_sample_exit_list, indices)
      print(new_indices.shape)

      logits_list[n] = torch.index_select(torch.cat(logits_total_list), 0, torch.tensor([1, 5]))
      labels_list[n] = torch.index_select(torch.cat(target_total_list), 0, new_indices)
    
    for i in range(self.n_exits):
      print("Exit: %s"%(i+1))

      if (len(logits_list[i]) == 0):
        before_temperature_nll_list.append(None), after_temperature_nll_list.append(None)
        before_ece_list.append(None), after_ece_list.append(None)
        continue

      self.temperature_branch = nn.Parameter((torch.ones(1)*1.).to(self.device))

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

      
      self.temperature_branches[i] = self.temperature_branch

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
    self.save_temperature(error_measure_dict)
    return self


class ModelBranchesCalibration(nn.Module):

  def __init__(self, model, device, modelPath, saveTempPath, lr=0.01, max_iter=50):
    super(ModelBranchesCalibration, self).__init__()
    
    self.model = model
    self.device = device
    self.n_exits = model.n_branches + 1

    self.temperature_branches = [nn.Parameter((1.5*torch.ones(1)).to(self.device)) for i in range(self.n_exits)]
    self.lr = lr
    self.max_iter = max_iter
    self.saveTempPath = saveTempPath

    self.model.load_state_dict(torch.load(modelPath, map_location=device)["model_state_dict"])


  def forwardBranchesCalibration(self, x):
    return self.model.forwardBranchesCalibration(x, self.temperature_branches)
  
  def temperature_scale_branches(self, logits):
    temperature = self.temperature_branch.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / temperature

  def save_temperature(self, result):

    df = pd.DataFrame()    
    df = df.append(pd.Series(result), ignore_index=True)
    df.to_csv(self.saveTempPath)
  
  def set_temperature(self, valid_loader, p_tar):

    nll_criterion = nn.CrossEntropyLoss().to(self.device)
    ece = _ECELoss()

    logits_list = [[] for i in range(self.n_exits)]
    labels_list = [[] for i in range(self.n_exits)]
    idx_sample_exit_list = [[] for i in range(self.n_exits)]
    before_temperature_nll_list, after_temperature_nll_list = [], []
    before_ece_list, after_ece_list = [], []

    error_measure_dict = {"p_tar": p_tar}

    self.model.eval()

    with torch.no_grad():
      for (data, target) in tqdm(valid_loader):

        data, target = data.to(self.device), target.to(self.device)
        
        logits, _, inf_class, exit_branch = self.model(data, p_tar, training=False)

        logits_list[exit_branch].append(logits)
        labels_list[exit_branch].append(target)


    for i in range(self.n_exits):
      print("Exit: %s"%(i+1))

      if (len(logits_list[i]) == 0):
        before_temperature_nll_list.append(None), after_temperature_nll_list.append(None)
        before_ece_list.append(None), after_ece_list.append(None)
        continue

      self.temperature_branch = nn.Parameter((torch.ones(1)*1.).to(self.device))

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

      
      self.temperature_branches[i] = self.temperature_branch

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
    self.save_temperature(error_measure_dict)
    return self

class ModelAllSamplesCalibration(nn.Module):

  def __init__(self, model, device, modelPath, saveTempPath, lr=0.01, max_iter=1000):
    super(ModelAllSamplesCalibration, self).__init__()

    self.model = model
    self.device = device
    self.n_exits = model.n_branches + 1

    self.temperature_branches = [nn.Parameter((1.*torch.ones(1)).to(self.device)) for i in range(self.n_exits)]
    self.lr = lr
    self.max_iter = max_iter
    self.saveTempPath = saveTempPath

    self.model.load_state_dict(torch.load(modelPath, map_location=device)["model_state_dict"])

  def forwardBranchesCalibration(self, x):
    return self.model.forwardBranchesCalibration(x, self.temperature_branches)
  
  def temperature_scale_branches(self, logits):
    return torch.div(logits, self.temperature_branch)

  def save_temperature(self, result):

    df = pd.DataFrame()    
    df = df.append(pd.Series(result), ignore_index=True)
    df.to_csv(self.saveTempPath)
  
  def set_temperature(self, val_loader, p_tar):

    nll_criterion = nn.CrossEntropyLoss().to(self.device)
    ece = _ECELoss()

    logits_list = [[] for i in range(self.n_exits)]
    labels_list = [[] for i in range(self.n_exits)]

    before_ece_list, after_ece_list = [], []    
    before_temperature_nll_list, after_temperature_nll_list = [], []

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

      #if (len(logits_list[i]) == 0):
      #  before_temperature_nll_list.append(None), after_temperature_nll_list.append(None)
      #  before_ece_list.append(None), after_ece_list.append(None)
      #  continue

      self.temperature_branch = nn.Parameter((torch.ones(1)*1.5).to(self.device))
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

    self.save_temperature(error_measure_dict)

    return self

def calibratingEEModels(model, val_loader, p_tar, device, model_path, temperaturePath, args):


    branches_model = ModelBranchesCalibrationAlternative(model, device, model_path, temperaturePath["overall_calib"], args.lr, args.max_iter)
    branches_model.set_temperature(val_loader, p_tar)

    overall_model = ModelOverallCalibration(model, device, model_path, temperaturePath["overall_calib"], args.lr, args.max_iter)
    overall_model.set_temperature(val_loader, p_tar)

    #branches_model = ModelBranchesCalibration(model, device, model_path, temperaturePath["branches_calib"], args.lr, args.max_iter)
    #branches_model.set_temperature(val_loader, p_tar)

    all_samples_model = ModelAllSamplesCalibration(model, device, model_path, temperaturePath["all_samples_calib"], args.lr, args.max_iter)
    all_samples_model.set_temperature(val_loader, p_tar)

    return {"calib_overall": overall_model, "calib_branches": branches_model, "calib_branches_all_samples": all_samples_model}




