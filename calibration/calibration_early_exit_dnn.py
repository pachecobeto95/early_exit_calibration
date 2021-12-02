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


class ModelOverallCalibration(nn.Module):

  def __init__(self, model, device, modelPath, lr=0.01, max_iter=50):
    super(ModelOverallCalibration, self).__init__()
    
    self.model = model
    self.device = device
    self.temperature_overall = nn.Parameter(1.5*torch.ones(1).to(self.device))
    self.lr = lr
    self.max_iter = max_iter

    self.model.load_state_dict(torch.load(modelPath, map_location=device)["model_state_dict"])

  def temperature_scale(self, logits):
    temperature = self.temperature_overall.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / temperature

  def forward(self, x):
    logits = self.model(x)
    return self.temperature_scale(logits)


  # This function probably should live outside of this class, but whatever
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
        print(confs, exit_branch)

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

    return self
