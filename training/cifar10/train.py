import torchvision
import os, sys, time, math
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
			output_list, conf_list, class_list = model(data, training=True)
			loss = 0
			for j, (output, inf_class, weight) in enumerate(zip(output_list, class_list, loss_weights), 1):
				loss += weight*criterion(output, target)
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



def testMainModel(model, testLoader, device):

	model.eval()

	conf_list, infered_class_list, target_list, correct_list = [], [], [], []
	softmax = nn.Softmax(dim=1)
	
	with torch.no_grad():
		for i, (data, target) in tqdm(enumerate(testLoader, 1)):

			data, target = data.to(device), target.to(device)

			output = model(data)
			conf, infered_class = torch.max(softmax(output), 1)
			correct = infered_class.eq(target.view_as(infered_class)).sum().item()

			conf_list.append(conf), infered_class_list.append(infered_class)
			target_list.append(target.item()), correct_list.append(correct)
			id_list.append(i)

			del data, target
			torch.cuda.empty_cache()


	conf_list = torch.cat(conf_list).to(device)
	infered_class_list = torch.cat(infered_class_list).to(device)

	conf_list = np.array(conf_list.cpu().detach().numpy())
	infered_class_list = np.array(infered_class_list.cpu().detach().numpy())
	correct_list = np.array(correct_list)
	target_list = np.array(target_list)

	acc = sum(correct_list)/len(correct_list)
	print("Acc: %s"%(acc))

	results = {"conf": conf_list, "target": target_list, "infered_class": infered_class_list, 
	"correct": correct_list, "id": id_list}

	return results