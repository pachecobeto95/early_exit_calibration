import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_dir(history_path, model_path):
	if not os.path.exists(model_path):
		
		os.makedirs(model_path)
		os.makedirs(history_path)


def verify_stop_condition(count, epoch, args):
	stop_condition = count <= args.patience if(args.pretrained) else epoch <= args.n_epochs
	return stop_condition
