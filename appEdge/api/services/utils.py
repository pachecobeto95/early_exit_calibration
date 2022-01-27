import os, pickle, requests, sys, config, time, json, io
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from .early_exit_dnn import Early_Exit_DNN
import pandas as pd

def transform_image(image_bytes):
	imagenet_mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
	imagenet_std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
	my_transforms = transforms.Compose([transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize(imagenet_mean, imagenet_std)])

	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0).to(config.device).float()

class ExpLoad:
	def __init__(self):
		self._exp_params = {}
	# using property decorator
	# a getter function
	@property
	def exp_params(self):
		return self._exp_params

	# a setter function
	@exp_params.setter
	def exp_params(self, params):
		self._exp_params = params


class ModelLoad():
	def __init__(self):
		self.model_params = None
		self.n_exits = config.n_branches + 1

	def load_model(self):
		self.ee_model = Early_Exit_DNN(self.model_params["model_name"], self.model_params["n_classes"], config.pretrained, 
			self.model_params["n_branches"], self.model_params["input_shape"], config.exit_type, config.device, 
			distribution=config.distribution)

		if(self.model_params["dataset_name"] == "caltech256"):
			model_file_name = "ee_%s_branches_%s_id_%s.pth"%(self.model_params["model_name"], 
				self.model_params["n_branches"], config.model_id_dict[self.model_params["model_name"]])

		elif((self.model_params["dataset_name"] == "cifar100") or (self.model_params["dataset_name"] == "cifar10")):
			
			model_file_name = "b_%s_early_exit_%s_id_1_%s_%s.pth"%(self.model_params["model_name"],
				self.model_params["dataset_name"], 
				self.model_params["pretrained"], self.model_params["weight_loss_type"])
		else:
			print("Error")

		model_path = os.path.join(config.edge_model_root_path, self.model_params["dataset_name"], 
			self.model_params["model_name"], 
			"models", model_file_name)
		
		self.ee_model.load_state_dict(torch.load(model_path, map_location=config.device)["model_state_dict"])


	def get_temperature(self, df, overall=False):
		#print(df)
		df = df.where(pd.notnull(df), None)
		#print(df)
		df.set_index("p_tar", inplace=True)
		if(overall):
			select_temp_branches = ["temperature"]
		else:
			select_temp_branches = ["temperature_branch_%s"%(i) for i in range(1, self.n_exits+1)]
		
		df_temp = df[select_temp_branches]

		return df_temp.where(pd.notnull(df_temp), None)

	def load_temperature(self):
		#./appEdge/api/services/models/caltech256/mobilenet/temperature/
		temp_root_path = os.path.join(config.edge_model_root_path, self.model_params["dataset_name"], self.model_params["model_name"],
			"temperature")

		if(self.model_params["dataset_name"] == "caltech256"):
			overall_calib_path = os.path.join(temp_root_path, "temp_overall_id_%s.csv"%(self.model_params["model_id"]))
			branches_calib_path = os.path.join(temp_root_path, "temp_branches_id_%s.csv"%(self.model_params["model_id"]))			

		elif(self.model_params["dataset_name"] == "cifar100"):
			overall_calib_path = os.path.join(temp_root_path, 
				"overall_temperature_%s_early_exit_cifar100_id_1_%s.csv"%(self.model_params["model_id"], self.model_params["pretrained"]))
			
			branches_calib_path = os.path.join(temp_root_path, 
				"branches_temperature_%s_early_exit_cifar100_id_1_%s.csv"%(self.model_params["model_id"], self.model_params["pretrained"]))
			
		else:
			print("Error")


		df_overall_calib = pd.read_csv(overall_calib_temp_path)
		df_branches_calib = pd.read_csv(branches_calib_temp_path)

		self.overall_temp = self.get_temperature(df_overall_calib, overall=True)
		self.temp_branches = self.get_temperature(df_branches_calib)

	def update_overall_temperature(self, p_tar):
		self.ee_model.temperature_overall = self.overall_temp.loc[p_tar].item()
		
	def update_branches_temperature(self, p_tar):
		self.ee_model.temperature_branches = self.temp_branches.loc[p_tar].values
		
	def update_all_samples_temperature(self, p_tar):
		self.ee_model.temperature_all_samples = self.temp_all_samples.loc[p_tar].values