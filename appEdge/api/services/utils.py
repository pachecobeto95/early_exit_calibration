import os, pickle, requests, sys, config, time, json, io
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from .early_exit_dnn import Early_Exit_DNN_CALTECH, Early_Exit_DNN_CIFAR
import pandas as pd

def transform_image(image_bytes, model):
	#imagenet_mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
	#imagenet_std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
	#my_transforms = transforms.Compose([transforms.Resize(model_params["input_shape"][1]),
	#	transforms.ToTensor(),
	#	transforms.Normalize(imagenet_mean, imagenet_std)])


	image = Image.open(io.BytesIO(image_bytes))
	return model.input_transformation(image).unsqueeze(0).to(config.device).float()


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

	def transform_input_configuration(self):

		if(self.model_params["dataset_name"]=="caltech256"):
			mean, std = [0.457342265910642, 0.4387686270106377, 0.4073427106250871], [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

			self.input_transformation = transforms.Compose([
				transforms.Resize(self.model_params["input_shape"][1]), 
				transforms.ToTensor(), 
				transforms.Normalize(mean = mean, std = std),
				])

		else:
			mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

			self.input_transformation = transforms.Compose([
				transforms.Resize(self.model_params["input_shape"][1]),
				transforms.RandomCrop(self.model_params["input_shape"][1], padding = 4),
				transforms.ToTensor(),
				transforms.Normalize(mean, std)])


	def load_model(self):

		model_name, n_classes = self.model_params["model_name"], self.model_params["n_classes"]
		n_branches, input_shape = self.model_params["n_branches"], self.model_params["input_shape"]
		model_id, dataset_name = self.model_params["model_id"], self.model_params["dataset_name"]
		pretrained = self.model_params["pretrained"]

		print(dataset_name)

		if(dataset_name == "caltech256"):

			self.ee_model = Early_Exit_DNN_CALTECH(model_name, n_classes, config.pretrained, n_branches, input_shape, config.exit_type, 
				config.device, config.disabled_branches, config.distribution)

			model_file_name = "ee_%s_branches_%s_id_%s.pth"%(model_name, n_branches, model_id)

		elif((dataset_name == "cifar100") or (dataset_name == "cifar10")):
			
			self.ee_model = Early_Exit_DNN_CIFAR(model_name, n_classes, pretrained, n_branches, input_shape, config.exit_type, config.device, 
				config.disabled_branches, config.distribution)

			model_file_name = "b_%s_early_exit_%s_id_1_%s_%s.pth"%(model_name,
				dataset_name, pretrained, self.model_params["weight_loss_type"])
		else:
			print("Error")

		model_path = os.path.join(config.edge_model_root_path, dataset_name, model_name, "models", model_file_name)
		print(model_path)
		self.ee_model.load_state_dict(torch.load(model_path, map_location=config.device)["model_state_dict"])
		self.ee_model.eval()


	def get_temperature(self, df, overall=False):

		df = df.where(pd.notnull(df), None)
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


		df_overall_calib = pd.read_csv(overall_calib_path)
		df_branches_calib = pd.read_csv(branches_calib_path)

		self.overall_temp = self.get_temperature(df_overall_calib, overall=True)
		self.temp_branches = self.get_temperature(df_branches_calib)

	def update_overall_temperature(self, p_tar):
		self.ee_model.temperature_overall = self.overall_temp.loc[p_tar].item()
		
	def update_branches_temperature(self, p_tar):
		self.ee_model.temperature_branches = self.temp_branches.loc[p_tar].values
		
	def update_all_samples_temperature(self, p_tar):
		self.ee_model.temperature_all_samples = self.temp_all_samples.loc[p_tar].values