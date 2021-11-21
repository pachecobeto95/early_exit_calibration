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
	my_transforms = transforms.Compose([transforms.Resize(config.input_dim),
		transforms.CenterCrop(config.input_resize),
		transforms.ToTensor(),
		transforms.Normalize(imagenet_mean, imagenet_std)])

	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0).to(config.device).float()


class ModelLoad():
	def __init__(self):
		self.model_params = None
		self.n_exits = config.n_branches + 1

	def load_model(self):
		self.ee_model = Early_Exit_DNN(self.model_params["model_name"], self.model_params["n_classes"], config.pretrained, 
			config.n_branches, config.input_shape, config.exit_type, config.device, distribution=config.distribution)

		model_path = os.path.join(config.edge_model_root_path, self.model_params["dataset_name"], self.model_params["model_name"], 
			"models", "ee_%s_branches_%s_id_%s.pth"%(self.model_params["model_name"], config.n_branches, config.model_id))
		
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
		
		return df[select_temp_branches].where(pd.notnull(df[select_temp_branches]), None)

	def load_temperature(self):
		overall_calib_temp_path = os.path.join(config.edge_model_root_path, self.model_params["dataset_name"], 
			self.model_params["model_name"], "temperature", "temp_overall_id_%s.csv"%(config.model_id))
		
		branches_calib_temp_path = os.path.join(config.edge_model_root_path, self.model_params["dataset_name"], 
			self.model_params["model_name"], "temperature", "temp_branches_id_%s.csv"%(config.model_id))

		all_samples_calib_temp_path = os.path.join(config.edge_model_root_path, self.model_params["dataset_name"], 
			self.model_params["model_name"], "temperature", "temp_all_samples_id_%s.csv"%(config.model_id))

		df_overall_calib = pd.read_csv(overall_calib_temp_path)
		df_branches_calib = pd.read_csv(branches_calib_temp_path)
		df_all_samples_calib = pd.read_csv(all_samples_calib_temp_path)

		self.overall_temp = self.get_temperature(df_overall_calib, overall=True)
		self.temp_branches = self.get_temperature(df_branches_calib)
		self.temp_all_samples = self.get_temperature(df_all_samples_calib)

	def update_overall_temperature(self, p_tar):
		self.ee_model.temperature_overall = self.overall_temp.loc[p_tar].item()
		
	def update_branches_temperature(self, p_tar):
		print(self.temp_branches.loc[p_tar])
		self.ee_model.temperature_branches = self.temp_branches.loc[p_tar].values
		
	def update_all_samples_temperature(self, p_tar):
		self.ee_model.temperature_all_samples = self.temp_all_samples.loc[p_tar].values


