from flask import jsonify, session, current_app as app
import os, pickle, requests, sys, config, time
import numpy as np, json
import torchvision.models as models
import torch
import datetime
import time, io
import torchvision.transforms as transforms
from PIL import Image
#from .utils import load_model
from .utils import ModelLoad, transform_image, ExpLoad
import torchvision.models as models

model = ModelLoad()
exp = ExpLoad()


def onlyCloudProcessing(fileImg):
	#try:
	image_bytes = fileImg.read()
	response_request = {"status": "ok"}

	#Starts measuring the inference time
	tensor_img = transform_image(image_bytes) #transform input data, which means resize the input image

	#Run DNN inference
	output = only_cloud_dnn_inference_cloud(tensor_img)
	return {"status": "ok"}


def only_cloud_dnn_inference_cloud(x):
	model.ee_model.eval()
	with torch.no_grad():
		output = dnn_model_full(x.to(device).float())
	
	return output


def cloudNoCalibInference(feature, conf_list, class_list):
	p_tar, nr_branch_edge = exp.exp_params["p_tar"], exp.exp_params["nr_branch"]

	feature = torch.Tensor(feature).to(config.device).float()
	conf, infer_class = ee_dnn_no_calib_inference(feature, conf_list, class_list, p_tar, nr_branch_edge)
	return {"status": "ok"}

def cloudOverallCalibInference(feature, conf_list, class_list):
	p_tar, nr_branch_edge = exp.exp_params["p_tar"], exp.exp_params["nr_branch"]

	feature = torch.Tensor(feature).to(config.device).float()
	conf, infer_class = ee_dnn_overall_calib_inference(feature, conf_list, class_list, p_tar, nr_branch_edge)
	return {"status": "ok"}

def cloudBranchesCalibInference(feature, conf_list, class_list):
	p_tar, nr_branch_edge = exp.exp_params["p_tar"], exp.exp_params["nr_branch"]

	feature = torch.Tensor(feature).to(config.device).float()
	conf, infer_class = ee_dnn_branches_calib_inference(feature, conf_list, class_list, p_tar, nr_branch_edge)
	return {"status": "ok"}

#def cloudAllSamplesCalibInference(feature, conf_list, class_list, p_tar, nr_branch_edge):
#	feature = torch.Tensor(feature).to(config.device).float()
#	conf, infer_class = ee_dnn_all_samples_calib_inference(feature, conf_list, class_list, p_tar, nr_branch_edge)
#	return {"status": "ok"}


def ee_dnn_no_calib_inference(x, conf_list, class_list, p_tar, nr_branch_edge):
	#model.ee_model.eval()
	with torch.no_grad():
		conf, infer_class = model.ee_model.forwardNoCalibCloudInference(x, conf_list, class_list, p_tar, nr_branch_edge)
	return conf, infer_class

def ee_dnn_overall_calib_inference(x, conf_list, class_list, p_tar, nr_branch_edge):
	#model.ee_model.eval()
	with torch.no_grad():
		conf, infer_class = model.ee_model.forwardOverallCalibCloudInference(x, conf_list, class_list, p_tar, nr_branch_edge)
	return conf, infer_class

def ee_dnn_branches_calib_inference(x, conf_list, class_list, p_tar, nr_branch_edge):
	#model.ee_model.eval()
	with torch.no_grad():
		conf, infer_class = model.ee_model.forwardBranchesCalibCloudInference(x, conf_list, class_list, p_tar, nr_branch_edge)
	return conf, infer_class

#def ee_dnn_all_samples_calib_inference(x, conf_list, class_list, p_tar, nr_branch_edge):
#	model.ee_model.eval()
#	with torch.no_grad():
#		conf, infer_class = model.ee_model.forwardAllSamplesCalibCloudInference(x, conf_list, class_list, p_tar, nr_branch_edge)
#	return conf, infer_class




