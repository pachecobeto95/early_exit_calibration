from flask import jsonify, session, current_app as app
import os, pickle, requests, sys, config, time, math
import numpy as np, json
import torchvision.models as models
import torch
import datetime
import time, io
import torchvision.transforms as transforms
from PIL import Image
from .utils import ModelLoad, transform_image, ExpLoad
import pandas as pd


model = ModelLoad()
exp = ExpLoad()


def edgeNoCalibInference(fileImg):

	#This line reads the fileImg, obtaining pixel matrix.
	response_request = {"status": "ok"}
	p_tar, nr_branch_edge = exp.exp_params["p_tar"], exp.exp_params["nr_branch"]

	start = time.time()
	image_bytes = fileImg.read()

	#Starts measuring the inference time
	tensor_img = transform_image(image_bytes) #transform input data, which resizes the input image

	#Run the Early-exit dnn inference
	output, conf_list, class_list, isTerminate = ee_dnn_no_calib_inference(tensor_img, p_tar, nr_branch_edge)
	return response_request

	if (not isTerminate):
		response_request = sendToCloud(config.url_cloud_no_calib, output, conf_list, class_list, p_tar, nr_branch_edge)

	inference_time = time.time() - start
	return response_request

	if(response_request["status"] == "ok"):
		saveInferenceTime(inference_time,  p_tar, nr_branch_edge, isTerminate)
	
	return response_request


def edgeOverallCalibInference(fileImg, p_tar, nr_branch_edge):
	response_request = {"status": "ok"}
	p_tar, nr_branch_edge = exp.exp_params["p_tar"], exp.exp_params["nr_branch"]

	#This line reads the fileImg, obtaining pixel matrix.
	start = time.time()
	image_bytes = fileImg.read()

	#Starts measuring the inference time
	tensor_img = transform_image(image_bytes) #transform input data, which resizes the input image

	#Run the Early-exit dnn inference
	output, conf_list, class_list, isTerminate = ee_dnn_overall_calib_inference(tensor_img, p_tar, nr_branch_edge)

	if (not isTerminate):
		response_request = sendToCloud(config.url_cloud_overall_calib, output, conf_list, class_list, p_tar, nr_branch_edge)

	inference_time = time.time() - start
	return response_request

	if(response_request["status"] == "ok"):
		saveInferenceTime(inference_time,  p_tar, nr_branch_edge, isTerminate)
	
	return response_request


def edgeBranchesCalibInference(fileImg, p_tar, nr_branch_edge):
	response_request = {"status": "ok"}

	#This line reads the fileImg, obtaining pixel matrix.
	start = time.time()
	image_bytes = fileImg.read()

	#Starts measuring the inference time
	tensor_img = transform_image(image_bytes) #transform input data, which resizes the input image

	#Run the Early-exit dnn inference
	output, conf_list, class_list, isTerminate = ee_dnn_branches_calib_inference(tensor_img, p_tar, nr_branch_edge)

	if (not isTerminate):
		response_request = sendToCloud(config.url_cloud_branches_calib, output, conf_list, class_list, p_tar, nr_branch_edge)

	inference_time = time.time() - start

	return response_request
	
	if(response_request["status"] == "ok"):
		saveInferenceTime(inference_time,  p_tar, nr_branch_edge, isTerminate)
	
	return response_request


def edgeAllSamplesCalibInference(fileImg, p_tar, nr_branch_edge):
	response_request = {"status": "ok"}

	#This line reads the fileImg, obtaining pixel matrix.
	start = time.time()
	image_bytes = fileImg.read()

	#Starts measuring the inference time
	tensor_img = transform_image(image_bytes) #transform input data, which resizes the input image

	#Run the Early-exit dnn inference
	output, conf_list, class_list, isTerminate = ee_dnn_all_samples_calib_inference(tensor_img, p_tar, nr_branch_edge)

	if (not isTerminate):
		response_request = sendToCloud(config.url_cloud_all_samples_calib, output, conf_list, class_list, p_tar, nr_branch_edge)

	inference_time = time.time() - start
	return response_request

	if(response_request["status"] == "ok"):
		saveInferenceTime(inference_time,  p_tar, nr_branch_edge, isTerminate)
	
	return response_request



def ee_dnn_no_calib_inference(tensor_img, p_tar, nr_branch_edge):
	model.ee_model.eval()

	with torch.no_grad():
		output, conf_list, class_list, isTerminate = model.ee_model.forwardEdgeNoCalibInference(tensor_img, p_tar, nr_branch_edge)

	return output, conf_list, class_list, isTerminate

def ee_dnn_overall_calib_inference(tensor_img, p_tar, nr_branch_edge):
	model.ee_model.eval()

	with torch.no_grad():
		output, conf_list, class_list, isTerminate = model.ee_model.forwardEdgeOverallCalibInference(tensor_img, p_tar, nr_branch_edge)

	return output, conf_list, class_list, isTerminate

def ee_dnn_branches_calib_inference(tensor_img, p_tar, nr_branch_edge):
	model.ee_model.eval()

	with torch.no_grad():
		output, conf_list, class_list, isTerminate = model.ee_model.forwardEdgeBranchesCalibInference(tensor_img, p_tar, nr_branch_edge)

	return output, conf_list, class_list, isTerminate

def ee_dnn_all_samples_calib_inference(tensor_img, p_tar, nr_branch_edge):
	model.ee_model.eval()

	with torch.no_grad():
		output, conf_list, class_list, isTerminate = model.ee_model.forwardEdgeAllSamplesCalibInference(tensor_img, p_tar, nr_branch_edge)

	return output, conf_list, class_list, isTerminate


def sendToCloud(url, feature_map, conf_list, class_list, p_tar, nr_branch_edge):
	"""
	This functions sends output data from a partitioning layer from edge device to cloud server.
	This function also sends the info of partitioning layer to the cloud.
	Argments:

	feature_map (Tensor): output data from partitioning layer
	conf_list (list): this list contains the confidence obtained for each early exit during Early-exit DNN inference
	"""

	conf_list = [0 if math.isnan(x) else x for x in conf_list] if(np.nan in conf_list) else conf_list
	print(conf_list)

	data = {'feature': feature_map.detach().cpu().numpy().tolist(), "conf": conf_list, "p_tar": p_tar, 
	"nr_branch_edge": nr_branch_edge, "class_list": class_list}
	#print(conf_list, class_list)
	#data = {'feature': feature_map.detach().cpu().numpy().tolist(), "conf": [1], "p_tar": 1, 
	#"nr_branch_edge": 1, "class_list": [1]}

	try:
		response = requests.post(url, json=data, timeout=config.timeout)
		response.raise_for_status()
		return {"status": "ok"}

	except Exception as e:
		print(e)
		return {"status": "error"}

