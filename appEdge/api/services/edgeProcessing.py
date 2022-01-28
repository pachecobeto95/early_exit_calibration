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
	p_tar, nr_branch_edge, model_name = exp.exp_params["p_tar"], exp.exp_params["nr_branch"], model.model_params["model_name"]

	start = time.time()
	image_bytes = fileImg.read()

	#Starts measuring the inference time
	tensor_img = transform_image(image_bytes, model) #transform input data, which resizes the input image

	#Run the Early-exit dnn inference
	output, conf_list, class_list, isTerminate = ee_dnn_no_calib_inference(tensor_img, p_tar, nr_branch_edge)

	if (not isTerminate):
		response_request = sendToCloud(config.url_cloud_no_calib, output, conf_list, class_list)

	inference_time = time.time() - start

	if(response_request["status"] == "ok"):
		saveInferenceTime(inference_time,  p_tar, nr_branch_edge, model_name, isTerminate, calibration_type="no_calib")
	
	return response_request


def edgeOverallCalibInference(fileImg, p_tar, nr_branch_edge):
	response_request = {"status": "ok"}
	p_tar, nr_branch_edge, model_name = exp.exp_params["p_tar"], exp.exp_params["nr_branch"], model.model_params["model_name"]

	#This line reads the fileImg, obtaining pixel matrix.
	start = time.time()
	image_bytes = fileImg.read()

	#Starts measuring the inference time
	tensor_img = transform_image(image_bytes) #transform input data, which resizes the input image

	#Run the Early-exit dnn inference
	output, conf_list, class_list, isTerminate = ee_dnn_overall_calib_inference(tensor_img, p_tar, nr_branch_edge)

	if (not isTerminate):
		response_request = sendToCloud(config.url_cloud_overall_calib, output, conf_list, class_list)

	inference_time = time.time() - start
	return response_request

	if(response_request["status"] == "ok"):
		saveInferenceTime(inference_time,  p_tar, nr_branch_edge, model_name, isTerminate, calibration_type="overall_calib")
	
	return response_request


def edgeBranchesCalibInference(fileImg, p_tar, nr_branch_edge):
	response_request = {"status": "ok"}
	p_tar, nr_branch_edge, model_name = exp.exp_params["p_tar"], exp.exp_params["nr_branch"], model.model_params["model_name"]


	#This line reads the fileImg, obtaining pixel matrix.
	start = time.time()
	image_bytes = fileImg.read()

	#Starts measuring the inference time
	tensor_img = transform_image(image_bytes) #transform input data, which resizes the input image

	#Run the Early-exit dnn inference
	output, conf_list, class_list, isTerminate = ee_dnn_branches_calib_inference(tensor_img, p_tar, nr_branch_edge)

	if (not isTerminate):
		response_request = sendToCloud(config.url_cloud_branches_calib, output, conf_list, class_list)

	inference_time = time.time() - start

	return response_request
	
	if(response_request["status"] == "ok"):
		saveInferenceTime(inference_time,  p_tar, nr_branch_edge, model_name, isTerminate, calibration_type="branches_calib")
	
	return response_request

def saveInferenceTime(inference_time,  p_tar, nr_branch_edge, model_name, isTerminate, calibration_type):
	
	result = {"inference_time": inference_time,"p_tar": p_tar, "nr_branch_edge": nr_branch_edge,
	"early_inference": isTerminate, "calibration_type": calibration_type}
	
	result_path = os.path.join(config.RESULTS_INFERENCE_TIME_EDGE, "inference_time_results_%s.csv"%(model_name))

	if (not os.path.exists(result_path)):
		df = pd.DataFrame()
	else:
		df = pd.read_csv(result_path)	
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')] 
	
	df = df.append(pd.Series(result), ignore_index=True)
	df.to_csv(result_path)


#def edgeAllSamplesCalibInference(fileImg, p_tar, nr_branch_edge):
#	response_request = {"status": "ok"}

	#This line reads the fileImg, obtaining pixel matrix.
#	start = time.time()
#	image_bytes = fileImg.read()

	#Starts measuring the inference time
#	tensor_img = transform_image(image_bytes) #transform input data, which resizes the input image

	#Run the Early-exit dnn inference
#	output, conf_list, class_list, isTerminate = ee_dnn_all_samples_calib_inference(tensor_img, p_tar, nr_branch_edge)

#	if (not isTerminate):
#		response_request = sendToCloud(config.url_cloud_all_samples_calib, output, conf_list, class_list, p_tar, nr_branch_edge)

#	inference_time = time.time() - start
#	return response_request

#	if(response_request["status"] == "ok"):
#		saveInferenceTime(inference_time,  p_tar, nr_branch_edge, isTerminate)
	
#	return response_request



def ee_dnn_no_calib_inference(tensor_img, p_tar, nr_branch_edge):
	#model.ee_model.eval()
	with torch.no_grad():
		output, conf_list, class_list, isTerminate = model.ee_model.forwardEdgeNoCalibInference(tensor_img, p_tar, nr_branch_edge)

	return output, conf_list, class_list, isTerminate

def ee_dnn_overall_calib_inference(tensor_img, p_tar, nr_branch_edge):
	#model.ee_model.eval()
	with torch.no_grad():
		output, conf_list, class_list, isTerminate = model.ee_model.forwardEdgeOverallCalibInference(tensor_img, p_tar, nr_branch_edge)

	return output, conf_list, class_list, isTerminate

def ee_dnn_branches_calib_inference(tensor_img, p_tar, nr_branch_edge):
	#model.ee_model.eval()
	with torch.no_grad():
		output, conf_list, class_list, isTerminate = model.ee_model.forwardEdgeBranchesCalibInference(tensor_img, p_tar, nr_branch_edge)

	return output, conf_list, class_list, isTerminate

#def ee_dnn_all_samples_calib_inference(tensor_img, p_tar, nr_branch_edge):
#	model.ee_model.eval()

#	with torch.no_grad():
#		output, conf_list, class_list, isTerminate = model.ee_model.forwardEdgeAllSamplesCalibInference(tensor_img, p_tar, nr_branch_edge)

#	return output, conf_list, class_list, isTerminate


def sendToCloud(url, feature_map, conf_list, class_list):
	"""
	This functions sends output data from a partitioning layer from edge device to cloud server.
	This function also sends the info of partitioning layer to the cloud.
	Argments:

	feature_map (Tensor): output data from partitioning layer
	conf_list (list): this list contains the confidence obtained for each early exit during Early-exit DNN inference
	"""

	conf_list = [0 if math.isnan(x) else x for x in conf_list] if(np.nan in conf_list) else conf_list

	data = {'feature': feature_map.detach().cpu().numpy().tolist(), "conf": conf_list, "class_list": class_list}

	try:
		response = requests.post(url, json=data, timeout=config.timeout)
		response.raise_for_status()
		return {"status": "ok"}

	except Exception as e:
		print(e)
		return {"status": "error"}

