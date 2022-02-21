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


def edgeNoCalibInference(fileImg, data_dict):

	#This line reads the fileImg, obtaining pixel matrix.
	response_request = {"status": "ok"}

	result_path = os.path.join(config.RESULTS_INFERENCE_TIME_EDGE, "inference_time_results_%s.csv"%(model.model_params["model_name"]))

	start = time.time()
	image_bytes = fileImg.read()

	#Starts measuring the inference time
	tensor_img = transform_image(image_bytes, model) #transform input data, which resizes the input image

	#Run the Early-exit dnn inference
	output, conf_list, class_list, isTerminate = ee_dnn_no_calib_inference(tensor_img, data_dict["p_tar"], data_dict["nr_branch"])

	if (not isTerminate):
		response_request = sendToCloud(config.url_cloud_no_calib, output, conf_list, class_list)

	inference_time = time.time() - start

	#if(response_request["status"] == "ok"):
	if((response_request["status"] == "ok") and (not data_dict["warmUp"])):
		saveInferenceTime(inference_time, class_list, data_dict, isTerminate, result_path, calibration_type="no_calib")
	
	return response_request


def edgeNoCalibInferenceOnlyEdge(fileImg, data_dict):

	#This line reads the fileImg, obtaining pixel matrix.
	response_request = {"status": "ok"}

	resultPath = os.path.join(config.RESULTS_INFERENCE_TIME_EDGE, "inference_time_results_%s_only_edge.csv"%(model.model_params["model_name"]))

	start = time.time()
	image_bytes = fileImg.read()

	#Starts measuring the inference time
	tensor_img = transform_image(image_bytes, model) #transform input data, which resizes the input image

	#Run the Early-exit dnn inference
	with torch.no_grad():
		output, conf_list, class_list, isTerminate = model.ee_model.forwardOnlyEdgeNoCalibInference(tensor_img, data_dict["p_tar"], data_dict["nr_branch"])


	inference_time = time.time() - start

	if((response_request["status"] == "ok") and (not data_dict["warmUp"])):
		saveInferenceTime(inference_time, class_list, data_dict, isTerminate, resultPath, calibration_type="no_calib")
	
	return response_request



def edgeOverallCalibInference(fileImg, data_dict):
	response_request = {"status": "ok"}

	result_path = os.path.join(config.RESULTS_INFERENCE_TIME_EDGE, "inference_time_results_%s.csv"%(model.model_params["model_name"]))

	#This line reads the fileImg, obtaining pixel matrix.
	start = time.time()
	image_bytes = fileImg.read()

	#Starts measuring the inference time
	tensor_img = transform_image(image_bytes, model) #transform input data, which resizes the input image

	#Run the Early-exit dnn inference
	output, conf_list, class_list, isTerminate = ee_dnn_overall_calib_inference(tensor_img, data_dict["p_tar"], data_dict["nr_branch"])

	if (not isTerminate):
		response_request = sendToCloud(config.url_cloud_overall_calib, output, conf_list, class_list)

	inference_time = time.time() - start

	if((response_request["status"] == "ok") and (not data_dict["warmUp"])):
		saveInferenceTime(inference_time, class_list, data_dict, isTerminate, result_path, calibration_type="overall_calib")
	
	return response_request


def edgeOverallCalibInferenceOnlyEdge(fileImg, data_dict):
	response_request = {"status": "ok"}

	resultPath = os.path.join(config.RESULTS_INFERENCE_TIME_EDGE, "inference_time_results_%s_only_edge.csv"%(model.model_params["model_name"]))

	#This line reads the fileImg, obtaining pixel matrix.
	start = time.time()
	image_bytes = fileImg.read()

	#Starts measuring the inference time
	tensor_img = transform_image(image_bytes, model) #transform input data, which resizes the input image

	#Run the Early-exit dnn inference
	with torch.no_grad():
		output, conf_list, class_list, isTerminate = model.ee_model.forwardOnlyEdgeOverallCalibInference(tensor_img, data_dict["p_tar"], data_dict["nr_branch"])


	inference_time = time.time() - start

	if((response_request["status"] == "ok") and (not data_dict["warmUp"])):
		saveInferenceTime(inference_time, class_list, data_dict, isTerminate, resultPath, calibration_type="overall_calib")
	
	return response_request



def edgeBranchesCalibInference(fileImg, data_dict):
	response_request = {"status": "ok"}

	result_path = os.path.join(config.RESULTS_INFERENCE_TIME_EDGE, "inference_time_results_%s.csv"%(model.model_params["model_name"]))


	#This line reads the fileImg, obtaining pixel matrix.
	start = time.time()
	image_bytes = fileImg.read()

	#Starts measuring the inference time
	tensor_img = transform_image(image_bytes, model) #transform input data, which resizes the input image

	#Run the Early-exit dnn inference
	output, conf_list, class_list, isTerminate = ee_dnn_branches_calib_inference(tensor_img, data_dict["p_tar"], data_dict["nr_branch"])

	if (not isTerminate):
		response_request = sendToCloud(config.url_cloud_branches_calib, output, conf_list, class_list)

	inference_time = time.time() - start
	
	if((response_request["status"] == "ok") and (not data_dict["warmUp"])):
		saveInferenceTime(inference_time, class_list,  data_dict, isTerminate, result_path, calibration_type="branches_calib")
	
	return response_request

def edgeBranchesCalibInferenceOnlyEdge(fileImg, data_dict):
	response_request = {"status": "ok"}
	
	resultPath = os.path.join(config.RESULTS_INFERENCE_TIME_EDGE, "inference_time_results_%s_only_edge.csv"%(model.model_params["model_name"]))


	#This line reads the fileImg, obtaining pixel matrix.
	start = time.time()
	image_bytes = fileImg.read()

	#Starts measuring the inference time
	tensor_img = transform_image(image_bytes, model) #transform input data, which resizes the input image

	#Run the Early-exit dnn inference
	with torch.no_grad():
		output, conf_list, class_list, isTerminate = model.ee_model.forwardOnlyEdgeBranchesCalibInference(tensor_img,data_dict["p_tar"], data_dict["nr_branch"])

	inference_time = time.time() - start
	
	if((response_request["status"] == "ok") and (not data_dict["warmUp"])):
		saveInferenceTime(inference_time, class_list,  data_dict, isTerminate, resultPath, calibration_type="branches_calib")
	
	return response_request


def saveInferenceTime(inference_time, inf_class,  data_dict, isTerminate, result_path, calibration_type):
	
	correct = 1 if(inf_class == data_dict["target"]) else 0

	result = {"id": data_dict["id"], "inference_time": inference_time,"p_tar": data_dict["p_tar"], 
	"nr_branch_edge": data_dict["nr_branch"], "early_inference": isTerminate, "calibration_type": calibration_type, "correct": correct}
	
	df = pd.DataFrame([result])
	df.to_csv(result_path, mode='a', header=not os.path.exists(result_path))


def ee_dnn_no_calib_inference(tensor_img, p_tar, nr_branch_edge):

	with torch.no_grad():
		output, conf_list, class_list, isTerminate = model.ee_model.forwardEdgeNoCalibInference(tensor_img, p_tar, nr_branch_edge)

	return output, conf_list, class_list, isTerminate

def ee_dnn_overall_calib_inference(tensor_img, p_tar, nr_branch_edge):

	with torch.no_grad():
		output, conf_list, class_list, isTerminate = model.ee_model.forwardEdgeOverallCalibInference(tensor_img, p_tar, nr_branch_edge)

	return output, conf_list, class_list, isTerminate

def ee_dnn_branches_calib_inference(tensor_img, p_tar, nr_branch_edge):

	with torch.no_grad():
		output, conf_list, class_list, isTerminate = model.ee_model.forwardEdgeBranchesCalibInference(tensor_img, p_tar, nr_branch_edge)

	return output, conf_list, class_list, isTerminate

def sendToCloud(url, feature_map, conf_list, class_list):
	"""
	This functions sends output data from a partitioning layer from edge device to cloud server.
	This function also sends the info of partitioning layer to the cloud.
	Argments:

	feature_map (Tensor): output data from partitioning layer
	conf_list (list): this list contains the confidence obtained for each early exit during Early-exit DNN inference
	"""

	data = {'feature': feature_map.detach().cpu().numpy().tolist(), "conf": conf_list, "class_list": class_list}

	try:
		response = requests.post(url, json=data, timeout=config.timeout)
		response.raise_for_status()
		return {"status": "ok"}

	except Exception as e:
		print(e)
		return {"status": "error"}

