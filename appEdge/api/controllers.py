from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
import json, os, time, sys, config, requests
from .services import edgeProcessing
from .services.edgeProcessing import model, exp

api = Blueprint("api", __name__, url_prefix="/api")


@api.route("/edge/modelConfiguration", methods=["POST"])
def edgeModelConfiguration():

	data = request.json
	model.model_params = data
	model.load_model()
	model.load_temperature()
	model.transform_input_configuration()
	return jsonify({"status": "ok"}), 200


@api.route("/edge/expConfiguration", methods=["POST"])
def edgeExpConfiguration():
	exp.exp_params = request.json
	return jsonify({"status": "ok"}), 200

# Define url for the user send the image
@api.route('/edge/edgeInference', methods=["POST"])
def edge_inference():
	"""
	This function receives an image from user or client with smartphone at the edge device into smart sity context
	"""	
	fileImg = request.files['img']

	#This functions process the DNN inference

	result = edgeProcessing.edgeInference(fileImg)


	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500

@api.route('edge/edgeNoCalibInference', methods=["POST"])
def edge_inference_no_calib():

	fileImg = request.files['img']
	data_dict = json.load(request.files['data'])

	#result = edgeProcessing.edgeNoCalibInference(fileImg)
	result = edgeProcessing.edgeNoCalibInference(fileImg, data_dict)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200
	else:
		return jsonify(result), 500

@api.route('edge/edgeNoCalibInferenceOnlyEdge', methods=["POST"])
def edge_inference_no_calib_only_edge():

	fileImg = request.files['img']
	data_dict = json.load(request.files['data'])

	#result = edgeProcessing.edgeNoCalibInference(fileImg)
	result = edgeProcessing.edgeNoCalibInferenceOnlyEdge(fileImg, data_dict)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200
	else:
		return jsonify(result), 500


@api.route('edge/edgeNoCalibInferenceOnlyEdgeStandardDNN', methods=["POST"])
def edge_inference_no_calib_only_edge_standard_dnn():

	fileImg = request.files['img']
	data_dict = json.load(request.files['data'])

	result = edgeProcessing.edgeNoCalibInferenceOnlyEdgeStandardDNN(fileImg, data_dict)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200
	else:
		return jsonify(result), 500


@api.route('edge/edgeOverallCalibInference', methods=["POST"])
def edge_inference_overall_calib():

	fileImg = request.files['img']
	data_dict = json.load(request.files['data'])

	model.update_overall_temperature(config.p_tar_calib)

	#result = edgeProcessing.edgeOverallCalibInference(fileImg)
	result = edgeProcessing.edgeOverallCalibInference(fileImg, data_dict)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200
	else:
		return jsonify(result), 500

@api.route('edge/edgeOverallCalibInferenceOnlyEdge', methods=["POST"])
def edge_inference_overall_calib_only_edge():

	fileImg = request.files['img']
	data_dict = json.load(request.files['data'])

	model.update_overall_temperature(config.p_tar_calib)

	#result = edgeProcessing.edgeOverallCalibInference(fileImg)
	result = edgeProcessing.edgeOverallCalibInferenceOnlyEdge(fileImg, data_dict)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200
	else:
		return jsonify(result), 500


@api.route('edge/edgeOverallCalibInferenceOnlyEdgeStandardDNN', methods=["POST"])
def edge_inference_overall_calib_only_edge_standard_dnn():

	fileImg = request.files['img']
	data_dict = json.load(request.files['data'])

	model.update_overall_temperature(config.p_tar_calib)

	result = edgeProcessing.edgeOverallCalibInferenceOnlyEdgeStandardDNN(fileImg, data_dict)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200
	else:
		return jsonify(result), 500


@api.route('edge/edgeBranchesCalibInference', methods=["POST"])
def edge_inference_branches_calib():

	fileImg = request.files['img']
	data_dict = json.load(request.files['data'])

	model.update_branches_temperature(config.p_tar_calib)

	#result = edgeProcessing.edgeBranchesCalibInference(fileImg)
	result = edgeProcessing.edgeBranchesCalibInference(fileImg, data_dict)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200
	else:
		return jsonify(result), 500


@api.route('edge/edgeBranchesCalibInferenceOnlyEdge', methods=["POST"])
def edge_inference_branches_calib_only_edge():

	fileImg = request.files['img']
	data_dict = json.load(request.files['data'])

	model.update_branches_temperature(config.p_tar_calib)

	result = edgeProcessing.edgeBranchesCalibInferenceOnlyEdge(fileImg, data_dict)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200
	else:
		return jsonify(result), 500


@api.route('edge/edgeBranchesCalibInferenceOnlyEdgeStandardDNN', methods=["POST"])
def edge_inference_branches_calib_only_edge_standard_dnn():

	fileImg = request.files['img']
	data_dict = json.load(request.files['data'])

	model.update_branches_temperature(config.p_tar_calib)

	result = edgeProcessing.edgeBranchesCalibInferenceOnlyEdgeStandardDNN(fileImg, data_dict)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200
	else:
		return jsonify(result), 500
