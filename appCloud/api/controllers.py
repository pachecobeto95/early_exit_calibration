from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
import json, os, time, sys, config, requests
from .services import cloudProcessing
from .services.cloudProcessing import model, exp


api = Blueprint("api", __name__, url_prefix="/api")

@api.route("/cloud/onlyCloudProcessing", methods=["POST"])
def onlyCloudProcessing():
	fileImg = request.files['img']

	#This functions process the DNN inference
	result = cloudProcessing.onlyCloudProcessing(fileImg)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500


@api.route("/cloud/expConfiguration", methods=["POST"])
def edgeExpConfiguration():
	exp.exp_params = request.json
	return jsonify({"status": "ok"}), 200


@api.route("/cloud/modelConfiguration", methods=["POST"])
def cloudModelConfiguration():
	data = request.json
	model.model_params = data
	model.load_model()
	model.load_temperature()
	return jsonify({"status": "ok"}), 200


"""
# Define url for the user send the image
@api.route('/cloud/cloudInference', methods=["POST"])
def cloud_inference():

	#This function receives an image or feature map from edge device (Access Point)
	#This functions is run in the cloud.

	data_from_edge = request.json
	result = cloudProcessing.dnnInferenceCloud(data_from_edge["feature"], data_from_edge["conf"], data_from_edge["class_list"], 
		data_from_edge["p_tar"], data_from_edge["nr_branch_edge"])

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500

"""

@api.route('/cloud/cloudNoCalibInference', methods=["POST"])
def cloud_no_calib_inference():
	"""
	This function receives an image or feature map from edge device (Access Point)
	This functions is run in the cloud.
	"""

	data = request.json
	result = cloudProcessing.cloudNoCalibInference(data["feature"], data["conf"], data["class_list"])

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500


@api.route('/cloud/cloudOverallCalibInference', methods=["POST"])
def cloud_overall_calib_inference():
	"""
	This function receives an image or feature map from edge device (Access Point)
	This functions is run in the cloud.
	"""

	data = request.json
	#model.update_overall_temperature(data_from_edge["p_tar"])

	result = cloudProcessing.cloudOverallCalibInference(data["feature"], data["conf"], data["class_list"])

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500


@api.route('/cloud/cloudBranchesCalibInference', methods=["POST"])
def cloud_branches_calib_inference():
	"""
	This function receives an image or feature map from edge device (Access Point)
	This functions is run in the cloud.
	"""

	data = request.json
	#model.update_branches_temperature(data_from_edge["p_tar"])

	result = cloudProcessing.cloudBranchesCalibInference(data["feature"], data["conf"], data["class_list"])

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500

@api.route('/cloud/cloudAllSamplesCalibInference', methods=["POST"])
def cloud_all_samples_calib_inference():
	"""
	This function receives an image or feature map from edge device (Access Point)
	This functions is run in the cloud.
	"""

	data_from_edge = request.json
	model.update_all_samples_temperature(data_from_edge["p_tar"])

	result = cloudProcessing.cloudAllSamplesCalibInference(data_from_edge["feature"], data_from_edge["conf"], data_from_edge["class_list"], 
		data_from_edge["p_tar"], data_from_edge["nr_branch_edge"])

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500






