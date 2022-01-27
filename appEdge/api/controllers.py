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

	#json_data = json.load(request.files['data'])

	#This functions process the DNN inference
	#result = edgeProcessing.edgeInference(fileImg, json_data["p_tar"], json_data["nr_branch_edge"])
	return jsonify({"status": "ok"}), 200

	result = edgeProcessing.edgeInference(fileImg)


	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500

@api.route('edge/edgeNoCalibInference', methods=["POST"])
def edge_inference_no_calib():

	fileImg = request.files['img']

	result = edgeProcessing.edgeNoCalibInference(fileImg)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200
	else:
		return jsonify(result), 500


@api.route('edge/edgeOverallCalibInference', methods=["POST"])
def edge_inference_overall_calib():

	fileImg = request.files['img']
	return jsonify({"status": "ok"}), 200

	#json_data = json.load(request.files['data'])

	model.update_overall_temperature(json_data["p_tar"])

	result = edgeProcessing.edgeOverallCalibInference(fileImg, json_data["p_tar"], json_data["nr_branch_edge"])

	if (result["status"] ==  "ok"):
		return jsonify(result), 200
	else:
		return jsonify(result), 500


@api.route('edge/edgeBranchesCalibInference', methods=["POST"])
def edge_inference_branches_calib():

	fileImg = request.files['img']
	#json_data = json.load(request.files['data'])
	return jsonify({"status": "ok"}), 200


	model.update_branches_temperature(json_data["p_tar"])

	result = edgeProcessing.edgeBranchesCalibInference(fileImg, json_data["p_tar"], json_data["nr_branch_edge"])

	if (result["status"] ==  "ok"):
		return jsonify(result), 200
	else:
		return jsonify(result), 500

@api.route('edge/edgeAllSamplesCalibInference', methods=["POST"])
def edge_inference_all_samples_calib():

	fileImg = request.files['img']
	#json_data = json.load(request.files['data'])
	return jsonify({"status": "ok"}), 200

	model.update_all_samples_temperature(json_data["p_tar"])

	result = edgeProcessing.edgeAllSamplesCalibInference(fileImg, json_data["p_tar"], json_data["nr_branch_edge"])

	if (result["status"] ==  "ok"):
		return jsonify(result), 200
	else:
		return jsonify(result), 500

@api.route('/edge/testJson', methods=["POST"])
def edge_test_json():
	"""
	This function tests the server to receive a simple json post request.
	"""	

	post_data = request.json

	result = {"status": "ok"}

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500
