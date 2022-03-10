import os, logging, time, sys, config
import numpy as np
from load_dataset import load_test_caltech_256
from PIL import Image
import pandas as pd
import argparse
from appEdge.api.services.early_exit_dnn import Early_Exit_DNN_CALTECH, Early_Exit_DNN_CIFAR
import torch

def load_dataset(args, dataset_path, savePath_idx):
	if(args.dataset_name=="caltech256"):
		return load_test_caltech_256(config.input_dim, dataset_path, args.split_ratio, savePath_idx, 
			config.model_id_dict[args.model_name])

	elif(args.dataset_name=="cifar100"):
		sys.exit()


def experiment_early_exit_inference(model, test_loader, p_tar, nr_branch_edge, device):

	df_result = pd.DataFrame()
	n_exits = nr_branch_edge + 1
	conf_branches_list, infered_class_branches_list, target_list = [], [], []
	correct_list, id_list = [], []

	model.eval()

	test_dataset_size = len(test_loader)

	with torch.no_grad():
		for i, (data, target) in enumerate(test_loader, 1):
			
			print("Image id: %s/%s"%(i, test_dataset_size))
			data, target = data.to(device).float(), target.to(device)

			conf_branches, infered_class_branches = model.forwardEarlyExitInference(data, p_tar)

			conf_branches_list.append([conf.item() for conf in conf_branches])
			infered_class_branches_list.append([inf_class.item() for inf_class in infered_class_branches])    
			correct_list.append([infered_class_branches[i].eq(target.view_as(infered_class_branches[i])).sum().item() for i in range(n_exits)])
			id_list.append(i), target_list.append(target.item())

			del data, target
			torch.cuda.empty_cache()

	conf_branches_list = np.array(conf_branches_list)
	infered_class_branches_list = np.array(infered_class_branches_list)
	correct_list = np.array(correct_list)

	results = {"p_tar": [p_tar]*len(target_list), "target": target_list, "id": id_list}

	for i in range(n_exits):
		results.update({"conf_branch_%s"%(i+1): conf_branches_list[:, i],
			"infered_class_branches_%s"%(i+1): infered_class_branches_list[:, i],
			"correct_branch_%s"%(i+1): correct_list[:, i]})

	return results


def save_results(result, save_path):
	df_result = pd.read_csv(save_path) if (os.path.exists(save_path)) else pd.DataFrame()
	df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
	df_result = df_result.append(df)
	df_result.to_csv(save_path)


def collectingConfidenceExperiment(test_loader, model, p_tar_list, nr_branch_edge, device, savePath):


	for p_tar in p_tar_list:
		logging.debug("P_tar: %s"%(p_tar))

		confidence_results = experiment_early_exit_inference(model, test_loader, p_tar, nr_branch_edge, device)

		save_results(confidence_results, savePath)


def loadEarlyExitDNN(model_name, dataset_name, n_classes, pretrained, nr_branches, input_shape, exit_type, device, distribution, root_path):

	model_id = config.model_id_dict[model_name]

	if(dataset_name == "caltech256"):
		ee_model = Early_Exit_DNN_CALTECH(model_name, n_classes, pretrained, nr_branches, input_shape, exit_type, device, distribution)
		model_file_name = "ee_%s_branches_%s_id_%s.pth"%(model_name, nr_branches, model_id)

	elif((dataset_name == "cifar100") or (dataset_name == "cifar10")):

		ee_model = Early_Exit_DNN_CIFAR(model_name, n_classes, pretrained, nr_branches, input_shape, exit_type, device, distribution)
		model_file_name = "b_%s_early_exit_%s_id_1_%s_decrescent.pth"%(model_name, dataset_name, pretrained)
	
	else:
		logging.debug("Error")
		sys.exit()

	model_path = os.path.join(root_path, dataset_name, model_name, "models", model_file_name)
	ee_model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])

	return ee_model

def main(args):
	#Number of side branches that exists in the early-exit DNNs

	DIR_NAME = os.path.dirname(__file__)
	p_tar_list = [0.7, 0.8, 0.9]
	dataset_path = os.path.join(DIR_NAME, "datasets", "caltech256", "256_ObjectCategories")
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	savePath = os.path.join(DIR_NAME, "osvaldo_experiments", "confidence_branches_%s_%s_pretrained.csv"%(args.model_name, args.dataset_name))
	edge_root_path = os.path.join(DIR_NAME, "appEdge", "api", "services", "models")

	#This line defines the number of side branches processed at the edge
	nr_branch_edge = 5
	n_classes = config.models_params[args.dataset_name]["n_classes"]

	early_exit_dnn = loadEarlyExitDNN(args.model_name, args.dataset_name, n_classes, True, nr_branch_edge, config.input_shape, 
		config.exit_type, device, config.distribution, edge_root_path)

	logPath = "./logConfidenceCollecting_%s_%s.log"%(args.model_name, args.dataset_name)

	logging.basicConfig(level=logging.DEBUG, filename=logPath, filemode="a+", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

	save_indices_path = os.path.join(DIR_NAME, "datasets", args.dataset_name, "indices", "test_idx_caltech256_id_1.npy")

	test_loader = load_dataset(args, dataset_path, save_indices_path)
	collectingConfidenceExperiment(test_loader, early_exit_dnn, p_tar_list, nr_branch_edge, device, savePath)


if (__name__ == "__main__"):
	# Input Arguments. Hyperparameters configuration
	parser = argparse.ArgumentParser(description="Evaluating early-exits DNNs perfomance")

	parser.add_argument('--dataset_name', type=str, default="caltech256", 
		choices=["caltech256", "cifar10", "cifar100"], help='Dataset name (default: Caltech-256)')

	parser.add_argument('--model_name', type=str, default="mobilenet", 
		choices=["mobilenet", "resnet18", "resnet152", "vgg16"], help='DNN model name (default: MobileNet)')

	parser.add_argument('--split_ratio', type=float, default=0.2, help='Split Ratio')

	args = parser.parse_args()

	main(args)
