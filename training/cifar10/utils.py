import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.mobilenet import MobileNetV2_2
from networks.resnet import resnet18, resnet152
from networks.vgg import vgg16_bn
import torchvision.models as models
import pandas as pd
import numpy as np

def create_dirs(history_path, model_path):
	if not os.path.exists(model_path):
		os.makedirs(model_path)
		os.makedirs(history_path)

def create_dir(dir_path):
	if (not os.path.exists(dir_path)):
		os.makedirs(dir_path)

def verify_stop_condition(count, epoch, args):
	stop_condition = count <= args.patience if(args.pretrained) else epoch <= args.n_epochs
	return stop_condition


def save_calibration_main_results(result, save_path):
	df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
	df.to_csv(save_path)


def get_model_arch(pretrained, model_name, n_classes, device):
	if (pretrained):
		print("OK")
		mobilenet = models.mobilenet_v2()
		mobilenet.classifier[1] = nn.Linear(1280, n_classes)

		vgg16 = models.vgg16_bn()
		vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, n_classes)

		model_resnet18 = models.resnet18()
		model_resnet18.fc = nn.Linear(model_resnet18.fc.in_features, n_classes)

		model_resnet152 = models.resnet152()
		model_resnet152.fc = nn.Linear(model_resnet152.fc.in_features, n_classes)

		dict_model = {"mobilenet": mobilenet, "vgg16": vgg16, 
		"resnet18": model_resnet18, "resnet152": model_resnet152}
	else:
		dict_model = {"mobilenet": MobileNetV2_2(n_classes, device), "vgg16": vgg16_bn(n_classes), 
		"resnet18": resnet18(n_classes), "resnet152": resnet152(n_classes)}

	return dict_model[model_name]

def save_all_results_ee_calibration(no_calib_result, overall_result, branches_result, all_samples_result, resultPathDict):
	save_result(no_calib_result, resultPath["no_calib"])
	save_result(overall_result, resultPath["overall_calib"])
	save_result(branches_result, resultPath["branches_calib"])
	save_result(all_samples_result, resultPath["all_samples_calib"])


def save_result(result, save_path):
	df_result = pd.read_csv(save_path) if (os.path.exists(save_path)) else pd.DataFrame()
	df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
	df_result = df_result.append(df)
	df_result.to_csv(save_path)


def testEarlyExitInference(model, n_branches, test_loader, threshold, device, model_type):
	df_result = pd.DataFrame()

	n_exits = n_branches + 1
	conf_branches_list, infered_class_branches_list, target_list = [], [], []
	correct_list, exit_branch_list, id_list = [], [], []

	model.eval()

	with torch.no_grad():
		for i, (data, target) in tqdm(enumerate(test_loader, 1)):

			data, target = data.to(device), target.to(device)

			if (model_type == "no_calib"):
				_, conf_branches, infered_class_branches = model.forwardAllExits(data)

			elif (model_type == "calib_overall"):
				conf_branches, infered_class_branches = model.forwardOverall(data)

			elif (model_type == "calib_branches"):
				conf_branches, infered_class_branches = model.forwardBranchesCalibration(data)
			
			else:
				conf_branches, infered_class_branches = model.forwardBranchesCalibration(data)

			conf_branches_list.append([conf.item() for conf in conf_branches])
			infered_class_branches_list.append([inf_class.item() for inf_class in infered_class_branches])    
			correct_list.append([infered_class_branches[i].eq(target.view_as(infered_class_branches[i])).sum().item() for i in range(n_exits)])
			id_list.append(i)
			target_list.append(target.item())

			del data, target
			torch.cuda.empty_cache()

	conf_branches_list = np.array(conf_branches_list)
	infered_class_branches_list = np.array(infered_class_branches_list)
	correct_list = np.array(correct_list)
  
	print(model_type)
	print("Acc:")
	print([sum(correct_list[:, i])/len(correct_list[:, i]) for i in range(n_exits)])

	results = {"p_tar": [p_tar]*len(target_list), "target": target_list, "id": id_list}
	for i in range(n_exits):
		results.update({"conf_branch_%s"%(i+1): conf_branches_list[:, i],
			"infered_class_branches_%s"%(i+1): infered_class_branches_list[:, i],
			"correct_branch_%s"%(i+1): correct_list[:, i]})

	return results

