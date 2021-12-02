import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
import os, sys, time, math, os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import product
import pandas as pd
import torchvision.models as models
from torchvision import datasets, transforms
import functools
from tqdm import tqdm
from load_dataset import LoadDataset
from early_exit_dnn import Early_Exit_DNN
from calibration_early_exit_dnn import ModelOverallCalibration, ModelBranchesCalibration, ModelAllSamplesCalibration


torch.multiprocessing.set_sharing_strategy('file_system')


def calibratingModels(model, val_loader, p_tar, device, model_path, temperaturePath):

	overall_model = ModelOverallCalibration(model, device, model_path, saveTempPath)
	overall_model.set_temperature(val_loader, p_tar)

	branches_model = ModelBranchesCalibration(model, device, model_path, saveTempPath)
	branches_model.set_temperature(val_loader, p_tar)

	all_samples_model = ModelAllSamplesCalibration(model, device, model_path, saveTempPath)
	all_samples_model.set_temperature(val_loader, p_tar)

	return {"calib_overall": overall_model, "calib_branches": branches_model, "calib_branches_all_samples": all_samples_model}


def save_result(result, save_path):
	df_result = pd.read_csv(save_path) if (os.path.exists(save_path)) else pd.DataFrame()
	df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
	df_result = df_result.append(df)
	df_result.to_csv(save_path)


def save_all_results(no_calib_result, overall_result, branches_result, all_samples_result, resultPath)

	save_result(no_calib_result, resultPath["no_calib"])
	save_result(overall_result, resultPath["calib_overall"])
	save_result(branches_result, resultPath["calib_branches"])
	save_result(all_samples_result, resultPath["calib_branches_all_samples"])

def evalEarlyExitInference(model, n_branches, test_loader, p_tar, device, model_type):

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

def expCalibration(model, val_loader, test_loader, device, p_tar_list, model_path, , resultPath, temperaturePath):
	for p_tar in p_tar_list:
	
		print("P_tar: %s"%(round(p_tar, 2)))

		no_calib_result = evalEarlyExitInference(model, model.n_branches, test_loader, 
			p_tar, device, model_type="no_calib")

		scaled_models_dict = calibratingModels(model, val_loader, p_tar, device, model_path, temperaturePath)


		overall_result = evalEarlyExitInference(scaled_models_dict["calib_overall"], model.n_branches, test_loader, 
			p_tar, device, model_type="calib_overall")

		branches_result = evalEarlyExitInference(scaled_models_dict["calib_branches"], model.n_branches, test_loader, 
			p_tar, device, model_type="calib_branches")

		all_samples_result = evalEarlyExitInference(scaled_models_dict["calib_branches_all_samples"], model.n_branches, 
			test_loader, p_tar, device, model_type="calib_branches_all_samples")

		save_all_results(no_calib_result, overall_result, branches_result, 
			all_samples_result, resultPath)


input_dim = 224
batch_size_train = 64
batch_size_test = 1
model_id = 1
split_ratio = 0.2
n_classes = 258
pretrained = False
n_branches = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_shape = (3, input_dim, input_dim)

distribution = "linear"
exit_type = "bnpool"
dataset_name = "caltech256"
model_name = "resnet152"


root_save_path = "."

save_indices_path = os.path.join(".", "caltech256", "indices")


dataset_path = os.path.join(root_save_path, "datasets", dataset_name, "256_ObjectCategories")

model_path = os.path.join(root_save_path, "appEdge", "api", "services", "models",
	dataset_name, model_name, "models", 
	"ee_%s_branches_%s_id_%s.pth"%(model_name, n_branches, model_id))

save_path = os.path.join(root_save_path, "appEdge", "api", "services", "models", dataset_name, model_name)

result_path = os.path.join(save_path, "results")

no_calib_result_path =  os.path.join(result_path, "no_calib_exp_data_%s_alert.csv"%(model_id))
calib_overall_result_path =  os.path.join(result_path, "calib_overall_exp_data_%s_alert.csv"%(model_id))
calib_branches_result_path =  os.path.join(result_path, "calib_branches_exp_data_%s_alert.csv"%(model_id))
calib_all_samples_result_path =  os.path.join(result_path, "calib_all_samples_branches_exp_data_%s_alert.csv"%(model_id))


saveTempOverallPath = os.path.join(save_path, "temperature", "temp_overall_id_%s.csv"%(model_id))
saveTempBranchesPath = os.path.join(save_path, "temperature", "temp_branches_id_%s.csv"%(model_id))
saveTempBranchesAllSamplesPath = os.path.join(save_path, "temperature", "temp_all_samples_id_%s.csv"%(model_id))

resultPath = {"no_calib": no_calib_result_path, 
"calib_overall": calib_overall_result_path, "calib_branches":calib_branches_result_path,
"calib_branches_all_samples": calib_all_samples_result_path}

temperaturePath = {"calib_overall": saveTempOverallPath, 
"calib_branches": saveTempBranchesPath, "calib_branches_all_samples":saveTempBranchesAllSamplesPath}

p_tar_list = [0.8, 0.85, 0.9, 0.95]

expCalibration(early_exit_dnn, val_loader, test_loader, device, p_tar_list, model_path, resultPath, temperaturePath)
