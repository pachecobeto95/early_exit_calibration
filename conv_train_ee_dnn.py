import config, utils, ee_dnn
import argparse, logging, os, torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd

def compute_metrics(criterion, output_list, conf_list, class_list, target, loss_weights):
	model_loss = 0
	ee_loss, acc_branches = [], []

	for i, (output, inf_class, weight) in enumerate(zip(output_list, class_list, loss_weights), 1):
		loss_branch = criterion(output, target)
		model_loss += weight*loss_branch

		acc_branch = 100*inf_class.eq(target.view_as(inf_class)).sum().item()/target.size(0)

		ee_loss.append(loss_branch.item()), acc_branches.append(acc_branch)

	acc_model = np.mean(np.array(acc_branches))

	return model_loss, ee_loss, acc_model, acc_branches


def trainEEDNNs(model, train_loader, optimizer, criterion, n_exits, epoch, device, loss_weights):

	model_loss_list, ee_loss_list = [], []
	model_acc_list, ee_acc_list = [], []

	model.train()

	for (data, target) in tqdm(train_loader):
		data, target = data.to(device), target.to(device)

		output_list, conf_list, class_list, _, _ = model.forward_training(data)
		optimizer.zero_grad()


		model_loss, ee_loss, model_acc, ee_acc = compute_metrics(criterion, output_list, conf_list, class_list, target, loss_weights)

		model_loss_list.append(float(model_loss.item())), ee_loss_list.append(ee_loss)

		model_acc_list.append(model_acc), ee_acc_list.append(ee_acc)

		model_loss.backward()
		optimizer.step()

		# clear variables
		del data, target, output_list, conf_list, class_list
		torch.cuda.empty_cache()


	avg_loss, avg_ee_loss = round(np.mean(model_loss_list), 4), np.mean(ee_loss_list, axis=0)

	avg_acc, avg_ee_acc = round(np.mean(model_acc_list), 2), np.mean(ee_acc_list, axis=0)

	#logging.debug("Epoch: %s, Train Model Loss: %s, Train Model Acc: %s"%(epoch, avg_loss, avg_acc))
	print("Epoch: %s, Train Model Loss: %s, Train Model Acc: %s"%(epoch, avg_loss, avg_acc))

	result_dict = {"epoch": epoch, "train_loss": avg_loss, "train_acc": avg_acc}

	for i in range(n_exits):
		result_dict.update({"train_ee_acc_%s"%(i+1): avg_ee_acc[i], "train_ee_loss_%s"%(i+1): avg_ee_loss[i]})
		logging.debug("Epoch: %s, Train Loss EE %s: %s, Train Acc EE %s: %s"%(epoch, i, avg_ee_loss[i], i, avg_ee_acc[i]))
		print("Epoch: %s, Train Loss EE %s: %s, Train Acc EE %s: %s"%(epoch, i, avg_ee_loss[i], i, avg_ee_acc[i]))

	return result_dict


def evalEEDNNs(model, val_loader, criterion, n_exits, epoch, device, loss_weights):

	model_loss_list, ee_loss_list = [], []
	model_acc_list, ee_acc_list = [], []

	model.eval()

	with torch.no_grad():
		for (data, target) in tqdm(val_loader):
			data, target = data.to(device), target.to(device)

			output_list, conf_list, class_list = model.forwardTraining(data)

			model_loss, ee_loss, model_acc, ee_acc, _, _ = compute_metrics(criterion, output_list, conf_list, class_list, target, loss_weights)

			model_loss_list.append(float(model_loss.item())), ee_loss_list.append(ee_loss)
			model_acc_list.append(model_acc), ee_acc_list.append(ee_acc)

			# clear variables
			del data, target, output_list, conf_list, class_list
			torch.cuda.empty_cache()

	avg_loss, avg_ee_loss = round(np.mean(model_loss_list), 4), np.mean(ee_loss_list, axis=0)

	avg_acc, avg_ee_acc = round(np.mean(model_acc_list), 2), np.mean(ee_acc_list, axis=0)

	logging.debug("Epoch: %s, Val Model Loss: %s, Val Model Acc: %s"%(epoch, avg_loss, avg_acc))
	#print("Epoch: %s, Val Model Loss: %s, Val Model Acc: %s"%(epoch, avg_loss, avg_acc))

	result_dict = {"epoch": epoch, "val_loss": avg_loss, "val_acc": avg_acc}

	for i in range(n_exits):
		result_dict.update({"val_ee_acc_%s"%(i+1): avg_ee_acc[i], "val_ee_loss_%s"%(i+1): avg_ee_loss[i]})
		#logging.debug("Epoch: %s, Val Loss EE %s: %s, Val Acc EE %s: %s"%(epoch, i, avg_ee_loss[i], i, avg_ee_acc[i]))
		print("Epoch: %s, Val Loss EE %s: %s, Val Acc EE %s: %s"%(epoch, i, avg_ee_loss[i], i, avg_ee_acc[i]))

	return result_dict



def main(args):

	models_dir_path = os.path.join(config.DIR_PATH, args.model_name, "models")
	history_dir_path = os.path.join(config.DIR_PATH, args.model_name, "history")

	os.makedirs(models_dir_path, exist_ok=True), os.makedirs(history_dir_path, exist_ok=True)

	model_save_path = os.path.join(models_dir_path, "ee_model_%s_%s_branches_id_%s.pth"%(args.model_name, args.n_branches, args.model_id))

	history_path = os.path.join(history_dir_path, "history_ee_model_%s_%s_branches_id_%s.csv"%(args.model_name, args.n_branches, args.model_id))

	indices_path = os.path.join(config.DIR_PATH, "indices_%s.pt"%(args.dataset_name))

	logPath = os.path.join(config.DIR_PATH, "log_train_ee_model_%s_%s_branches_id_%s.csv"%(args.model_name, args.n_branches, args.model_id))

	logging.basicConfig(level=logging.DEBUG, filename=logPath, filemode="a+", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

	device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

	train_loader, val_loader, test_loader = utils.load_caltech256(args, config.dataset_path, indices_path)

	n_classes = 257

	loss_weights_dict = {"crescent": np.linspace(0.3, 1, args.n_branches+1), 
	"decrescent": np.linspace(1, 0.3, args.n_branches+1), "equal": np.ones(args.n_branches+1)}
	
	loss_weights = loss_weights_dict[args.loss_weights_type]

	current_result = {"exit_type": args.exit_type, "distribution": args.distribution, "n_classes": n_classes,
	"input_dim": args.dim, "loss_weights_type": args.loss_weights_type}

	#Instantiate the Early-exit DNN model.
	ee_model = ee_dnn.Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.n_branches, 
		args.dim, args.exit_type, device, args.distribution)

	#Load the trained early-exit DNN model.
	ee_model = ee_model.to(device)

	lr = [1.5e-4, 0.01]

	criterion = nn.CrossEntropyLoss()

	optimizer = optim.Adam([{'params': ee_model.stages.parameters(), 'lr': lr[0]}, 
		{'params': ee_model.exits.parameters(), 'lr': lr[1]},
		{'params': ee_model.classifier.parameters(), 'lr': lr[0]}], weight_decay=args.weight_decay)

	#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1, verbose=True)
	n_exits = args.n_branches + 1

	epoch, count_patience = 0, 0
	best_val_loss = np.inf
	df_history = pd.DataFrame()

	while (count_patience < args.max_patience):
		count_patience += 1

		train_result = trainEEDNNs(ee_model, train_loader, optimizer, criterion, n_exits, epoch, device, loss_weights)
		val_result = evalEEDNNs(ee_model, val_loader, criterion, n_exits, epoch, device, loss_weights)
		#scheduler.step()

		current_result.update(train_result), current_result.update(val_result)
		df_history = df_history.append(pd.Series(current_result), ignore_index=True)
		df_history.to_csv(history_path)

		if (val_result["val_loss"] < best_val_loss):
			save_dict  = {}	
			best_val_loss = val_result["val_loss"]
			count_patience = 0

			save_dict.update(current_result)
			save_dict.update({"model_state_dict": ee_model.state_dict(), "opt_state_dict": optimizer.state_dict()})
			torch.save(save_dict, model_save_path)

		else:
			count_patience += 1
			print("Current Patience: %s"%(count_patience))

	print("Stop! Patience is finished")




if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Training Early-exit DNN. These are the hyperparameters")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech256", "cifar10"], help='Dataset name.')

	#We here insert the argument model_name. 
	#We evalue our novel calibration method Offloading-driven Temperature Scaling in four early-exit DNN:
	#MobileNet, ResNet18, ResNet152, VGG16
	parser.add_argument('--model_name', type=str, default=config.model_name, choices=["mobilenet", "alexnet"], 
		help='DNN model name (default: %s)'%(config.model_name))

	#This argument defines the ratio to split the Traning Set, Val Set, and Test Set.
	parser.add_argument('--split_ratio', type=float, default=config.split_ratio, help='Split Ratio')

	#This argument defined the batch sizes. 
	parser.add_argument('--batch_size_train', type=int, default=config.batch_size_train, 
		help='Train Batch Size. Default: %s'%(config.batch_size_train))

	parser.add_argument('--input_dim', type=int, default=330, help='Input Dim.')

	parser.add_argument('--dim', type=int, default=300, help='Image dimension')

	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')

	#parser.add_argument('--cuda', type=bool, default=config.use_gpu, help='Use GPU? Default: %s'%(config.use_gpu))

	parser.add_argument('--n_branches', type=int, help='Number of side branches.')

	parser.add_argument('--exit_type', type=str, default=config.exit_type, 
		help='Exit Type. Default: %s'%(config.exit_type))

	parser.add_argument('--distribution', type=str, default=config.distribution, 
		help='Distribution of the early exits. Default: %s'%(config.distribution))

	parser.add_argument('--pretrained', type=bool, default=config.pretrained, help='Backbone DNN is pretrained.')

	#parser.add_argument('--epochs', type=int, default=config.epochs, help='Epochs.')

	parser.add_argument('--max_patience', type=int, default=config.max_patience, help='Max Patience.')

	parser.add_argument('--model_id', type=int, help='Model_id.')

	parser.add_argument('--loss_weights_type', type=str, help='loss_weights_type.')

	parser.add_argument('--weight_decay', type=int, default=config.weight_decay, help='Weight Decay.')

	args = parser.parse_args()

	main(args)