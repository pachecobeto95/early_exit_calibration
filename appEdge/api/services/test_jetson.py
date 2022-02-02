import torch, os, sys, time
import torchvision
from early_exit_dnn import Early_Exit_DNN_CALTECH
import torchvision.transforms as transforms
import numpy as np
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler



DIR_NAME = os.path.dirname(__file__)
dataset_root_path = os.path.join(DIR_NAME, "datasets")

model_name = "mobilenet"
n_classes = 258
n_branches = 5
input_shape = (3, 224, 224)
seed = 42
p_tar = 0.7
nr_branch_edge = 5
pretrained = False
exit_type = "bnpool"
distribution = "linear"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device: %s"%(device))

models_params = {"caltech256": {"n_classes": 258, "input_shape":(3, 224, 224),
"dataset_path": os.path.join(dataset_root_path, "caltech256", "256_ObjectCategories"), 
"indices": os.path.join(DIR_NAME, "datasets", "caltech256", "indices", "test_idx_caltech256_id_1.npy")}, 
"cifar100": {"n_classes": 100, "input_shape":(3, 32, 32),
"indices": os.path.join(DIR_NAME, "datasets", "cifar100", "indices"),
"dataset_path": os.path.join(dataset_root_path, "cifar100")}
}


dataset_path = os.path.join(".", "256_ObjectCategories")
	
save_indices_path = os.path.join(DIR_NAME, "indices", "test_idx_caltech256_id_1.npy")

#edge_model_root_path = os.path.join(DIR_NAME, "appEdge", "api", "services", "models")
edge_model_root_path = os.path.join(DIR_NAME, "models")


ee_model = Early_Exit_DNN_CALTECH(model_name, n_classes, pretrained, n_branches, input_shape, exit_type, 
	device, distribution)

model_path = os.path.join(edge_model_root_path, "caltech256", "mobilenet", "models", 
	"ee_mobilenet_branches_5_id_1.pth")

ee_model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
ee_model.eval()

mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
std =  [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

# Note that we do not apply data augmentation in the test dataset.
transformations_test = transforms.Compose([
	transforms.Resize(224),
	transforms.ToTensor(), 
	transforms.Normalize(mean = mean, std = std),
	])


torch.manual_seed(seed)
np.random.seed(seed=seed)

train_set = datasets.ImageFolder(dataset_path)

val_set = datasets.ImageFolder(dataset_path, transform=transformations_test)
    
test_set = datasets.ImageFolder(dataset_path, transform=transformations_test)

test_idx_path = os.path.join(savePath_idx, "test_idx_caltech256_id_%s.npy"%(model_id))
test_idx = np.load(test_idx_path, allow_pickle=True)
test_idx = np.array(list(test_idx.tolist()))
test_data = torch.utils.data.Subset(test_set, indices=test_idx)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=4)

for i, (data, target) in enumerate(test_loader, 1):
	start = time.time()
	output, conf_list, class_list, isTerminate = ee_model.forwardEdgeNoCalibInference(data, p_tar, nr_branch_edge)
	end = time.time()

	print("Duration: %s"%(end-start))

