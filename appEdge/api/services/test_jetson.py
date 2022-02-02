import torch, os, sys, config, time
import torchvision
from .early_exit_dnn import Early_Exit_DNN_CALTECH




model_name = "mobilenet"
n_classes = 258
n_branches = 5
input_shape = (3, 224, 224)
seed = 42
p_tar = 0.7
nr_branch_edge = 5

dataset_path = config.models_params["caltech256"]["dataset_path"]
	
save_indices_path = config.models_params["caltech256"]["indices"]


ee_model = Early_Exit_DNN_CALTECH(model_name, n_classes, config.pretrained, n_branches, input_shape, config.exit_type, 
	config.device, config.distribution)

model_path = os.path.join(config.edge_model_root_path, "caltech256", "mobilenet", "models", 
	"ee_mobilenet_branches_5_id_1.pth")

ee_model.load_state_dict(torch.load(model_path, map_location=config.device)["model_state_dict"])
ee_model.eval()

mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
std =  [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

# Note that we do not apply data augmentation in the test dataset.
transformations_test = transforms.Compose([
	transforms.Resize(input_dim),
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

