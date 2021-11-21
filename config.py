import os, torch

DIR_NAME = os.path.dirname(__file__)

DEBUG = True

# Edge URL Configuration 
HOST_EDGE = "146.164.69.144"
#HOST_EDGE = "192.168.0.20"
PORT_EDGE = 5001
URL_EDGE = "http://%s:%s"%(HOST_EDGE, PORT_EDGE)
URL_EDGE_DNN_INFERENCE = "%s/api/edge/edgeInference"%(URL_EDGE)
urlConfModelEdge = "%s/api/edge/modelConfiguration"%(URL_EDGE)

url_edge_no_calib = "%s/api/edge/edgeNoCalibInference"%(URL_EDGE)
url_edge_overall_calib = "%s/api/edge/edgeOverallCalibInference"%(URL_EDGE)
url_edge_branches_calib = "%s/api/edge/edgeBranchesCalibInference"%(URL_EDGE)
url_edge_all_samples_calib = "%s/api/edge/edgeAllSamplesCalibInference"%(URL_EDGE)

edge_model_root_path = os.path.join(DIR_NAME, "appEdge", "api", "services", "models")

# Cloud URL Configuration 
HOST_CLOUD = "146.164.69.144"
#HOST_CLOUD = "192.168.0.20"
PORT_CLOUD = 3001
URL_CLOUD = "http://%s:%s"%(HOST_CLOUD, PORT_CLOUD)
URL_CLOUD_DNN_INFERENCE = "%s/api/cloud/cloudInference"%(URL_CLOUD)
urlConfModelCloud = "%s/api/cloud/modelConfiguration"%(URL_CLOUD)
urlOnlyCloudProcessing = "%s/api/cloud/onlyCloudProcessing"%(URL_CLOUD)
url_cloud_no_calib = "%s/api/cloud/cloudNoCalibInference"%(URL_CLOUD)
url_cloud_overall_calib = "%s/api/cloud/cloudOverallCalibInference"%(URL_CLOUD)
url_cloud_branches_calib = "%s/api/cloud/cloudBranchesCalibInference"%(URL_CLOUD)
url_cloud_all_samples_calib = "%s/api/cloud/cloudAllSamplesCalibInference"%(URL_CLOUD)

cloud_model_root_path = os.path.join(DIR_NAME, "appCloud", "api", "services", "models")


#Dataset Path
dataset_path = os.path.join(DIR_NAME, "datasets", "test_dataset_caltech256")

nr_max_branches = 5
nr_min_branches = 2
model_name = "mobilenet"
dataset_name = "caltech256"
nr_branch_model = nr_branch_model = 5
input_dim = 224
input_shape = (3, input_dim, input_dim)
pretrained = False
n_branches = 5
exit_type = "bnpool"
distribution = "linear"

disabled_branches = [1, 2]

timeout = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


models_params = {"caltech256": {"n_classes": 258}, "cifar100": {"n_classes": 100}}
model_id = 1

