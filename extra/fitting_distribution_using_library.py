import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, argparse
from fitter import Fitter, get_common_distributions, get_distributions
from scipy import stats

def kstest(data, dist_name, paramtup, ksN=100):
	p_value = stats.kstest(data, dist_name, paramtup, ksN)[1]   # return p-value
	return p_value             # return p-value


def getDeltaConf(df, gamma):

	df_branch = df[df.conf_branch_1 <= gamma]
	#df_branch = df
	conf_branch = df_branch.conf_branch_1.values
	#conf_final = np.maximum(df_branch.conf_branch_2.values, conf_branch)
	conf_final = df_branch.conf_branch_2.values
	delta_conf = conf_final - conf_branch
	return delta_conf


def compute_p_value(f, data):
	distribution_name_list = f.summary().iloc[:, 0].index.values
	p_value_list = []
	for dist_name in distribution_name_list:
		params = f.fitted_param[dist_name]
		p_value = kstest(data, dist_name, params)
		p_value_list.append(p_value)
  
	return p_value_list


def expFittingDistributionsUsingLibrary(df, gamma_list, dist_list, paramsDict):

	for gamma in gamma_list:

		file_name = "pdf_gamma_%s_using_library3"%(gamma)
		filePath = os.path.join(paramsDict["plotPath"], file_name)
		saveErrosPathLibrary = os.path.join(paramsDict["plotPath"], "results_error_library.csv")

		delta_conf = getDeltaConf(df, gamma)

		f = Fitter(delta_conf, distributions=get_common_distributions(), bins=paramsDict["n_bins"])
		f.fit()
		df_errors_fitting = f.summary()
		df_errors_fitting["p-value"] = compute_p_value(f, delta_conf)
		df_errors_fitting["gamma"] = [gamma]*len(df_errors_fitting)

		plt.legend(frameon=False, fontsize=paramsDict["fontsize"]-2)
		plt.tick_params(axis='both', which='major', labelsize=paramsDict["fontsize"]-4)
		plt.title("Gamma: %s"%(gamma))

		if(paramsDict["shouldSave"]):
			plt.savefig("%s.jpg"%(filePath))
			plt.savefig("%s.eps"%(filePath))
			df_errors_fitting.to_csv(saveErrosPathLibrary, mode='a', header=not os.path.exists(saveErrosPathLibrary))	

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='UCB using Alexnet')
	parser.add_argument('--model_id', type=int, default=2, help='Model Id (default: 2)')
	args = parser.parse_args()

	root_path = "."
	results_path = os.path.join(root_path, "inference_exp_ucb_%s.csv"%(args.model_id))
	plotPath = "./delta_conf_bin_results"


	df_result = pd.read_csv(results_path)
	df_result = df_result.loc[:, ~df_result.columns.str.contains('^Unnamed')]
	df_result = df_result[df_result.conf_branch_2 > df_result.conf_branch_1]



	fontsize = 16
	n_bins = 100
	gamma_list = [0.5, 0.6, 0.7, 0.8, 0.9]
	shouldSave = True
	n_rank = 3

	paramsDict = {"fontsize": fontsize, "shouldSave": shouldSave, "plotPath": plotPath, "n_bins": n_bins, "n_rank": n_rank}

	expFittingDistributionsUsingLibrary(df_result, gamma_list, dist_list, paramsDict)