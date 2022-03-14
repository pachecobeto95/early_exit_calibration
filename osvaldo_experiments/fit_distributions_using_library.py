import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, argparse
from fitter import Fitter, get_common_distributions, get_distributions
from scipy import stats

def compute_p_value(f, data):
	distribution_name_list = f.summary().iloc[:, 0].index.values
	p_value_list = []
	for dist_name in distribution_name_list:
		params = f.fitted_param[dist_name]
		p_value = kstest(data, dist_name, params)
		p_value_list.append(p_value)

	return p_value



def kstest(data, dist_name, paramtup):
	p_value = stats.kstest(data, dist_name, paramtup, 100)[1]   # return p-value
	return p_value             # return p-value


def saveResults(df_list, savePath):

	for df in df_list:
		df.to_csv(savePath, mode='a', header=not os.path.exists(savePath))



def expFittingDistributionsUsingLibrary(df, gamma_list, nr_branches, dist_list, paramsDict, saveResultsPath):
	for gamma in gamma_list:

		file_name = "pdf_%s_%s_branches_gamma_%s_%s_using_library"%(paramsDict["model_name"], nr_branches, gamma, paramsDict["mode"])
		saveErrosPathLibrary = os.path.join(saveResultsPath, "results_error_library_%s.csv"%(paramsDict["model_name"]))

		df_branch = df[df["conf_branch_%s"%(nr_branches-1)] < gamma]

		correct_branch, conf_branch = df_branch["correct_branch_%s"%(nr_branches)], df_branch["conf_branch_%s"%(nr_branches)]

		f = Fitter(conf_branch, distributions=get_distributions(), bins=100)
		f.fit()
		df_errors_fitting = f.summary()
		df_errors_fitting["p-value"] = compute_p_value(f, conf_branch.values)
		df_errors_fitting["nr_branches"] = [nr_branches]*len(df_errors_fitting)
		df_errors_fitting["gamma"] = [gamma]*len(df_errors_fitting)

		plt.legend(frameon=False, fontsize=paramsDict["fontsize"]-2)
		plt.tick_params(axis='both', which='major', labelsize=paramsDict["fontsize"]-4)
		plt.title("Nr Branches: %s, Gamma: %s"%(nr_branches, gamma))

		if(paramsDict["shouldSave"]):
			plt.savefig(os.path.join(paramsDict["plotPath"], "eps", "%s.eps"%(file_name)))
			plt.savefig(os.path.join(paramsDict["plotPath"], "jpg", "%s.jpg"%(file_name)))
			df_errors_fitting.to_csv(saveErrosPathLibrary, mode='a', header=not os.path.exists(saveErrosPathLibrary))


def main(args):
	DIR_NAME = os.path.dirname(__file__)


	data_path = os.path.join(DIR_NAME, "data")
	plotPath = os.path.join(DIR_NAME, "plots")
	saveResultsPath = os.path.join(DIR_NAME, "results")

	df_path = os.path.join(data_path, "no_calib_results_%s_early_exit_%s_id_%s_%s.csv"%(args.model_name, args.dataset_name, args.model_id, args.mode))

	df = pd.read_csv(df_path)
	df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
	df = df[df.p_tar==0.8]

	n_exits = 6
	#nr_branches_list = np.arange(3, n_exits+1)
	nr_branches_list = [2, 5, 6]

	gamma_list = [0.5, 0.6, 0.7, 0.8, 0.9]
	n_bins_hist = 100
	n_bins = 15
	fontsize = 16
	n_rank = 3
	shouldSave = True
	mode = "random"
	ksN = 100           # Kolmogorov-Smirnov KS test for goodness of fit: samples
	ALPHA = 0.05        # significance level for hypothesis test

	paramsDict = {"fontsize": fontsize, "shouldSave": shouldSave, "plotPath": plotPath, "mode": mode, "dataset_name": args.dataset_name,  
	"model_name": args.model_name, "n_bins": n_bins_hist, "n_rank": n_rank}
	dist_list = ['alpha','anglit','arcsine','beta','betaprime','bradford','burr','burr12','cauchy','chi','chi2','cosine','dgamma','dweibull','expon','exponnorm','exponweib','exponpow','f','fatiguelife','fisk','foldcauchy','foldnorm','genlogistic','genpareto','gennorm','genexpon','genextreme','gausshyper','gamma','gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r','gumbel_l','halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant','invgamma','invgauss','invweibull','johnsonsb','johnsonsu','kstwobign','laplace','levy','levy_l','logistic','loggamma','loglaplace','lognorm','lomax','maxwell','mielke','nakagami','ncx2','ncf','nct','norm','pareto','pearson3','powerlaw','powerlognorm','powernorm','rdist','reciprocal','rayleigh','rice','recipinvgauss','semicircular','t','triang','truncexpon','truncnorm','tukeylambda','uniform','vonmises','vonmises_line','wald','weibull_min','weibull_max']

	for nr_branches in nr_branches_list:
		expFittingDistributionsUsingLibrary(df, gamma_list, nr_branches, dist_list, paramsDict, saveResultsPath)

if (__name__ == "__main__"):
	# Input Arguments. Hyperparameters configuration
	parser = argparse.ArgumentParser(description="Evaluating early-exits DNNs perfomance")

	parser.add_argument('--dataset_name', type=str, default="caltech256", 
		choices=["caltech256", "cifar10"], help='Dataset name (default: Caltech-256)')

	parser.add_argument('--model_name', type=str, default="mobilenet", 
		choices=["mobilenet", "resnet18", "resnet152", "vgg16"], help='DNN model name (default: MobileNet)')

	parser.add_argument('--split_ratio', type=float, default=0.2, help='Split Ratio')
	parser.add_argument('--model_id', type=int, help='Model id')
	parser.add_argument('--mode', type=str, default="ft", choices=["ft", "scratch"], help='EE training mode')

	args = parser.parse_args()

	main(args)
