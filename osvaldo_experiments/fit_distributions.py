import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, argparse
from fitter import Fitter, get_common_distributions, get_distributions
from scipy import stats

def plotPDFDistribution(conf_branch, gamma, nr_branches, paramsDict):
	fig, ax = plt.subplots()
	plt.hist(conf_branch, bins=paramsDict["n_bins"], density=True, histtype='stepfilled', alpha=0.2)
	plt.title("Nr Branches: %s, gamma: %s"%(nr_branches, gamma), fontsize=paramsDict["fontsize"])
	plt.xlabel("Confidence on Branch", fontsize=paramsDict["fontsize"])
	ax.tick_params(axis='both', which='major', labelsize=paramsDict["fontsize"]-2)  

def plotMultipleDistributionPDFDistribution(df, gamma_list, nr_branches, n_bins, paramsDict):
	for gamma in gamma_list:

		if(nr_branches == 1):
			df_branch = df
		else:
			df_branch = df[df["conf_branch_%s"%(nr_branches-1)] < gamma]
			#for i in range(1, nr_branches):
			#  df = df[df["conf_branch_%s"%(nr_branches-1)] < gamma]
		#df_branch = df

	correct_branch, conf_branch = df["correct_branch_%s"%(nr_branches)], df["conf_branch_%s"%(nr_branches)]
	plotPDFDistribution(conf_branch, gamma, nr_branches, paramsDict)

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

def fitdist(data, dist):
	distribution = getattr(stats, dist)
	fitted = distribution.fit(data, floc=0.0)
	ks = kstest(data, dist, fitted)
	res = (dist, ks, *fitted)
	return res

def saveResults(df_list, savePath):

	for df in df_list:
		df.to_csv(savePath, mode='a', header=not os.path.exists(savePath))



def fitMultipleDistributions(df, gamma_list, nr_branches, dist_list, paramsDict):

	df_results_list = []

	for gamma in gamma_list:
		print("Nr Branches: %s, gamma: %s"%(nr_branches, gamma))
		if(nr_branches == 1):
			df_branch = df
		else:
			df_branch = df[df["conf_branch_%s"%(nr_branches-1)] < gamma]
			#for i in range(1, nr_branches):
			#  df = df[df["conf_branch_%s"%(nr_branches-1)] < gamma]
			#df_branch = df

		conf_branch = df_branch["conf_branch_%s"%(nr_branches)].values
		results_list = [fitdist(conf_branch, dist) for dist in dist_list]

		pd.options.display.float_format = '{:,.5f}'.format
		df_results = pd.DataFrame(results_list, columns=["distribution", "KS p-value", "param1", "param2", "param3", "param4", "param5", "param6"])
		df_results["gamma"] = len(df_results)*[gamma]
		df_results["nr_branches"] = len(df_results)*[nr_branches]
		df_results.sort_values(by=["KS p-value"], inplace=True, ascending=False)
		df_results.reset_index(inplace=True)
		df_results.drop("index", axis=1, inplace=True)
		df_results_list.append(df_results)

		#df_ks = df_results.loc[df_results["KS p-value"] > ALPHA]
		#print(df_results)
		plotFittedDist(df_results.iloc[:paramsDict["n_rank"], :], conf_branch, gamma, nr_branches, paramsDict)
		#plotFittedDist(df_ks, conf_branch.values, gamma, nr_branches, paramsDict)

		return df_results_list

def plotFittedDist(df, data, gamma, nr_branches, paramsDict):
	fig, ax = plt.subplots()

	# plot histogram of actual observations
	plt.hist(data, bins=paramsDict["n_bins"], density=True, histtype='stepfilled', alpha=0.2)

	plt.title("Nr Branches: %s, gamma: %s"%(nr_branches, gamma), fontsize=paramsDict["fontsize"])
	for i in df.index:
		dist_name = df.iloc[i, 0]
		D = getattr(stats, dist_name)
		p_value = df.iloc[i, 1]
		params = df.iloc[i, 2: -3].values
		params = [p for p in params if ~np.isnan(p)]

		# calibrate x-axis by finding the 1% and 99% quantiles in percent point function
		x = np.linspace(D.ppf(0.01, *params), D.ppf(0.99, *params), 100)

		# plot fitted distribution
		rv = D(*params)

		plt.plot(x, rv.pdf(x), label=dist_name)
		plt.xlim(0, 1)
		plt.legend(loc="best", frameon=False, fontsize=paramsDict["fontsize"])
	if (paramsDict["shouldSave"]):
		file_name = "pdf_%s_%s_branches_gamma_%s_%s_%s"%(paramsDict["model_name"], nr_branches, gamma, paramsDict["mode"], paramsDict["dataset_name"])
		plt.savefig(os.path.join(paramsDict["plotPath"], "eps", "%s.eps"%(file_name)) )
		plt.savefig(os.path.join(paramsDict["plotPath"], "jpg", "%s.jpg"%(file_name)) )



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

def expReliabilityDiagram(df, nr_branches_list, paramsDict):
	for nr_branches in nr_branches_list:
		correct_branch, conf_branch = df["correct_branch_%s"%(nr_branches)].values, df["conf_branch_%s"%(nr_branches)].values
		plotReliabilityDiagram(correct_branch, conf_branch, nr_branches, paramsDict)


def plotReliabilityDiagram(correct, confs, nr_branches, paramsDict):

	bin_boundaries = np.linspace(0, 1, paramsDict["n_bins"] + 1)
	bin_lowers = bin_boundaries[:-1]
	bin_uppers = bin_boundaries[1:]
	conf_list, acc_list = [], [] 

	bin_size = 1/n_bins
	positions = np.arange(0+bin_size/2, 1+bin_size/2, bin_size)
	ece = 0

	for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):

		in_bin = np.where((confs > bin_lower) & (confs <= bin_upper), True, False)
		prop_in_bin = np.mean(in_bin)
		confs_in_bin, correct_in_bin = confs[in_bin],correct[in_bin] 
		avg_confs_in_bin = sum(confs_in_bin)/len(confs_in_bin) if (len(confs_in_bin)>0) else 0
		avg_acc_in_bin = sum(correct_in_bin)/len(correct_in_bin) if (len(confs_in_bin)>0) else 0
		conf_list.append(avg_confs_in_bin), acc_list.append(avg_acc_in_bin)

	acc_list = np.array(acc_list)
	plt.style.use('ggplot')
	fig, ax = plt.subplots()
	gap_plt = ax.bar(positions, conf_list, width=bin_size, edgecolor="red", color="red", alpha=0.3, label="Gap", linewidth=2, zorder=2)
	output_plt = ax.bar(positions, acc_list, width=bin_size, edgecolor="black", color="blue", label="Outputs", zorder = 3)
	ax.set_aspect('equal')
	ax.plot([0,1], [0,1], linestyle = "--")
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.set_title("Nr Branches: %s"%(nr_branches), fontsize=paramsDict["fontsize"] - 2)
	ax.legend(handles = [gap_plt, output_plt], frameon=False)
	ax.set_ylabel("Accuracy", fontsize=paramsDict["fontsize"], color="black")
	ax.set_xlabel("Confidence", fontsize=paramsDict["fontsize"], color="black")
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

	if(paramsDict["shouldSave"]):
		plt.savefig(os.path.join(paramsDict["plotPath"], "eps", "reliability_diagram_branch_%s_%s.eps"%(nr_branches, paramsDict["mode"], paramsDict["dataset_name"]) ))
		plt.savefig(os.path.join(paramsDict["plotPath"], "jpg", "reliability_diagram_branch_%s_%s.jpg"%(nr_branches, paramsDict["mode"], paramsDict["dataset_name"]) ))


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
	nr_branches_list = np.arange(3, n_exits+1)

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
		fitMultipleDistributions(df, gamma_list, nr_branches, dist_list, paramsDict)
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
