import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, argparse
from fitter import Fitter, get_common_distributions, get_distributions
from scipy import stats


def kstest(data, dist_name, paramtup, ksN=100):
	p_value = stats.kstest(data, dist_name, paramtup, ksN)[1]   # return p-value
	return p_value             # return p-value

def fitdist(data, dist):
  distribution = getattr(stats, dist)
  fitted = distribution.fit(data, floc=0.0)
  ks = kstest(data, dist, fitted)
  res = (dist, ks, *fitted)
  return res


def getDeltaConf(df, gamma):

	df_branch = df[df.conf_branch_1 <= gamma]
	#df_branch = df
	conf_branch = df_branch.conf_branch_1.values
	conf_final = np.maximum(df_branch.conf_branch_2.values, conf_branch)
	delta_conf = conf_final - conf_branch
	return delta_conf


def expFittingDistributions(df, gamma_list, dist_list, paramsDict):
	df_results_list = []
	for gamma in gamma_list:
		paramsDict["filename"] = "pdf_distribution_%s"%(gamma)

		delta_conf = getDeltaConf(df, gamma)

		results_list = [fitdist(delta_conf, dist) for dist in dist_list]

		pd.options.display.float_format = '{:,.5f}'.format
		df_results = pd.DataFrame(results_list, columns=["distribution", "KS p-value", "param1", "param2", "param3", "param4", "param5", "param6"])
		df_results["gamma"] = len(df_results)*[gamma]
		df_results.sort_values(by=["KS p-value"], inplace=True, ascending=False)
		df_results.reset_index(inplace=True)
		df_results.drop("index", axis=1, inplace=True)
		df_results_list.append(df_results)

		plotFittingDistributions(df_results.iloc[:paramsDict["n_rank"], :], delta_conf, gamma, paramsDict)
	return df_results_list

def plotFittingDistributions(df, data, gamma, paramsDict):
	fig, ax = plt.subplots()
  
	# plot histogram of actual observations
	plt.hist(data, bins=paramsDict["n_bins"], density=True, histtype='stepfilled', alpha=0.2)

	plt.title("Gamma: %s"%(gamma), fontsize=paramsDict["fontsize"])
	for i in df.index:

		dist_name = df.iloc[i, 0]
		D = getattr(stats, dist_name)
		p_value = df.iloc[i, 1]
		params = df.iloc[i, 2: -2].values
		params = [p for p in params if ~np.isnan(p)]

		# calibrate x-axis by finding the 1% and 99% quantiles in percent point function
		x = np.linspace(D.ppf(0.01, *params), D.ppf(0.99, *params), 100)
    
		# plot fitted distribution
		rv = D(*params)

		plt.plot(x, rv.pdf(x), label=dist_name)
		plt.xlim(0, 1)
		plt.legend(loc="best", frameon=False, fontsize=paramsDict["fontsize"])
		if (paramsDict["shouldSave"]):
			filePath = os.path.join(paramsDict["plotPath"], paramsDict["filename"])
			plt.savefig("%s.jpg"%(filePath))
			plt.savefig("%s.eps"%(filePath))

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='UCB using Alexnet')
	parser.add_argument('--model_id', type=int, default=2, help='Model Id (default: 2)')
	args = parser.parse_args()

	root_path = "."
	results_path = os.path.join(root_path, "inference_exp_ucb_%s.csv"%(args.model_id))


	df_result = pd.read_csv(results_path)
	df_result = df_result.loc[:, ~df_result.columns.str.contains('^Unnamed')]
	df_result = df_result[df_result.conf_branch_2 > df_result.conf_branch_1]

	plotPath = "./delta_conf_bin_results"


	fontsize = 16
	n_bins = 100
	gamma_list = [0.5, 0.6, 0.7, 0.8, 0.9]
	shouldSave = True
	n_rank = 3
	dist_list = ['alpha','anglit','arcsine','beta','betaprime','bradford','burr','burr12','cauchy','chi','chi2','cosine','dgamma','dweibull','expon','exponnorm','exponweib','exponpow','f','fatiguelife','fisk','foldcauchy','foldnorm','genlogistic','genpareto','gennorm','genexpon','genextreme','gausshyper','gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r','gumbel_l','halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant','invgamma','invgauss','invweibull','johnsonsb','johnsonsu','kstwobign','laplace','levy','levy_l','logistic','loggamma','loglaplace','lomax','maxwell','mielke','nakagami','ncx2','ncf','nct','norm','pareto','pearson3','powerlaw','powerlognorm','powernorm','rdist','reciprocal','rayleigh','rice','recipinvgauss','semicircular','t','triang','truncexpon','truncnorm','tukeylambda','uniform','vonmises','vonmises_line','wald','weibull_min','weibull_max']

	paramsDict = {"fontsize": fontsize, "shouldSave": shouldSave, "plotPath": plotPath, "n_bins": n_bins, "n_rank": n_rank}

	expFittingDistributions(df_result, gamma_list, dist_list, paramsDict)