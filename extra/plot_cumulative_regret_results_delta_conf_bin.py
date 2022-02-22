import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os


def chunker(seq, size=500000):
	return np.array([seq[pos:pos + size] for pos in range(0, len(seq), size)])

def plotCumulativeRegretConfBin(df, overhead_list, bin_lowers, bin_uppers, paramsDict, savePath):
	for overhead in overhead_list:
		df_overhead = df[df.overhead == overhead]    

		fig, ax = plt.subplots()

		for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
			df_final = df_overhead[(df_overhead.bin_lower==bin_lower) & (df_overhead.bin_upper==bin_upper)]

			x = np.arange(len(df_final))
			y = df_final.cumulative_regret.values

			plt.plot(x, y, label="[%s, %s]"%(bin_lower, bin_upper))      
			plt.title("Overhead: %s"%(overhead))
			plt.ylabel("Cumulative Regret", fontsize=paramsDict["fontsize"])
			plt.xlabel("Epochs")
			plt.legend(frameon=False)


		plt.savefig(os.path.join(savePath, "cumulative_regret_delta_conf_bin_overhead_%s.pdf"%(overhead)))


result_conf_bin_path = "./ucb_bin_delta_conf_result_c_1.0.csv"
df_conf_bin = pd.read_csv(result_conf_bin_path)
df_conf_bin = df_conf_bin.loc[:, ~df_conf_bin.columns.str.contains('^Unnamed')] 
savePath = "./delta_conf_bin_results"

fontsize = 16

overhead_list = df_conf_bin.overhead.unique()
bin_lowers = df_conf_bin.bin_lower.unique()
bin_uppers = df_conf_bin.bin_upper.unique()


paramsDict = {"fontsize": fontsize}
plotCumulativeRegretConfBin(df_conf_bin, overhead_list, bin_lowers, bin_uppers, paramsDict, savePath)
