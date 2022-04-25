import numpy as np
import pandas as pd
import itertools, argparse
from tqdm import tqdm
import os, sys, random
from statistics import mode
import logging


def compute_reward(conf_branch, overhead):
	return conf_branch/overhead

def run_ucb(df, threshold_list, overhead, n_rounds, c, bin_lower, bin_upper, savePath, logPath, verbose):
	df = df.sample(frac=1)
	delta = 1e-10

	df_result = pd.read_csv(savePath) if(os.path.exists(savePath)) else pd.DataFrame()

	amount_arms = len(threshold_list)
	avg_reward_actions, n_actions = np.zeros(amount_arms), np.zeros(amount_arms)

	df_size = len(df)
	indices_list = np.arange(df_size)

	reward_actions = [[] for i in range(amount_arms)]
	
	for n_round in range(n_rounds):
		idx = random.choice(indices_list)
		row = df.iloc[[idx]]

		if (n_round < amount_arms):
			action = random.randint(0, amount_arms)
		else:
			q = avg_reward_actions + c*np.sqrt(np.log(n_round)/(n_actions+delta))
			action = np.argmax(q)

		threshold = threshold_list[action]

		conf_branch = row.conf_branch_1.item()
		conf_final = row.conf_branch_2.item()

		confidence_action = conf_branch if (action==0) else conf_final

		reward = compute_reward(conf_branch, overhead)

		n_actions[action] += 1

		reward_actions[action].append(reward)

		avg_reward_actions = np.mean(reward_actions, axis=1)

		optimal_reward = max (conf_branch/overhead, conf_final/overhead)

		inst_regret = optimal_reward - reward
		cumulative_regret += inst_regret
		cumulative_regret_list[n_round] = cumulative_regret

		inst_regret_list[n_round] = inst_regret
		selected_arm_list[n_round] = threshold

		if (n_round%100000 == 0):
			print("N Round: %s, Overhead: %s"%(n_round, overhead), file=open(logPath, "a"))

	result = {"selected_arm": selected_arm_list, "regret": inst_regret_list, 
	"overhead":[round(overhead, 2)]*n_rounds,
	"bin_lower": [round(bin_lower, 2)]*n_rounds, "bin_upper": [round(bin_upper, 2)]*n_rounds,
	"cumulative_regret": cumulative_regret_list}

	return result




def ucb_experiment(df, threshold_list, overhead_list, n_rounds, c, savePath, logPath, verbose):

	df_result = pd.DataFrame()
	
	bin_boundaries = np.arange(0, 1.1, 0.2)
	bin_lowers = bin_boundaries[:-1]
	bin_uppers = bin_boundaries[1:]

	for overhead in overhead_list:
		for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):

			logging.debug("Overhead: %s, Bin: [%s, %s]"%(overhead, bin_lower, bin_upper))

			df_temp = df[(df.delta_conf >= bin_lower) & (df.delta_conf <= bin_upper)] 
			
			if(len(df_temp.delta_conf.values) > 0):
				result = run_ucb(df_temp, threshold_list, overhead, n_rounds, c, bin_lower, bin_upper, savePath, logPath, verbose)
				df2 = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
				df_result = df_result.append(df2)
				df_result.to_csv(savePath)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='UCB using Alexnet')
	parser.add_argument('--model_id', type=int, default=2, help='Model Id (default: 2)')
	parser.add_argument('--c', type=float, default=1.0, help='Parameter c (default: 1.0)')
	parser.add_argument('--n_rounds', type=int, default=1000000, help='Model Id (default: 1M)')

	args = parser.parse_args()

	root_path = os.path.join(".")
	results_path = os.path.join(root_path, "inference_exp_ucb_%s.csv"%(args.model_id))
	df_result = pd.read_csv(results_path)
	df_result = df_result.loc[:, ~df_result.columns.str.contains('^Unnamed')]
	threshold_list = np.arange(0, 1.1, 0.1)
	overhead_list = np.arange(0, 1.1, 0.1)
	verbose = False
	savePath = os.path.join(".", "ucb_bin_delta_conf_result_c_%s_id_%s.csv"%(args.c, args.model_id))
	logPath = os.path.join(".", "logLEEMUltipleThresholds_%s.txt"%(args.model_id))

	logging.basicConfig(level=logging.DEBUG, filename=logPath, filemode="a+", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

	ucb_experiment(df_result, threshold_list, overhead_list, args.n_rounds, args.c, savePath, logPath, verbose)	