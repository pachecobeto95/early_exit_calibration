import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import os, sys, random
from statistics import mode


def compute_reward(conf_branch, arm, delta_conf, overhead):
  if(conf_branch >= arm):
    return 0
  else:
    return delta_conf - overhead
  
def get_row_data(row, threshold):
  conf_branch = row.conf_branch_1.item()
  conf_final = row.conf_branch_2.item()
  if(conf_final >= threshold):
    delta_conf = conf_final - conf_branch
    return conf_branch, delta_conf
  else:
    conf_list = [conf_branch, conf_final]
    delta_conf = max(conf_list) - conf_branch 
    return conf_branch, delta_conf

def ucb_run_resampling(df, threshold_list, overhead, label, n_rounds, c, verbose):
  if (label != "all"):
    df = df[df.label == label]
  
  df = df.sample(frac=1)
  delta = 1e-10

  avg_reward_actions, n_actions = np.zeros(len(threshold_list)), np.zeros(len(threshold_list))
  reward_actions = [[] for i in range(len(threshold_list))]
  cum_regret = 0
  inst_regret_list = []
  t = 0
  selected_arm_list = []

  for n_round in tqdm(range(n_rounds)):
    idx = random.choice(np.arange(len(df)))
    row = df.iloc[[idx]]
    #print(row.conf_branch_1.item(), row.conf_branch_2.item())

    if (t < len(threshold_list)):
      action = t
    else:
      q = avg_reward_actions + c*np.sqrt(np.log(t)/(n_actions+delta))
      action = np.argmax(q)

    threshold = threshold_list[action]

    conf_branch, delta_conf = get_row_data(row, threshold)
    
    reward = compute_reward(conf_branch, threshold, delta_conf, overhead)
        
    n_actions[action] += 1
    t += 1

    reward_actions[action].append(reward)
    
    avg_reward_actions = np.array([sum(reward_actions[i])/n_actions[i] for i in range(len(threshold_list))])
    optimal_reward = max(0, delta_conf - overhead)

    inst_regret = optimal_reward - reward

    print(n_actions, inst_regret, threshold, delta_conf, reward)
    #cum_regret += inst_regret
    inst_regret_list.append(inst_regret)
    selected_arm_list.append(threshold)

  result = {"selected_arm": selected_arm_list, "regret": inst_regret_list, "cum_regret":cum_regret_list, 
            "label":[label]*len(inst_regret_list), "overhead":[overhead]*len(inst_regret_list)}
  return result


def ucb_experiment(df, threshold_list, overhead_list, label_list, n_round, c, savePath, verbose=False):
  df_result = pd.DataFrame()

  config_list = list(itertools.product(*[label_list, overhead_list]))    
  
  for label, overhead in tqdm(config_list):
    result = ucb_run_resampling(df, threshold_list, overhead, label, n_round, c, verbose)
    df2 = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
    df_result = df_result.append(df2)
    df_result.to_csv(savePath)


model_id = 2
root_path = os.path.join(".")
results_path = os.path.join(root_path, "inference_exp_ucb_%s.csv"%(model_id))
df_result = pd.read_csv(results_path)
df_result = df_result.loc[:, ~df_result.columns.str.contains('^Unnamed')]
threshold_list = np.arange(0, 1.1, 0.1)
overhead_list = np.arange(0, 1.1, 0.1)
n_rounds = 100000
verbose = False
label_list = df_result.label.unique()
c = 1.0
savePath = os.path.join(".", "ucb_result_c_%s.csv"%(c))

ucb_experiment(df_result, threshold_list, overhead_list, label_list, n_rounds, c, savePath, verbose)