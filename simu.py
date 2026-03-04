# %%
from linearEnv import Simulator, opt_policy
from SSRL import SSLFQI, PEVI
from rewardLB import MLRewardLB, PDSRewardLB
import numpy as np
from utils import remove_data_by_action, eval_policy
import pandas as pd
from tqdm import trange

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
parser = argparse.ArgumentParser(description='Linear Environment Args')
parser.add_argument('--case', type=str, default='FULL')
parser.add_argument('--ratio', type=int, default=10)
args = parser.parse_args()

CASE = args.case
UNLABEL_SCALE = args.ratio
print("Case {} with ratio {}".format(CASE, UNLABEL_SCALE))

RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)
SIMU_NREP = 100
SE_MULTIPLIER = 1.0 / np.sqrt(SIMU_NREP)
NUM_LIST = np.array([2, 4, 8, 16, 32, 64, 128])
RIDGE_PENALTY = 128.0

TIME = 30
state_dim = 2
pevi_strain_args = {
    'n_epoch': 500, 'eps': 1e-2, 'batch_size':max(NUM_LIST)*TIME, 'gamma': 0.99, 
    'verbose': False, 'print_freq': 20, 'pessimism_scale': 2.0,
    'SSRL_args': {'model_type': "linear", 'trans_type': 'poly', 'poly_degree': 1, 'l2_penalty': RIDGE_PENALTY},
}
train_args = {
    'n_epoch': 500, 'eps': 1e-6, 'batch_size':max(NUM_LIST)*TIME, 'combine_data': True, 'gamma': 0.99, 
    'mc_time': 0, 'verbose': False, 'print_freq': 20,
    'rewardLB_args': {'model_type': "linear", 'trans_type': 'poly', 'poly_degree': 2, 'rbf_feature_num': 12, }, 
    'trans_args': {''}, 
    'SSRL_args': {'model_type': "linear", 'trans_type': 'poly', 'poly_degree': 1, 'l2_penalty': RIDGE_PENALTY},
}
THRESHOLD = 0.3 if CASE == "DIFF2" else 0.9
MODEL = None

def est_opt_policy(state):
    Q_sa = np.hstack([MODEL.estimate_optQ.Q_value(state, np.full((state.shape[0], 1), a)) for a in MODEL.A_set])
    max_a_index = np.argmax(Q_sa, axis=1)
    action = np.array([MODEL.A_set[a_index] for a_index in max_a_index]).reshape(-1, 1)
    return action

simulator = Simulator(dim_state=state_dim)
OPT_VALUE = eval_policy(simulator, opt_policy, mean_only=True)
RESULT = {
    'NoShare': [], 
    'PNoShare': [], 
    'PL': [], 
    'UDS': [],
    'PDS': [],
    'SPL': [],
}
METHOD = list(RESULT.keys())

for j, num in enumerate(NUM_LIST):
    print("Number of trajectories: {}".format(num))
    for i in trange(SIMU_NREP):
        label_data = simulator.sample_trajectory(num_time=TIME, num_trajectory=num, iid_tuple=True, seed=i)
        if CASE == "DIFF2":
            label_data, retain_num = remove_data_by_action(label_data, remove_action=[0], retain_prop=0.2)
        else:
            label_data, retain_num = remove_data_by_action(label_data, remove_action=[-1, 0, 1], retain_prop=1.0)
        U_NUM = int(UNLABEL_SCALE * (label_data['actions'].size / TIME))
        unlabel_data = simulator.sample_trajectory(num_time=TIME, num_trajectory=U_NUM, iid_tuple=True, seed=2024 + i)
        if CASE == "DIFF":
            unlabel_data, retain_num = remove_data_by_action(unlabel_data, remove_action=[-1, 1], retain_prop=0.0)
        if 'NoShare' in METHOD:
            no_share_data = dict()
            for key in label_data.keys():
                no_share_data[key] = np.copy(label_data[key])
            ssl = SSLFQI(no_share_data, unlabel_data=None, rewardLB_estimate=None, transition_estimate=None, **train_args)
            ssl.fit()
            MODEL = ssl
            RESULT['NoShare'].append(OPT_VALUE - eval_policy(simulator, est_opt_policy, mean_only=True))
            del MODEL           
        if 'PNoShare' in METHOD:
            no_share_data = dict()
            for key in label_data.keys():
                no_share_data[key] = np.copy(label_data[key])
            reward_lb = MLRewardLB(train_args['rewardLB_args'], pessimism_scale=2.0)
            reward_lb.fit(no_share_data['observations'], no_share_data['actions'], no_share_data['rewards'])
            no_share_data['rewards'] = reward_lb.compute_mean_lb(no_share_data['observations'], no_share_data['actions'])
            ssl = SSLFQI(no_share_data, unlabel_data=None, rewardLB_estimate=None, transition_estimate=None, **train_args)
            ssl.fit()
            MODEL = ssl
            RESULT['PNoShare'].append(OPT_VALUE - eval_policy(simulator, est_opt_policy, mean_only=True))
            del MODEL
        if 'PL' in METHOD:
            reward_lb = MLRewardLB(train_args['rewardLB_args'], pessimism_scale=0.0)
            reward_lb.fit(label_data['observations'], label_data['actions'], label_data['rewards'])
            unlabel_data['rewards'] = reward_lb.compute_mean_lb(unlabel_data['observations'], unlabel_data['actions'])
            pred_data = dict()
            for key in label_data.keys():
                pred_data[key] = np.vstack([label_data[key], unlabel_data[key]])
            ssl = SSLFQI(pred_data, unlabel_data=None, rewardLB_estimate=None, transition_estimate=None, **train_args)
            ssl.fit()
            MODEL = ssl
            RESULT['PL'].append(OPT_VALUE - eval_policy(simulator, est_opt_policy, mean_only=True))
            del MODEL
        if 'UDS' in METHOD:
            unlabel_data['rewards'] *= 0
            unlabel_data['rewards'] += np.min(label_data['rewards'])
            uds_data = dict()
            for key in label_data.keys():
                uds_data[key] = np.vstack([label_data[key], unlabel_data[key]])
            ssl = SSLFQI(uds_data, unlabel_data=None, rewardLB_estimate=None, transition_estimate=None, **train_args)
            ssl.fit()
            MODEL = ssl
            RESULT['UDS'].append(OPT_VALUE - eval_policy(simulator, est_opt_policy, mean_only=True))
            del MODEL  
        if 'PDS' in METHOD:
            reward_lb = PDSRewardLB(train_args['rewardLB_args'], pessimism_scale=2.0)
            reward_lb.fit(label_data['observations'], label_data['actions'], label_data['rewards'])
            unlabel_data['rewards'] = reward_lb.compute_lb(unlabel_data['observations'], unlabel_data['actions'])
            pess_data = dict()
            for key in label_data.keys():
                pess_data[key] = np.vstack([label_data[key], unlabel_data[key]])
            ssl = PEVI(pess_data, **pevi_strain_args)
            ssl.fit()
            MODEL = ssl
            RESULT['PDS'].append(OPT_VALUE - eval_policy(simulator, est_opt_policy, mean_only=True))
            del MODEL
        if 'SPL' in METHOD:
            reward = MLRewardLB(train_args={'model_type': "forest", 'min_samples_leaf': max(int(0.01*num*TIME), 5)}, pessimism_scale=0.0)
            reward.fit(label_data['observations'], label_data['actions'], label_data['rewards'])
            reward.compute_ppi_se2(
                label_data['observations'], label_data['actions'], label_data['rewards'],
                unlabel_data['observations'], unlabel_data['actions'], 
            )
            # Apply PPI correction to labeled data
            ppi_std = reward.predict_ppi_se(label_data['observations'], label_data['actions'])
            ppi_mean = reward.predict_ppi_mean(label_data['observations'], label_data['actions'])
            label_data['rewards'] = ppi_mean - pevi_strain_args['pessimism_scale'] * ppi_std
            
            # Apply PPI correction to unlabeled data
            ppi_std = reward.predict_ppi_se(unlabel_data['observations'], unlabel_data['actions'])
            select_index = (ppi_std <= np.quantile(ppi_std, q=THRESHOLD)).flatten()
            ppi_mean = reward.predict_ppi_mean(unlabel_data['observations'], unlabel_data['actions'])
            unlabel_data['rewards'] = ppi_mean - pevi_strain_args['pessimism_scale'] * ppi_std

            for key in label_data.keys():
                estimated_data[key] = np.vstack([label_data[key], unlabel_data[key][select_index]])
            ssl = SSLFQI(estimated_data, unlabel_data=None, rewardLB_estimate=None, transition_estimate=None, **train_args)
            ssl.fit()
            MODEL = ssl
            RESULT['SPL'].append(OPT_VALUE - eval_policy(simulator, est_opt_policy, mean_only=True))
            del MODEL 
    RESULT_DF = pd.DataFrame(data=RESULT)
    RESULT_DF['num'] = [item for item in NUM_LIST[range(j+1)] for _ in range(SIMU_NREP)]
    RESULT_DF.to_csv("{}/FQI_trajs_{}_{}_{}_{}.csv".format(RESULT_DIR, CASE, str(UNLABEL_SCALE), str(THRESHOLD), "_".join(METHOD)), index=False)
