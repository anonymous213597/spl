import numpy as np
from utils import sample_batch
from optQ import Q_Awise_func
from copy import deepcopy

import numpy as np

class PEVI():
    def __init__(self, data, **train_args) -> None:
        self.pr_data = data
        self.train_args = train_args
        args = train_args['SSRL_args']
        pessimism_scale = train_args['pessimism_scale']
        self.A_set = np.unique(self.pr_data['actions'])
        self.estimate_optQ = Q_Awise_func(A_set=self.A_set, pessimism_scale=pessimism_scale, args=args)
        self.estimate_optQ.initialize(state=self.pr_data['observations'], 
                                      action=self.pr_data['actions'], 
                                      target=self.pr_data['rewards'] / (1 - train_args['gamma']))
        self.V_max = np.max(self.pr_data['rewards'])  / (1 - train_args['gamma'])
        pass

    def fit(self):
        batch_size = self.pr_data['actions'].size
        one_vec = np.ones((batch_size, 1))
        for i in range(self.train_args['n_epoch']):
            sampled_data, indices = sample_batch(self.pr_data, batch_size)
            obs, action, reward, next_obs, _ = sampled_data.values()
            Q_sa = np.hstack([self.estimate_optQ.Q_value(next_obs, one_vec*a) for a in self.A_set])
            max_next_Q = np.max(Q_sa, axis=1, keepdims=True)
            max_next_Q = np.clip(max_next_Q, -self.V_max, self.V_max)
            target = reward + self.train_args['gamma'] * max_next_Q
            coef_diff, pred_diff = self.estimate_optQ.step(obs, action, target)
            if min(pred_diff, coef_diff) <= self.train_args['eps']:
                if self.train_args['verbose']:
                    print("Stop at iter {} with diff in predict: {}".format(i+1, np.round(pred_diff, 8)))
                break
            if self.train_args['verbose'] and ((i+1) % self.train_args['print_freq'] == 0):
                print("Iter {} with difference in coef: {} and predict: {}".format(i+1, coef_diff, pred_diff))
    

class SSLFQI():
    def __init__(self, label_data, unlabel_data, rewardLB_estimate, transition_estimate, **train_args) -> None:
        self.rewardLB_estimate = rewardLB_estimate
        self.transition_estimate = transition_estimate
        if train_args['combine_data'] and (unlabel_data is not None) and (transition_estimate is not None):
            self.hybrid = True
            self.estimate_transition(unlabel_data)
        else:
            self.hybrid = False

        ## change to pessimism rewards
        if rewardLB_estimate is None:
            self.pr_data = deepcopy(label_data)
        else:
            self.estimate_rewardLB(label_data)
            new_label_data = deepcopy(label_data)
            new_label_data['rewards'] = self.rewardLB_estimate.estimate_lb_parametric(label_data['observations'], label_data['actions'])

            ## prepare datasets for training
            if unlabel_data is not None:
                self.pr_data = dict()
                for key in label_data.keys():
                    if key == "rewards":
                        if train_args['combine_data']:
                            self.pr_data[key] = np.vstack([
                                new_label_data[key], 
                                self.rewardLB_estimate.estimate_lb_parametric(
                                    unlabel_data['observations'], unlabel_data['actions']
                                )
                            ])
                        else:
                            self.pr_data[key] = self.rewardLB_estimate.estimate_lb_parametric(
                                unlabel_data['observations'], unlabel_data['actions']
                            )
                    else:
                        if train_args['combine_data']:
                            self.pr_data[key] = np.vstack([new_label_data[key], unlabel_data[key]])
                        else:
                            self.pr_data[key] = unlabel_data[key]
            else:
                self.pr_data = deepcopy(new_label_data)

        self.train_args = train_args
        self.A_set = np.unique(self.pr_data['actions'])
        args = train_args['SSRL_args']
        # self.estimate_optQ = Q_function(A_set=self.A_set, args=args)
        self.estimate_optQ = Q_Awise_func(A_set=self.A_set, args=args)
        self.estimate_optQ.initialize(state=self.pr_data['observations'], 
                                      action=self.pr_data['actions'], 
                                      target=self.pr_data['rewards'] / (1 - train_args['gamma']))
        self.V_max = np.max(self.pr_data['rewards'])  / (1 - train_args['gamma'])
        pass

    def estimate_transition(self, unlabel_data):
        obs = unlabel_data['observations']
        action = unlabel_data['actions']
        next_obs = unlabel_data['next_observations']
        self.transition_estimate.fit(obs, action, next_obs)
    
    def estimate_rewardLB(self, label_data):
        obs = label_data['observations']
        action = label_data['actions']
        rewards = label_data['rewards']
        self.rewardLB_estimate.fit(obs, action, rewards)

    def fit(self):
        # batch_size = self.train_args['batch_size']
        batch_size = self.pr_data['actions'].size
        mc_time = self.train_args['mc_time']
        one_vec = np.ones((batch_size, 1))
        if mc_time > 0:
            mc_next_obs = self.transition_estimate.sample(self.pr_data['observations'], self.pr_data['actions'], mc_time)
        for i in range(self.train_args['n_epoch']):
            sampled_data, indices = sample_batch(self.pr_data, batch_size)
            obs, action, reward, next_obs, _ = sampled_data.values()
            Q_sa = np.hstack([self.estimate_optQ.Q_value(next_obs, one_vec*a) for a in self.A_set])
            max_next_Q = np.max(Q_sa, axis=1, keepdims=True)
            if mc_time > 0:
                tmp_mc_next_obs = np.copy(mc_next_obs[:, indices, :])
                for j in range(mc_time):
                    ns = tmp_mc_next_obs[j, :, :]
                    Q_sa = np.hstack([self.estimate_optQ.Q_value(ns, one_vec*a) for a in self.A_set])
                    max_next_Q += np.max(Q_sa, axis=1, keepdims=True)
                max_next_Q /= (mc_time + 1)   # consider observation from label dataset
            max_next_Q = np.clip(max_next_Q, -self.V_max, self.V_max)
            target = reward + self.train_args['gamma'] * max_next_Q
            coef_diff, pred_diff = self.estimate_optQ.step(obs, action, target)
            if min(pred_diff, coef_diff) <= self.train_args['eps']:
                if self.train_args['verbose']:
                    print("Stop at iter {} with diff in predict: {}".format(i+1, np.round(pred_diff, 8)))
                break
            if self.train_args['verbose'] and ((i+1) % self.train_args['print_freq'] == 0):
                print("Iter {} with difference in coef: {} and predict: {}".format(i+1, coef_diff, pred_diff))
    

class SSLPlan():
    def __init__(self, label_data, unlabel_data, rewardLB_estimate, transition_estimate, **train_args) -> None:
        self.rewardLB_estimate = rewardLB_estimate
        self.transition_estimate = transition_estimate
        if unlabel_data is None:
            self.SSRL = False
        else:
            self.SSRL = True

        ## change to pessimism rewards
        if train_args['fit_reward']:
            self.estimate_rewardLB(label_data)
        new_label_data = deepcopy(label_data)
        new_label_data['rewards'] = self.rewardLB_estimate.estimate_lb_parametric(label_data['observations'], label_data['actions'])

        ## prepare datasets for training
        if self.SSRL:
            self.pr_data = dict()
            for key in label_data.keys():
                if key == "rewards":
                    self.pr_data[key] = np.vstack([
                        new_label_data[key], 
                        self.rewardLB_estimate.estimate_lb_parametric(unlabel_data['observations'], unlabel_data['actions'])
                    ])
                else:
                    self.pr_data[key] = np.vstack([new_label_data[key], unlabel_data[key]])
        else:
            self.pr_data = deepcopy(new_label_data)

        # estimate transition
        if train_args['fit_transition']:
            self.estimate_transition(deepcopy(self.pr_data))

        self.train_args = train_args
        self.A_set = np.unique(self.pr_data['actions'])
        args = train_args['SSRL_args']

        # initial Q function
        self.estimate_optQ = Q_Awise_func(A_set=self.A_set, args=args)
        self.estimate_optQ.initialize(state=self.pr_data['observations'], 
                                      action=self.pr_data['actions'], 
                                      target=self.pr_data['rewards'] / (1 - train_args['gamma']))
        self.V_max = np.max(self.pr_data['rewards'])  / (1 - train_args['gamma'])
        pass

    def estimate_transition(self, unlabel_data):
        obs = unlabel_data['observations']
        action = unlabel_data['actions']
        next_obs = unlabel_data['next_observations']
        self.transition_estimate.fit(obs, action, next_obs)
    
    def estimate_rewardLB(self, label_data):
        obs = label_data['observations']
        action = label_data['actions']
        rewards = label_data['rewards']
        self.rewardLB_estimate.fit(obs, action, rewards)
    
    def fit(self):
        batch_size = self.pr_data['actions'].size
        one_vec = np.ones((batch_size, 1))
        sampled_data, _ = sample_batch(self.pr_data, batch_size)
        for j in range(self.train_args['n_rollout']):
            obs, action, reward, next_obs, _ = sampled_data.values()
            for i in range(self.train_args['n_epoch']):
                Q_sa = np.hstack([self.estimate_optQ.Q_value(next_obs, one_vec*a) for a in self.A_set])
                max_next_Q = np.max(Q_sa, axis=1, keepdims=True)
                max_next_Q = np.clip(max_next_Q, -self.V_max, self.V_max)
                target = reward + self.train_args['gamma'] * max_next_Q
                coef_diff, pred_diff = self.estimate_optQ.step(obs, action, target)

                if min(pred_diff, coef_diff) <= self.train_args['eps']:
                    if self.train_args['verbose'] and j % 5 == 0:
                        print("(The {} batch samples) Stop at iter {} with diff in predict: {}".format(j+1, i+1, np.round(pred_diff, 8)))
                    break
            sampled_data['observations'] = next_obs
            ## epsilon-greedy policy:
            sampled_data['actions'] = self.A_set[np.argmax(Q_sa, axis=1).reshape(-1, 1)]
            random_explore_num = min(int(batch_size * self.train_args['epsilon']), 1)
            sampled_data['actions'][np.random.choice(range(batch_size), size=random_explore_num)] = np.random.choice(self.A_set, size=random_explore_num)
            # sampled_data['actions'] = np.random.choice(self.A_set, size=action.shape)   ## random policy
            sampled_data['rewards'] = self.rewardLB_estimate.estimate_lb_parametric(sampled_data['observations'], sampled_data['actions'])
            sampled_data['next_observations'] = self.transition_estimate.sample(sampled_data['observations'], sampled_data['actions'])[0]

