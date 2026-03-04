import numpy as np
from scipy.special import expit

class Simulator:
    def save_init_state(self, num):
        init_state = np.random.normal(size=(num, self.dim_state), loc=0.0, scale=1.0)
        return init_state

    def save_s2action_model(self, state):
        pa = np.mean(state, axis=1, keepdims=True)
        pa = expit(0.1*pa)     
        pa = np.hstack([0.5*pa, 1-pa, 0.5*pa])
        return pa

    def save_reward_std(self, reward_mean):
        return np.copy(np.absolute(reward_mean))

    def save_sa2reward_model(self, state, action, random):
        rmean = 10 * np.mean(state, axis=1, keepdims=True) * action   
        rstd = self.reward_std
        if random:
            reward = np.random.normal(size=rmean.shape, loc=rmean, scale=rstd)
            non_optimal_reward = np.random.normal(size=rmean.shape, loc=0.0, scale=0.64)
            reward[rmean <= 0.0] = rmean[rmean <= 0.0] + non_optimal_reward[rmean <= 0.0]
        else:
            reward = rmean
        return reward

    def save_sa2nextstate_model(self, state, action, random):
        obs_dim = state.shape[1]
        num = state.shape[0]
        next_obs_mean = (self.transition_mat @ state.T).T
        next_obs_mean = np.vstack([(self.transition_mat * action[i]) @ state[i, :].T for i in range(num)])
        if random:
            next_obs = [np.random.multivariate_normal(size=1, mean=next_obs_mean[i, :], cov=0.01*np.eye(obs_dim)) for i in range(num)]  
            next_obs = np.vstack(next_obs)
        else:
            next_obs = next_obs_mean
        return next_obs

    def __init__(self, model_type='save', policy=None, dim_state=3, seed=42):
        self.dim_state = dim_state
        if model_type == 'save':
            self.init_state_model = self.save_init_state
            if policy is None:
                self.s2action_model = self.save_s2action_model
            else:
                self.s2action_model = policy
            self.reward_std = 0.1
            self.sa2reward_model = self.save_sa2reward_model
            decay_mat = np.absolute(np.arange(dim_state).reshape(-1, 1) - np.arange(dim_state).reshape(1, -1))
            transition_mat = np.power(np.array(0.3), decay_mat)
            transition_mat = transition_mat / np.sum(transition_mat, axis=1)
            np.random.seed(seed); random_sign = 2 * np.random.binomial(n=1, p=0.5, size=(dim_state, dim_state)) - 1
            self.transition_mat = random_sign * transition_mat
            self.sa2nextstate_model = self.save_sa2nextstate_model
        elif model_type == "standard":
            pass
        elif model_type == "toy":
            pass
        else:
            pass

        self.trajectory_list = []
        self.target_policy_trajectory_list = []
        self.target_policy_state_density_list = None
        self.stationary_behaviour_policy_state_density = None
        pass

    def sample_init_state(self, num):
        init_state = self.init_state_model(num)
        return init_state

    def logistic_sampler(self, prob):
        if prob.shape[1] == 1:
            random_y = np.random.binomial(n=1, p=prob, size=prob.shape)
            random_y = 2 * random_y - 1
        else:
            prob_dim = prob.shape[1]
            num = prob.shape[0]
            options = np.arange(-(prob_dim>>1), ((prob_dim>>1) + 1))
            if prob_dim % 2 == 0:
                options = options[options != 0]
            random_y = [np.random.choice(a=options, size=1, p=prob[i, :]).reshape(1, -1) for i in range(num)]
            random_y = np.vstack(random_y)
        return random_y

    def sample_s2action(self, state, random=True):
        if random:
            random_action = self.logistic_sampler(self.s2action_model(state))
        else:
            random_action = self.s2action_model(state)
        return random_action

    def sample_sa2reward(self, state, action, random=True):
        random_reward = self.sa2reward_model(state, action, random=random)
        return random_reward

    def sample_sa2nextstate(self, state, action, random=True):
        random_next_state = self.sa2nextstate_model(
            state, action, random=random)
        return random_next_state

    def sample_trajectory(self, num_trajectory, num_time, policy=None, seed=1, burn_in_time=50, iid_tuple=True, random_reward_opt=True, random_trans_opt=True):
        raw_num_time = num_time
        np.random.seed(seed)
        if burn_in_time > 0:
            burn_in = True
            num_time += burn_in_time
        else:
            burn_in = False

        init_state = self.sample_init_state(num_trajectory)
        random_state = np.zeros((num_time+1, num_trajectory, self.dim_state))
        random_action = np.zeros((num_time, num_trajectory, 1))
        random_reward = np.zeros((num_time, num_trajectory, 1))

        random_state[0, :, :] = init_state
        for i in range(num_time):
            if policy is None:
                random_action[i, :, :] = self.sample_s2action(random_state[i, :, :], random=True)
            else:
                random_action[i, :, :] = policy(random_state[i, :, :])
            random_reward[i, :, :] = self.sample_sa2reward(random_state[i, :, :], random_action[i, :, :], random=random_reward_opt)
            random_state[i+1, :, :] = self.sample_sa2nextstate(random_state[i, :, :], random_action[i, :, :], random=random_trans_opt)
            pass
        
        if burn_in:
            valid_index = range(burn_in_time, num_time+1)
            random_state = random_state[valid_index, :, :]
            valid_index = range(burn_in_time, num_time)
            random_action = random_action[valid_index, :, :]
            random_reward = random_reward[valid_index, :, :]

        random_trajectory = {
            'observations': random_state, 
            'actions': random_action, 
            'rewards': random_reward, 
        }

        random_next_state = random_state[1:, :, :]
        random_state = random_state[:(-1), :, :]
        if iid_tuple:
            random_trajectory = {
                'observations': np.vstack([random_state[i, :, :] for i in range(raw_num_time)]),
                'actions': np.vstack([random_action[i, :, :] for i in range(raw_num_time)]),
                'rewards': np.vstack([random_reward[i, :, :] for i in range(raw_num_time)]),
                'next_observations': np.vstack([random_next_state[i, :, :] for i in range(raw_num_time)]),
            }
            pool_size = random_trajectory['actions'].shape[0]
            random_trajectory['terminals'] = np.zeros((pool_size, 1))
        return random_trajectory
    
def opt_policy(state):
    action = np.sign(np.mean(state, axis=1, keepdims=True))
    return action
