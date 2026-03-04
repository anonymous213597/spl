from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import numpy as np

class Q_Awise_func:
    def __init__(self, A_set, pessimism_scale=0.0, time_difference=None, args=None):
        if time_difference is None:
            self.time_difference = np.ones(time_difference)  # just skip it first
        else:
            self.time_difference = np.copy(time_difference)

        self.pessimism_scale = pessimism_scale
        self.train_inv_Lambda = None
        self.model_type = args['model_type']
        if self.model_type == "linear":
            self.A_set = A_set.tolist()
            self.base_l2_penalty = args['l2_penalty']

            if args['trans_type'] == "rbf":
                self.feature_trans = RBFSampler(random_state=1, n_components=args['rbf_feature_num'])
            elif args['trans_type'] == "poly":
                self.feature_trans = PolynomialFeatures(degree=args['poly_degree'], interaction_only=True, include_bias=True)
            elif args['trans_type'] == "none":
                self.feature_trans = None
            self.q_model = {a: Ridge(solver='lsqr', alpha=args['l2_penalty'], random_state=1, fit_intercept=False) for a in self.A_set}
        elif self.model_type == "forest":
            self.q_model = RandomForestRegressor(random_state=1, min_samples_leaf=args['min_samples_leaf'])
        else:
            pass

        self.iteration_time = 0
        self.q_model_list = []
        self.pred_diff = []
        self.args = args
        pass

    def _get_adaptive_l2_penalty(self, n_samples):
        if n_samples <= 0:
            return self.base_l2_penalty
        adaptive_penalty = self.base_l2_penalty / np.sqrt(n_samples)
        min_penalty = max(0.01, self.base_l2_penalty / 100.0)
        return max(adaptive_penalty, min_penalty)

    def linear_fit(self, state_feat, action, target):
        error = np.empty(action.shape)
        x = np.copy(state_feat)
        for a in self.A_set:
            a_indices = np.where(action == a)[0]
            if a_indices.size > 0:
                x_a = x[a_indices, :]
                y_a = target[a_indices, :]
            else:
                x_a = x[0, :].reshape(1, -1)
                y_a = np.array([[self.worst_value]])
            self.q_model[a].fit(x_a, y_a)
            error[a_indices, :] = y_a - self.q_model[a].predict(x_a).reshape(-1, 1)
        return error
    
    def linear_predict(self, model, state_feat, action):
        predict = np.empty(action.shape)
        x = np.copy(state_feat)
        for a in self.A_set:
            a_indices = np.where(action == a)[0]
            if a_indices.size > 0:
                x_a = x[a_indices, :]
                predict[a_indices, :] = model[a].predict(x_a).reshape(-1, 1)
        return predict

    def linear_uncertainty(self, model, state_feat, action):
        uncertainty = np.empty(action.shape)
        x = np.copy(state_feat)
        for a in self.A_set:
            a_indices = np.where(action == a)[0]
            if a_indices.size > 0:
                x_a = x[a_indices, :]
                if self.train_inv_Lambda is None:
                    self.train_inv_Lambda = model[a].get_params()['alpha'] * np.eye(x_a.shape[1]) + x_a.transpose() @ x_a
                    self.train_inv_Lambda = np.linalg.inv(self.train_inv_Lambda)
                a_uncertainty = np.einsum('ij,jk,ik->i', x_a, self.train_inv_Lambda, x_a).reshape(-1, 1)
                uncertainty[a_indices, :] = np.sqrt(a_uncertainty)
        return uncertainty

    def initialize(self, state, action, target):
        self.worst_value = np.min(target)
        if self.model_type == "linear" and self.args['trans_type'] != "none":
            x = self.feature_trans.fit_transform(state)
            error = self.linear_fit(x, action, target)
        else:
            x = np.hstack((state, action))
            self.q_model.fit(x, target)
            error = target - self.q_model.predict(x).reshape(-1, 1)
        self.est_std = np.linalg.norm(error) / np.sqrt(error.size)
        self.q_model_list.append(self.q_model)

    def step(self, state, action, target):
        if self.model_type == "linear":
            self.q_model = {k: clone(v) for k, v in self.q_model.items()}
            x = self.feature_trans.transform(state)
            error = self.linear_fit(x, action, target)
        else:
            self.q_model = clone(self.q_model)
            x = np.hstack((state, action))
            self.q_model.fit(x, target)
            error = target - self.q_model.predict(x).reshape(-1, 1)
        self.est_std = np.linalg.norm(error) / np.sqrt(error.size)

        self.q_model_list.append(self.q_model)
        self.iteration_time += 1
        pred_new = self.linear_predict(self.q_model_list[self.iteration_time], x, action)
        pred_old = self.linear_predict(self.q_model_list[self.iteration_time-1], x, action)
        pred_diff = np.abs(pred_new - pred_old).mean() / (np.abs(pred_new).mean()+1e-6)
        self.pred_diff.append(pred_diff)
        return 9999.0, pred_diff

    def Q_value(self, state, action):
        if self.model_type == "linear":
            x = self.feature_trans.transform(state)
            pred = self.linear_predict(self.q_model_list[self.iteration_time], x, action)
            if self.pessimism_scale > 0.0:
                uncertainty_pred = self.linear_uncertainty(self.q_model_list[self.iteration_time], x, action)
                pred = pred - self.pessimism_scale * uncertainty_pred
        else:
            x = np.hstack([state, action])
            pred = self.q_model.predict(x).reshape(-1, 1)

        return pred

class Q_function:
    def __init__(self, A_set, time_difference=None, args=None):
        if time_difference is None:
            self.time_difference = np.ones(time_difference)  
        else:
            self.time_difference = np.copy(time_difference)

        self.model_type = args['model_type']
        if self.model_type == "linear":
            self.A_set = A_set
            if self.A_set.ndim == 1:
                tmp_unique_action = self.A_set.reshape(-1, 1)
            self.ohe_action_feature = OneHotEncoder(drop='first', sparse=False).fit(tmp_unique_action)
            self.action_dummy_size = np.size(self.ohe_action_feature.categories_[0]) - 1

            if args['trans_type'] == "rbf":
                self.feature_trans = RBFSampler(random_state=1, n_components=args['rbf_feature_num'])
            elif args['trans_type'] == "poly":
                self.feature_trans = PolynomialFeatures(degree=args['poly_degree'], interaction_only=True, include_bias=True)
            elif args['trans_type'] == "none":
                self.feature_trans = None
            self.q_model = Ridge(solver='lsqr', alpha=1024.0, random_state=1, fit_intercept=False)
        elif self.model_type == "forest":
            self.q_model = RandomForestRegressor(random_state=1, min_samples_leaf=6)
        else:
            pass

        self.iteration_time = 0
        self.q_model_list = []
        self.coef_diff = []
        self.pred_diff = []
        self.args = args
        pass

    def initialize(self, state, action, target):
        x_action = self.ohe_action_feature.transform(action)
        init_x = np.hstack((state, x_action))
        if self.model_type == "linear" and self.args['trans_type'] != "none":
            init_x = self.feature_trans.fit_transform(init_x)
        self.q_model.fit(init_x, target)
        self.q_model_list.append(self.q_model)
        self.est_std = np.linalg.norm(target - self.q_model.predict(init_x)) / np.sqrt(init_x.shape[0])

    def step(self, state, action, target):
        if len(target.shape) == 2:
            target = target.flatten()
        if self.model_type == "linear":
            X_action = self.ohe_action_feature.transform(action)
            if X_action.ndim == 1:
                X_action = X_action.reshape(-1, self.action_dummy_size)

            X_state = np.copy(state)
            train_X = np.hstack((X_state, X_action))
            if self.model_type == "linear" and self.feature_trans is not None:
                train_X = self.feature_trans.transform(train_X)
        else:
            train_X = np.hstack((state, action))

        self.q_model = clone(self.q_model)
        self.q_model.fit(train_X, target)
        self.iteration_time += 1
        self.q_model_list.append(self.q_model)
        coef_new = self.q_model_list[self.iteration_time].coef_
        coef_old = self.q_model_list[self.iteration_time-1].coef_
        coef_diff = np.abs(coef_new - coef_old).mean() / (np.abs(coef_new).mean()+1e-6)
        self.coef_diff.append(coef_diff)
        pred_new = self.q_model_list[self.iteration_time].predict(train_X).reshape(-1, 1)
        pred_old = self.q_model_list[self.iteration_time-1].predict(train_X).reshape(-1, 1)
        pred_diff = np.abs(pred_new - pred_old).mean() / (np.abs(pred_new).mean()+1e-6)
        self.pred_diff.append(pred_diff)
        self.est_std = np.linalg.norm(target - pred_new) / np.sqrt(train_X.shape[0])
        return coef_diff, pred_diff

    def Q_value(self, state, action):
        if self.model_type == "linear":
            x_action = self.ohe_action_feature.transform(action)

            ## concat and transform
            x_state = np.copy(state)
            x = np.hstack((x_state, x_action))
            if self.feature_trans is not None:
                x = self.feature_trans.transform(x)
        else:
            x = np.hstack([state, action])

        pred = self.q_model.predict(x).reshape(-1, 1)
        return pred