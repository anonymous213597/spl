import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold

import statsmodels.api as sm
from sklearn.feature_selection import VarianceThreshold
try:
    from ppi_py.cross_ppi import crossppi_ols_pred_ci
except ImportError:
    raise ImportError("Please install ppi_py package")
from sklearn.neural_network import MLPRegressor


class RewardLB:
    def __init__(self) -> None:
        pass

    def compute_mean_lb(self, observation, action):
        reward_mean = self.compute_mean(observation, action)
        reward_std = self.compute_se(observation, action)
        reward_mean_lb = reward_mean - self.pessimism_scale * reward_std
        return reward_mean_lb

    def compute_lb_parametric(self, observation, action, reward):
        reward_std = self.compute_se(observation, action)
        reward_mean_lb = reward - self.pessimism_scale * reward_std
        return reward_mean_lb

    def estimate_lb_parametric(self, observation, action):
        reward_std = self.compute_se(observation, action)
        reward_mean_est = self.compute_mean(observation, action)
        reward_mean_lb = reward_mean_est - self.pessimism_scale * reward_std
        return reward_mean_lb.reshape(-1, 1)

    def compute_std(self, x, reward):
        error_std = np.sum((x @ self.model.cov_HC0) * x, axis=1).reshape(-1, 1)
        return error_std

    def sieve_refit(self, observation, action, reward, transform="poly"):
        self.REFIT_RIDGE_PENALTY = 16.0
        # training data:
        if transform == "rbf":
            self.feature_trans = RBFSampler(
                random_state=1, n_components=self.train_args["rbf_feature_num"]
            )
        elif transform == "poly":
            self.feature_trans = PolynomialFeatures(
                degree=self.train_args["poly_degree"],
                interaction_only=True,
                include_bias=True,
            )
        if np.unique(action).shape[0] == 1:
            self.ohe_action_feature = OneHotEncoder(sparse=False).fit(action)
            self.action_dummy_size = np.size(self.ohe_action_feature.categories_[0])
        else:
            try:
                self.ohe_action_feature = OneHotEncoder(
                    drop="first", sparse=False, categories="auto"
                ).fit(action)
            except TypeError:
                self.ohe_action_feature = OneHotEncoder(
                    drop="first", sparse_output=False, categories="auto"
                ).fit(action)
            self.action_dummy_size = np.size(self.ohe_action_feature.categories_[0]) - 1
        x_action = self.ohe_action_feature.transform(action)
        x = np.hstack([observation, x_action])
        x = self.feature_trans.fit_transform(x)
        self.const_excluder = VarianceThreshold()
        x = self.const_excluder.fit_transform(x)
        self.model = sm.OLS(reward, x).fit()
        self.pred_std = self.compute_std(x, reward)

    def compute_refit_lb(self, observation, action, scale=2.0):
        x_action = self.ohe_action_feature.transform(action)
        x = np.hstack([observation, x_action])
        x = self.feature_trans.transform(x)
        x = self.const_excluder.transform(x)
        fitted_mean = self.model.predict(x).reshape(-1, 1)
        if scale is not None:
            fitted_lb = fitted_mean - scale * self.pred_std
        else:
            fitted_lb = fitted_mean - self.pessimism_scale * self.pred_std
        return fitted_lb


class OracleRewardLB(RewardLB):
    def __init__(self, simulator, pessimism_scale=1.0) -> None:
        super().__init__()
        self.pessimism_scale = pessimism_scale
        self.simulator = simulator

    def fit(self, observation, action, reward):
        pass

    def compute_mean(self, observation, action):
        reward_mean = self.simulator.sa2reward_model(observation, action, random=False)
        return reward_mean

    def compute_se(self, observation, action):
        num = observation.shape[0]
        reward_mean = self.simulator.sa2reward_model(observation, action, random=False)
        reward_std = self.simulator.save_reward_std(reward_mean)
        reward_std /= np.sqrt(num)
        return reward_std


class MLRewardLB(RewardLB):
    def __init__(self, train_args, pessimism_scale=1.0, l2_penalty=8.0) -> None:
        super().__init__()
        self.pessimism_scale = pessimism_scale
        self.train_args = train_args
        self.RIDGE_PENALTY = l2_penalty

        self.model_type = train_args["model_type"]
        self.auto_tune = False
        if self.model_type == "linear":
            if train_args["trans_type"] == "rbf":
                self.feature_trans = RBFSampler(
                    random_state=1, n_components=train_args["rbf_feature_num"]
                )
            elif train_args["trans_type"] == "poly":
                self.feature_trans = PolynomialFeatures(
                    degree=train_args["poly_degree"],
                    interaction_only=True,
                    include_bias=True,
                )
            self.model = Ridge(
                solver="lsqr",
                alpha=self.RIDGE_PENALTY,
                random_state=1,
                fit_intercept=False,
            )
        elif self.model_type == "forest":
            self.model = RandomForestRegressor(
                random_state=1,
                min_samples_leaf=train_args["min_samples_leaf"],
                oob_score=True,
                n_estimators=100,
                max_features="sqrt",
            )
        elif self.model_type == "gbt":
            self.model = GradientBoostingRegressor(
                random_state=1, subsample=0.7, n_estimators=train_args["n_estimators"]
            )
        elif self.model_type == "network":
            self.model = MLPRegressor(
                random_state=1, hidden_layer_sizes=train_args["hidden_layer_sizes"]
            )
        else:
            pass
        self.const_excluder = VarianceThreshold()

    def _auto_tune_forest_params(self, X, y, original_min_samples_leaf):
        n_samples = len(y)
        
        candidates = self._generate_candidates(n_samples, original_min_samples_leaf)
        results = []
        best_score = float('-inf')
        best_min_samples_leaf = original_min_samples_leaf
        
        cv_folds = min(5, n_samples // 10)
        if cv_folds < 2:
            return original_min_samples_leaf, None
            
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for i, candidate in enumerate(candidates):
            rf = RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=candidate,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )
            
            scores = cross_val_score(rf, X, y, cv=kf, 
                                    scoring='neg_mean_squared_error')
            mean_score = scores.mean()
            std_score = scores.std()
            
            results.append({
                'min_samples_leaf': candidate,
                'mean_score': mean_score,
                'std_score': std_score
            })
            
            if mean_score > best_score:
                best_score = mean_score
                best_min_samples_leaf = candidate
        
        self.tuning_results = results
        return best_min_samples_leaf, results

    def _generate_candidates(self, n_samples, original_value):
        candidates = []
        min_val = max(2, int(0.001 * n_samples))
        max_val = min(int(0.2 * n_samples), 500)
        current = min_val
        while current <= max_val:
            candidates.append(current)
            if current < 10:
                current += 2
            elif current < 50:
                current += 5
            elif current < 100:
                current += 10
            else:
                current = int(current * 1.3)
        if original_value not in candidates and min_val <= original_value <= max_val:
            candidates.append(original_value)
        for factor in [0.5, 0.75, 1.25, 1.5, 2.0]:
            val = int(original_value * factor)
            if min_val <= val <= max_val and val not in candidates:
                candidates.append(val)
        
        return sorted(list(set(candidates)))


    def fit(self, observation, action, reward):
        if self.model_type == "linear":
            if np.unique(action).shape[0] == 1:
                try:
                    self.ohe_action_feature = OneHotEncoder(sparse=False).fit(action)
                except:
                    self.ohe_action_feature = OneHotEncoder(sparse_output=False).fit(
                        action
                    )
                self.action_dummy_size = np.size(self.ohe_action_feature.categories_[0])
            else:
                try:
                    self.ohe_action_feature = OneHotEncoder(
                        drop="first", sparse=False, categories="auto"
                    ).fit(action)
                except TypeError:
                    self.ohe_action_feature = OneHotEncoder(
                        drop="first", sparse_output=False, categories="auto"
                    ).fit(action)
                self.action_dummy_size = (
                    np.size(self.ohe_action_feature.categories_[0]) - 1
                )
            x_action = self.ohe_action_feature.transform(action)
            x = np.hstack([observation, x_action])
            x = self.feature_trans.fit_transform(x)
            x = self.const_excluder.fit_transform(x)
            self.model = sm.OLS(reward, x).fit()
        else:
            x = np.hstack([observation, action])
            original_min_samples_leaf = self.train_args["min_samples_leaf"]
            if self.auto_tune:
                best_min_samples_leaf, _ = self._auto_tune_forest_params(x, reward.ravel(), original_min_samples_leaf)
            else:
                best_min_samples_leaf = original_min_samples_leaf
            self.model = RandomForestRegressor(
                random_state=1,
                min_samples_leaf=best_min_samples_leaf,
                oob_score=True,
                n_estimators=100,
                max_features="sqrt",
            )
            self.train_args["min_samples_leaf"] = best_min_samples_leaf
            self.model.fit(x, reward.ravel())
            self.x_train = np.copy(x)
            self.error_var = np.mean(
                np.square(reward - self.model.predict(x).reshape(-1, 1))
            )

    def cross_fit(self, observation, action, reward, u_obs, u_act, K):
        if self.model_type == "linear":
            pass
        else:
            l_x = np.hstack([observation, action])
            l_y = np.copy(reward.ravel())
            u_x = np.hstack([u_obs, u_act])
            l_y_hat = np.zeros(l_y.shape)
            u_y_hat = np.zeros((u_x.shape[0], K))

            kf = KFold(n_splits=K)
            for i, (train_index, test_index) in enumerate(kf.split(l_x)):
                X_train, X_test = l_x[train_index], l_x[test_index]
                y_train, y_test = l_y[train_index], l_y[test_index]
                ml = RandomForestRegressor(
                    random_state=i,
                    max_features="sqrt",
                    min_samples_leaf=self.train_args["min_samples_leaf"],
                ).fit(X_train, y_train)
                l_y_hat[test_index] += ml.predict(X_test)
                u_y_hat[:, i] = ml.predict(u_x)
        return l_y_hat, u_y_hat

    def compute_mean(self, observation, action):
        if self.model_type == "linear":
            x_action = self.ohe_action_feature.transform(action)
            x = np.hstack([observation, x_action])
            x = self.feature_trans.transform(x)
            x = self.const_excluder.transform(x)
        else:
            x = np.hstack([observation, action])
        reward_mean = self.model.predict(x)
        reward_mean = reward_mean.reshape(-1, 1)
        return reward_mean

    def compute_se(self, observation, action):
        if self.pessimism_scale != 0.0:
            if self.model_type == "linear":
                x_action = self.ohe_action_feature.transform(action)
                x = np.hstack([observation, x_action])
                x = self.feature_trans.transform(x)
                x = self.const_excluder.transform(x)
                pred_std = np.sum((x @ self.model.cov_HC0) * x, axis=1).reshape(-1, 1)
            else:
                pred_std = np.repeat(self.pred_std, action.size).reshape(-1, 1)
        else:
            pred_std = np.zeros((action.size, 1))
        return pred_std

    def compute_ppi_se2(
        self,
        l_obs,
        l_act,
        l_reward,
        u_obs,
        u_act,
        train_args={"trans_type": "poly", "poly_degree": 2},
    ):
        if self.model_type == "linear":
            raise ValueError
        u_y = self.model.predict(np.hstack([u_obs, u_act])).reshape(-1, 1)
        l_resi = l_reward - self.model.predict(np.hstack([l_obs, l_act])).reshape(-1, 1)
        ## linear model:
        ### pre-processing:
        if np.unique(u_act).shape[0] == 1:
            try:
                ohe_action_feature = OneHotEncoder(sparse=False).fit(u_act)
            except TypeError:
                ohe_action_feature = OneHotEncoder(sparse_output=False).fit(u_act)
        else:
            try:
                ohe_action_feature = OneHotEncoder(
                    drop="first", sparse=False, categories="auto"
                ).fit(u_act)
            except TypeError:
                ohe_action_feature = OneHotEncoder(
                    drop="first", sparse_output=False, categories="auto"
                ).fit(u_act)
        l_x_action = ohe_action_feature.transform(l_act)
        u_x_action = ohe_action_feature.transform(u_act)
        if train_args["trans_type"] == "rbf":
            feature_trans = RBFSampler(
                random_state=1, n_components=train_args["rbf_feature_num"]
            )
        elif train_args["trans_type"] == "poly":
            feature_trans = PolynomialFeatures(
                degree=train_args["poly_degree"],
                interaction_only=True,
                include_bias=True,
            )
        const_excluder = VarianceThreshold()
        l_x = np.hstack([l_obs, l_x_action])
        l_x = feature_trans.fit_transform(l_x)
        l_x = const_excluder.fit_transform(l_x)
        u_x = np.hstack([u_obs, u_x_action])
        u_x = feature_trans.transform(u_x)
        u_x = const_excluder.transform(u_x)
        N, p = u_x.shape
        M = l_x.shape[0]
        l_model = sm.OLS(l_resi, l_x).fit()
        u_model = sm.OLS(u_y, u_x).fit()
        l_var = np.sum((u_x @ l_model.cov_HC0) * u_x, axis=1)
        u_var = np.sum((u_x @ u_model.cov_HC0) * u_x, axis=1)
        c_std = u_var / N + l_var / M

        self.ppi_ohe_action_feature = ohe_action_feature
        self.ppi_feature_trans = feature_trans
        self.ppi_const_excluder = const_excluder
        self.ppi_l_model = l_model
        self.ppi_u_model = u_model
        self.l_num = M
        self.u_num = N
        return c_std

    def predict_cross_ppi_lb(
        self,
        l_obs,
        l_act,
        l_reward,
        l_y,
        u_obs,
        u_act,
        u_y,
        train_args={"trans_type": "rbf", "rbf_feature_num": 30},
    ):
        if self.model_type == "linear":
            raise ValueError
        if np.unique(u_act).shape[0] == 1:
            ohe_action_feature = OneHotEncoder(sparse=False).fit(u_act)
        else:
            try:
                ohe_action_feature = OneHotEncoder(
                    drop="first", sparse=False, categories="auto"
                ).fit(u_act)
            except TypeError:
                ohe_action_feature = OneHotEncoder(
                    drop="first", sparse_output=False, categories="auto"
                ).fit(u_act)
        l_x_action = ohe_action_feature.transform(l_act)
        u_x_action = ohe_action_feature.transform(u_act)
        if train_args["trans_type"] == "rbf":
            feature_trans = RBFSampler(
                random_state=1, n_components=train_args["rbf_feature_num"]
            )
        elif train_args["trans_type"] == "poly":
            feature_trans = PolynomialFeatures(
                degree=train_args["poly_degree"],
                interaction_only=True,
                include_bias=True,
            )
        const_excluder = VarianceThreshold()
        l_x = np.hstack([l_obs, l_x_action])
        l_x = feature_trans.fit_transform(l_x)
        l_x = const_excluder.fit_transform(l_x)
        u_x = np.hstack([u_obs, u_x_action])
        u_x = feature_trans.transform(u_x)
        u_x = const_excluder.transform(u_x)
        l_y_lb = crossppi_ols_pred_ci(
            l_x, l_reward.flatten(), l_y, u_x, u_y, l_x, alpha=0.05
        )[0]
        u_y_lb = crossppi_ols_pred_ci(
            l_x, l_reward.flatten(), l_y, u_x, u_y, u_x, alpha=0.05
        )[0]
        return l_y_lb.reshape(-1, 1), u_y_lb.reshape(-1, 1)

    def predict_ppi_se(self, state, action):
        x_action = self.ppi_ohe_action_feature.transform(action)
        x = np.hstack([state, x_action])
        x = self.ppi_feature_trans.transform(x)
        x = self.ppi_const_excluder.transform(x)
        l_var = np.sum((x @ self.ppi_l_model.cov_HC0) * x, axis=1)
        u_var = np.sum((x @ self.ppi_u_model.cov_HC0) * x, axis=1)
        ppi_std = u_var / self.u_num + l_var / self.l_num
        ppi_std = ppi_std.reshape(-1, 1)
        return ppi_std

    def predict_ppi_mean(self, state, action):
        x_action = self.ppi_ohe_action_feature.transform(action)
        x = np.hstack([state, x_action])
        x = self.ppi_feature_trans.transform(x)
        x = self.ppi_const_excluder.transform(x)
        u_predict = self.ppi_u_model.predict(x)
        l_predict = self.ppi_l_model.predict(x)
        ppi_mean = u_predict + l_predict
        ppi_mean = ppi_mean.reshape(-1, 1)
        return ppi_mean

    def compute_ppi_se(self, l_obs, l_act, l_reward, u_obs, u_act):
        RIDGE_PENALTY = 16.0
        if self.model_type == "linear":
            raise ValueError
        u_y = self.model.predict(np.hstack([u_obs, u_act])).reshape(-1, 1)
        l_resi = l_reward - self.model.predict(np.hstack([l_obs, l_act])).reshape(-1, 1)
        ## linear model:
        ### pre-processing:
        if np.unique(u_act).shape[0] == 1:
            ohe_action_feature = OneHotEncoder(sparse=False).fit(u_act)
        else:
            try:
                ohe_action_feature = OneHotEncoder(
                    drop="first", sparse=False, categories="auto"
                ).fit(u_act)
            except TypeError:
                ohe_action_feature = OneHotEncoder(
                    drop="first", sparse_output=False, categories="auto"
                ).fit(u_act)
        l_x_action = ohe_action_feature.transform(l_act)
        u_x_action = ohe_action_feature.transform(u_act)
        feature_trans = PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)
        l_x = np.hstack([l_obs, l_x_action])
        l_x = feature_trans.fit_transform(l_x)
        u_x = np.hstack([u_obs, u_x_action])
        u_x = feature_trans.fit_transform(u_x)
        N, p = u_x.shape
        M = l_x.shape[0]
        l_model = Ridge(solver="lsqr", alpha=RIDGE_PENALTY, random_state=1, fit_intercept=False)
        u_model = Ridge(solver="lsqr", alpha=RIDGE_PENALTY, random_state=2, fit_intercept=False)
        l_model.fit(l_x, l_resi)
        u_model.fit(u_x, u_y)
        l_indiv_resi_var = np.square(l_resi - l_model.predict(l_x).reshape(-1, 1))
        l_inverse_design_mat = np.linalg.inv(
            l_x.transpose() @ l_x + RIDGE_PENALTY * np.eye(p)
        )
        temp_l = l_x.T @ (l_indiv_resi_var * l_x)
        Q_l = l_inverse_design_mat @ temp_l @ l_inverse_design_mat
        l_var = (M / (M - p)) * np.sum(u_x @ Q_l * u_x, axis=1, keepdims=True)
        u_indiv_resi_var = np.square(u_y - l_model.predict(u_x).reshape(-1, 1))
        u_inverse_design_mat = np.linalg.inv(
            u_x.transpose() @ u_x + RIDGE_PENALTY * np.eye(p)
        )
        temp_u = u_x.T @ (u_indiv_resi_var * u_x)
        Q_u = u_inverse_design_mat @ temp_u @ u_inverse_design_mat
        u_var = (N / (N - p)) * np.sum(u_x @ Q_u * u_x, axis=1, keepdims=True)
        c_var = u_var / N + l_var / M
        self.ppi_l_model = l_model
        self.ppi_u_model = u_model
        self.ppi_feature_trans = feature_trans
        self.ppi_ohe_action_feature = ohe_action_feature

        return c_var

    def compute_ppi_mean(self, new_obs, new_act):
        new_act_feat = self.ppi_ohe_action_feature.transform(new_act)
        new_x = self.ppi_feature_trans.transform(np.hstack([new_obs, new_act_feat]))
        new_y = self.ppi_l_model.predict(new_x) + self.ppi_u_model.predict(new_x)
        return new_y


class PDSRewardLB:
    def __init__(self, train_args, pessimism_scale=2.0, l2_penalty=8.0) -> None:
        super().__init__()
        self.pessimism_scale = pessimism_scale
        self.train_args = train_args
        self.RIDGE_PENALTY = l2_penalty

        self.model_type = train_args["model_type"]
        if self.model_type == "linear":
            if train_args["trans_type"] == "rbf":
                self.feature_trans = RBFSampler(
                    random_state=1, n_components=train_args["rbf_feature_num"]
                )
            elif train_args["trans_type"] == "poly":
                self.feature_trans = PolynomialFeatures(
                    degree=train_args["poly_degree"],
                    interaction_only=True,
                    include_bias=True,
                )
            self.model = Ridge(
                solver="lsqr",
                alpha=self.RIDGE_PENALTY,
                random_state=1,
                fit_intercept=False,
            )
        else:
            pass
        self.const_excluder = VarianceThreshold()

    def fit(self, observation, action, reward):
        if np.unique(action).shape[0] == 1:
            self.ohe_action_feature = OneHotEncoder(sparse=False).fit(action)
            self.action_dummy_size = np.size(self.ohe_action_feature.categories_[0])
        else:
            try:
                self.ohe_action_feature = OneHotEncoder(
                    drop="first", sparse=False, categories="auto"
                ).fit(action)
            except TypeError:
                self.ohe_action_feature = OneHotEncoder(
                    drop="first", sparse_output=False, categories="auto"
                ).fit(action)
            self.action_dummy_size = np.size(self.ohe_action_feature.categories_[0]) - 1
        x_action = self.ohe_action_feature.transform(action)
        x = np.hstack([observation, x_action])
        x = self.feature_trans.fit_transform(x)
        x = self.const_excluder.fit_transform(x)
        self.model = sm.OLS(reward, x).fit()

    def compute_lb(self, observation, action, pessimism_scale=2.0):
        x_action = self.ohe_action_feature.transform(action)
        x = np.hstack([observation, x_action])
        x = self.feature_trans.transform(x)
        x = self.const_excluder.transform(x)
        reward_mean = self.model.predict(x).reshape(-1, 1)
        pred_std = np.sum((x @ self.model.cov_HC0) * x, axis=1).reshape(-1, 1)
        reward_lb_value = reward_mean - pessimism_scale * pred_std
        return reward_lb_value
