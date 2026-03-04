import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from typing import Dict


class SPLRewardModel:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_rff: int = 1000,
        rff_sigma: float = 1.0,
        alpha: float = 0.1,
        rf_n_estimators: int = 100,
        seed: int = 0
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_rff = n_rff
        self.rff_sigma = rff_sigma
        self.alpha = alpha
        self.rf_n_estimators = rf_n_estimators
        self.seed = seed
        d = obs_dim + action_dim
        rng = np.random.RandomState(seed)
        self.W = rng.normal(0, 1.0 / rff_sigma, size=(d, n_rff))  # (d, n_rff)
        self.b = rng.uniform(0, 2 * np.pi, size=(n_rff,))          # (n_rff,)
        self.rf = None
        self.theta_SUQ = None
        self.Sigma_L = None
        self.Sigma_U = None
        self.n_L = None
        self.n_U = None

    def _compute_rff_features(self, sa: np.ndarray) -> np.ndarray:
        z = sa @ self.W + self.b          # (N, n_rff)
        scale = np.sqrt(2.0 / self.n_rff)
        return scale * np.concatenate([np.cos(z), np.sin(z)], axis=1)  # (N, 2*n_rff)

    def _estimate_density_ratio(self, sa_L: np.ndarray, sa_U: np.ndarray) -> np.ndarray:
        n_L = len(sa_L)
        n_U = len(sa_U)
        X = np.concatenate([sa_L, sa_U], axis=0)
        y = np.concatenate([np.zeros(n_L), np.ones(n_U)])
        clf = LogisticRegression(max_iter=1000, random_state=self.seed)
        clf.fit(X, y)
        proba_U = clf.predict_proba(sa_U)
        p0 = proba_U[:, 0]
        p1 = proba_U[:, 1]
        w = (p0 / (p1 + 1e-12)) * (n_U / n_L)
        w = np.clip(w, 1e-3, 1e3)
        return w

    def _sandwich_ols(self, X: np.ndarray, y: np.ndarray):
        lam = 1e-6
        D = X.shape[1]
        XtX = X.T @ X + lam * np.eye(D)
        XtX_inv = np.linalg.inv(XtX)
        beta = XtX_inv @ (X.T @ y)                
        e = y - X @ beta                          
        meat = X.T @ ((e ** 2)[:, None] * X)      
        Sigma = XtX_inv @ meat @ XtX_inv
        return beta, Sigma

    def _sandwich_wls(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        lam = 1e-6
        D = X.shape[1]
        XtWX = X.T @ (w[:, None] * X) + lam * np.eye(D)
        XtWX_inv = np.linalg.inv(XtWX)
        beta = XtWX_inv @ (X.T @ (w * y))                  
        e = y - X @ beta                                   
        meat = X.T @ ((w ** 2 * e ** 2)[:, None] * X)      
        Sigma = XtWX_inv @ meat @ XtWX_inv
        return beta, Sigma

    def fit(self, labeled_dataset: Dict, unlabeled_dataset: Dict) -> None:
        obs_L = labeled_dataset["observations"].astype(np.float32)
        act_L = labeled_dataset["actions"].astype(np.float32)
        rew_L = labeled_dataset["rewards"].astype(np.float32).flatten()
        obs_U = unlabeled_dataset["observations"].astype(np.float32)
        act_U = unlabeled_dataset["actions"].astype(np.float32)
        sa_L = np.concatenate([obs_L, act_L], axis=1)  # (n_L, d)
        sa_U = np.concatenate([obs_U, act_U], axis=1)  # (n_U, d)
        n_L = len(sa_L)
        n_U = len(sa_U)
        self.rf = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            random_state=self.seed,
            n_jobs=-1
        )
        self.rf.fit(sa_L, rew_L)
        g_L = self._compute_rff_features(sa_L)
        residuals_L = rew_L - self.rf.predict(sa_L)
        theta_L, Sigma_L = self._sandwich_ols(g_L, residuals_L)
        w = self._estimate_density_ratio(sa_L, sa_U)
        g_U = self._compute_rff_features(sa_U)
        rf_pred_U = self.rf.predict(sa_U)
        theta_U, Sigma_U = self._sandwich_wls(g_U, rf_pred_U, w)
        self.theta_SUQ = theta_L + theta_U
        self.Sigma_L = Sigma_L
        self.Sigma_U = Sigma_U
        self.n_L = n_L
        self.n_U = n_U

    def predict(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        sa = np.concatenate([obs, action], axis=1).astype(np.float32)  # (N, d)
        N = len(sa)
        g = self._compute_rff_features(sa)              
        R_SUQ = g @ self.theta_SUQ                      
        chunk = 10000
        var_L = np.empty(N, dtype=np.float64)
        var_U = np.empty(N, dtype=np.float64)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            gc = g[start:end]                           
            var_L[start:end] = np.einsum('nd,dd,nd->n', gc, self.Sigma_L, gc)
            var_U[start:end] = np.einsum('nd,dd,nd->n', gc, self.Sigma_U, gc)
        delta = np.sqrt(var_L / self.n_L + var_U / self.n_U)
        z = norm.ppf(1.0 - self.alpha)
        R_SPL = R_SUQ - z * delta
        return R_SPL.reshape(-1, 1).astype(np.float32)
