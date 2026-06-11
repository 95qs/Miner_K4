"""
检测器模块：GMM / KDE / OC-SVM / DeepSVDD

参考 K4 论文，支持四种检测器基于 PRDC 描述子做异常评分。

此模块独立于原始 K4 代码库，由 K4-service 自行维护。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from typing import Optional, Literal, Any


class BaseDetector:
    """检测器基类"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, prdc_features: np.ndarray):
        raise NotImplementedError

    def score(self, prdc_features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, prdc_features: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        raise NotImplementedError

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        return self.scaler.transform(X)


class GMMDetector(BaseDetector):
    """高斯混合模型检测器"""

    def __init__(self, n_components: int = 3, covariance_type: str = "full", random_state: int = 42):
        super().__init__()
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )

    def fit(self, prdc_features: np.ndarray):
        X_scaled = self.scaler.fit_transform(prdc_features)
        self.model.fit(X_scaled)
        self.fitted = True

    def score(self, prdc_features: np.ndarray) -> np.ndarray:
        X_scaled = self._scale(prdc_features)
        return -self.model.score_samples(X_scaled)


class KDETector(BaseDetector):
    """核密度估计检测器"""

    def __init__(self, bandwidth: float = 0.5, kernel: str = "gaussian"):
        super().__init__()
        from scipy.stats import gaussian_kde
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde_model = None

    def fit(self, prdc_features: np.ndarray):
        from scipy.stats import gaussian_kde
        X_scaled = self.scaler.fit_transform(prdc_features)
        try:
            self.kde_model = gaussian_kde(X_scaled.T, bw_method="scott")
        except Exception:
            self.kde_model = gaussian_kde(X_scaled.T, bw_method=self.bandwidth)
        self.fitted = True

    def score(self, prdc_features: np.ndarray) -> np.ndarray:
        X_scaled = self._scale(prdc_features)
        density = self.kde_model(X_scaled.T)
        return -np.log(density + 1e-10)


class OCSVMDetector(BaseDetector):
    """单类 SVM 检测器"""

    def __init__(self, nu: float = 0.1, kernel: str = "rbf", gamma: str = "scale"):
        super().__init__()
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def fit(self, prdc_features: np.ndarray):
        X_scaled = self.scaler.fit_transform(prdc_features)
        self.model.fit(X_scaled)
        self.fitted = True

    def score(self, prdc_features: np.ndarray) -> np.ndarray:
        X_scaled = self._scale(prdc_features)
        return -self.model.decision_function(X_scaled)


class DeepSVDDDetector(BaseDetector):
    """
    Deep SVDD 检测器（K4 论文中的 adapted 版本）

    训练一个自编码器，最小化所有样本到中心的距离。
    推理时，距离中心越远越异常。
    """

    def __init__(self, hidden_dim: int = 32, epochs: int = 50, lr: float = 1e-3, device: str = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = None
        self.center = None

    def _build_encoder(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def fit(self, prdc_features: np.ndarray):
        X_scaled = self.scaler.fit_transform(prdc_features)
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)

        input_dim = X_tensor.shape[1]
        self.encoder = self._build_encoder(input_dim).to(self.device)
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)

        self.encoder.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            representations = self.encoder(X_tensor)
            if self.center is None:
                self.center = representations.mean(dim=0, keepdim=True)
            loss = ((representations - self.center) ** 2).mean()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 20 == 0:
                self.center = self.encoder(X_tensor).mean(dim=0, keepdim=True).detach()

        self.center = self.center.detach()
        self.fitted = True

    def score(self, prdc_features: np.ndarray) -> np.ndarray:
        X_scaled = self._scale(prdc_features)
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)
        self.encoder.eval()
        with torch.no_grad():
            representations = self.encoder(X_tensor)
        dists = torch.norm(representations - self.center, dim=1)
        return dists.cpu().numpy()


class IsolationForestDetector(BaseDetector):
    """
    Isolation Forest 检测器。

    不依赖数据的概率分布假设，通过随机切分来隔离异常点。
    异常点路径短（fewer splits needed to isolate），因此 anomaly score 高。

    分数含义：值越高越异常（与 GMM 的 -log-likelihood 方向一致）。
    """

    def __init__(self, n_estimators: int = 100, contamination: float = 0.1, random_state: int = 42):
        super().__init__()
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = None

    def fit(self, prdc_features: np.ndarray):
        from sklearn.ensemble import IsolationForest
        X_scaled = self.scaler.fit_transform(prdc_features)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.fitted = True

    def score(self, prdc_features: np.ndarray) -> np.ndarray:
        X_scaled = self._scale(prdc_features)
        # decision_function: 值越高越正常 → 取负使其越高越异常
        raw = self.model.decision_function(X_scaled)
        return -raw


class DetectorFactory:
    """检测器工厂"""

    _DETECTORS: dict[str, type[BaseDetector]] = {
        "gmm": GMMDetector,
        "kde": KDETector,
        "ocsvm": OCSVMDetector,
        "deepsvd": DeepSVDDDetector,
        "iforest": IsolationForestDetector,
    }

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseDetector:
        if name not in cls._DETECTORS:
            raise ValueError(
                f"Unknown detector: {name}. Available: {list(cls._DETECTORS.keys())}"
            )
        return cls._DETECTORS[name](**kwargs)

    @classmethod
    def list_detectors(cls) -> list[str]:
        return list(cls._DETECTORS.keys())
