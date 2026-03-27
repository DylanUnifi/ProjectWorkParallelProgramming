import os
import joblib
import numpy as np
from typing import Optional, Union

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score
)
from sklearn.base import BaseEstimator, ClassifierMixin
import torch

class EnhancedSVM(BaseEstimator, ClassifierMixin):
    """
    SVM wrapper with:
      - support for the ``precomputed`` kernel
      - configurable ``class_weight`` (defaults to ``balanced``)
      - an optional decision threshold for binary classification
      - save/load helpers and Torch interop
      - exposed ``max_iter``, ``tol``, and ``cache_size`` safety controls
    """
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "precomputed",
        gamma: Union[str, float] = "scale",
        use_pca: bool = False,
        pca_model: Optional[object] = None,
        save_dir: Optional[str] = None,
        probability: bool = True,
        class_weight: Optional[Union[dict, str]] = "balanced",
        decision_threshold: Optional[float] = None,
        random_state: Optional[int] = 42,
        max_iter: int = 10000,
        tol: float = 1e-3,
        cache_size: float = 1000,
        verbose: bool = False
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.use_pca = use_pca
        self.pca_model = pca_model
        self.save_dir = save_dir or "./checkpoints_svm"
        self.probability = probability
        self.class_weight = class_weight
        self.decision_threshold = decision_threshold
        self.random_state = random_state
        
        # Solver parameters
        self.max_iter = max_iter
        self.tol = tol
        self.cache_size = cache_size
        self.verbose = verbose

        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=self.probability,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
            cache_size=self.cache_size,
            verbose=self.verbose
        )

    def fit(self, X, y):
        if self.kernel == "precomputed":
            if not np.all(np.isfinite(X)):
                if self.verbose:
                    print("Warning: EnhancedSVM received NaN/Inf values in the kernel matrix. Cleaning input.")
                X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
                
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Use ``SVC.predict()`` by default.
        If ``decision_threshold`` is set for a binary problem, use
        ``decision_function >= threshold`` instead.
        """
        if self.kernel == "precomputed" and not np.all(np.isfinite(X)):
             X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

        if self.decision_threshold is None:
            return self.model.predict(X)

        scores = self.decision_function(X)
        y_pred = (scores >= float(self.decision_threshold)).astype(int)
        return y_pred

    def predict_proba(self, X):
        """Return probabilities. Requires ``probability=True`` during fitting."""
        if self.kernel == "precomputed" and not np.all(np.isfinite(X)):
             X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        return self.model.predict_proba(X)

    def decision_function(self, X):
        """Expose ``SVC.decision_function()`` for threshold selection."""
        if self.kernel == "precomputed" and not np.all(np.isfinite(X)):
             X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        return self.model.decision_function(X)

    def score(self, X, y, **kwargs):
        return accuracy_score(y, self.predict(X))

    def evaluate(self, X, y_true, average: Optional[str] = None):
        """
        Compute standard classification metrics.
        If ``average`` is ``None``, use ``binary`` for two classes and
        ``weighted`` otherwise.
        """
        y_true = np.asarray(y_true)
        num_labels = np.unique(y_true).size
        if average is None:
            average = "binary" if num_labels == 2 else "weighted"

        y_pred = self.predict(X)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average=average),
            "precision": precision_score(y_true, y_pred, average=average),
            "recall": recall_score(y_true, y_pred, average=average),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        }

        try:
            if num_labels == 2:
                if self.probability:
                    y_score = self.predict_proba(X)[:, 1]
                else:
                    y_score = self.decision_function(X)
                metrics["roc_auc"] = roc_auc_score(y_true, y_score)
            else:
                metrics["roc_auc"] = float("nan")
        except Exception as e:
            print(f"[Warning] Could not compute ROC AUC: {e}")
            metrics["roc_auc"] = float("nan")

        return metrics

    def find_best_threshold(
        self, X_val, y_val, metric: str = "f1", num_points: int = 201
    ) -> float:
        y_val = np.asarray(y_val)
        scores = self.decision_function(X_val)
        t_min, t_max = float(scores.min()), float(scores.max())
        thresholds = np.linspace(t_min, t_max, num_points)

        best_t, best_val = thresholds[0], -np.inf
        for t in thresholds:
            y_pred = (scores >= t).astype(int)
            val = f1_score(y_val, y_pred) if metric == "f1" else 0.0
            if val > best_val:
                best_val, best_t = val, float(t)

        self.decision_threshold = best_t
        return best_t

    def set_threshold(self, t: Optional[float]):
        self.decision_threshold = None if t is None else float(t)

    def save(self, filename: str = "svm_model.pkl"):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, filename)
        joblib.dump(self, path)
        print(f"Model saved: {path}")
        return path

    @staticmethod
    def load(path: str) -> "EnhancedSVM":
        return joblib.load(path)

    @staticmethod
    def _to_torch_tensor(X):
        return torch.tensor(X, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")

    def predict_torch(self, X):
        X_t = self._to_torch_tensor(X)
        return self.predict(X_t.cpu().numpy())

    def decision_function_torch(self, X):
        X_t = self._to_torch_tensor(X)
        return self.decision_function(X_t.cpu().numpy())
