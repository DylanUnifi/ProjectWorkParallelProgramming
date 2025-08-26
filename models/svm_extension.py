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
    SVM enveloppée avec:
      - support du kernel 'precomputed'
      - class_weight configurable (par défaut "balanced")
      - seuil de décision custom pour classification binaire
      - helpers de sauvegarde/chargement et torch interop
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
        decision_threshold: Optional[float] = None,  # seuil custom (binaire). None => predict() standard
        random_state: Optional[int] = 42,
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

        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=self.probability,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

    # ---------- sklearn API ----------
    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Par défaut, délègue à SVC.predict().
        Si decision_threshold est défini ET problème binaire, utilise decision_function >= threshold.
        """
        if self.decision_threshold is None:
            return self.model.predict(X)

        # seuil custom → nécessite binaire
        scores = self.decision_function(X)
        y_pred = (scores >= float(self.decision_threshold)).astype(int)
        return y_pred

    def predict_proba(self, X):
        """Probabilités (nécessite probability=True au fit)."""
        return self.model.predict_proba(X)

    def decision_function(self, X):
        """Expose la decision_function de SVC (utile pour choisir un seuil)."""
        return self.model.decision_function(X)

    def score(self, X, y, **kwargs):
        return accuracy_score(y, self.predict(X))

    # ---------- métriques ----------
    def evaluate(self, X, y_true, average: Optional[str] = None):
        """
        Calcule les métriques classiques. Si average=None:
          - 'binary' si 2 classes, sinon 'weighted'.
        Respecte le seuil custom s'il est défini.
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

        # AUC seulement si binaire
        try:
            if num_labels == 2:
                # si probability=True, on peut utiliser predict_proba[:,1]
                # sinon, decision_function comme score continu
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

    # ---------- seuil optimal ----------
    def find_best_threshold(
        self, X_val, y_val, metric: str = "f1", num_points: int = 201
    ) -> float:
        """
        Cherche le meilleur seuil sur decision_function pour un problème binaire.
        metric: 'f1' (par défaut). Peut être étendu si besoin.
        """
        y_val = np.asarray(y_val)
        assert np.unique(y_val).size == 2, "best_threshold nécessite un problème binaire."

        scores = self.decision_function(X_val)
        t_min, t_max = float(scores.min()), float(scores.max())
        thresholds = np.linspace(t_min, t_max, num_points)

        best_t, best_val = thresholds[0], -np.inf
        for t in thresholds:
            y_pred = (scores >= t).astype(int)
            if metric == "f1":
                val = f1_score(y_val, y_pred)
            else:
                raise ValueError(f"metric '{metric}' non supportée")
            if val > best_val:
                best_val, best_t = val, float(t)

        self.decision_threshold = best_t
        return best_t

    def set_threshold(self, t: Optional[float]):
        """Définit (ou efface avec None) le seuil de décision custom."""
        self.decision_threshold = None if t is None else float(t)

    # ---------- sauvegarde / chargement ----------
    def save(self, filename: str = "svm_model.pkl"):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, filename)
        joblib.dump(self, path)
        print(f"Model saved: {path}")
        return path

    @staticmethod
    def load(path: str) -> "EnhancedSVM":
        return joblib.load(path)

    # ---------- torch helpers ----------
    @staticmethod
    def _to_torch_tensor(X):
        return torch.tensor(X, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")

    def predict_torch(self, X):
        X_t = self._to_torch_tensor(X)
        return self.predict(X_t.cpu().numpy())

    def decision_function_torch(self, X):
        X_t = self._to_torch_tensor(X)
        return self.decision_function(X_t.cpu().numpy())
