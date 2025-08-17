import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import torch

class EnhancedSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='precomputed', gamma='scale', use_pca=False, pca_model=None, save_path=None, probability=False):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.use_pca = use_pca
        self.pca_model = pca_model
        self.save_path = save_path or './enhanced_svm.pkl'
        self.model = SVC(C=self.C, kernel=self.kernel, probability=probability)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.kernel != 'precomputed':
            assert X.shape[1] == self.model.support_vectors_.shape[1], \
                f"Expected {self.model.support_vectors_.shape[1]} features, got {X.shape[1]}"
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Retourne les probabilités (uniquement si probability=True au fit)
        """
        return self.model.predict_proba(X)

    def score(self, X, y, **kwargs):
        return self.model.score(X, y)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
        }

        try:
            y_proba = self.predict_proba(X)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except Exception as e:
            print(f"[Warning] Could not compute ROC AUC: {e}")
            metrics['roc_auc'] = float('nan')

        return metrics

    def save(self):
        os.makedirs(self.save_path, exist_ok=True)
        model_path = os.path.join(self.save_path, "svm_model.pkl")
        joblib.dump(self, model_path)
        print(f"✅ Modèle sauvegardé avec succès : {model_path}")

    @staticmethod
    def load(path):
        return joblib.load(path)

    def to_torch_tensor(self, X):
        return torch.tensor(X, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(X)

    def predict_torch(self, X):
        X_torch = self.to_torch_tensor(X)
        X_cpu = X_torch.cpu().numpy()
        return self.predict(X_cpu)
