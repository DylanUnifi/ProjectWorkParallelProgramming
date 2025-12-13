import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import torch


class EnhancedSVM(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        C=1.0,
        kernel='rbf',
        gamma='scale',
        use_pca=False,
        pca_model=None,
        scaler=None,
        save_path=None,
        probability=False,
        auto_transform=True,
        use_gpu=False,
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.use_pca = use_pca
        self.pca_model = pca_model
        self.scaler = scaler
        self.save_path = save_path or './enhanced_svm.pkl'
        self.auto_transform = auto_transform
        self.use_gpu = use_gpu
        self.probability = probability
        self.xp = None
        self._xp_module_name = None
        self.model = self._build_model()

    def _build_model(self):
        # --- Priorité 1: Tenter CUDA (NVIDIA cuML/cuPy) si use_gpu=True ---
        if self.use_gpu:
            try:
                import cupy as cp  # type: ignore
                from cuml.svm import SVC as cuSVC  # type: ignore

                self.xp = cp
                self._xp_module_name = 'cupy'
                print("[INFO] SVM: Utilisation du GPU CUDA (cuML).")
                return cuSVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=self.probability)
            except Exception as e:
                # Afficher l'avertissement et passer à la prochaine priorité
                print(f"[Warning] GPU SVM CUDA (cuML) indisponible ({e}).")

        # --- Priorité 2: Tenter l'accélération Intel/Optimisation XPU/CPU (sklearnex) ---
        try:
            from sklearnex import patch_sklearn, unpatch_sklearn
            # S'assurer qu'aucun autre patch n'interfère, puis appliquer le patch Intel
            unpatch_sklearn()
            patch_sklearn()
            print("[INFO] SVM: Utilisation de Scikit-learn accéléré par Intel (sklearnex).")
        except ImportError:
            print("[INFO] SVM: sklearnex non trouvé. Utilisation de Scikit-learn standard.")
            # Si sklearnex n'est pas installé, la logique passe au Scikit-learn standard.
            pass

        # --- Priorité 3: CPU standard ou optimisé par Intel (si patch_sklearn a réussi) ---
        import numpy as np
        self.xp = np
        self._xp_module_name = 'numpy'
        
        # SVC ici sera soit le SVC standard, soit le SVC patché par sklearnex
        return SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=self.probability)

    def _transform_input(self, X):
        if not self.auto_transform:
            return X
        X_transformed = X
        if self.scaler is not None:
            X_transformed = self.scaler.transform(X_transformed)
        if self.use_pca and self.pca_model is not None:
            X_transformed = self.pca_model.transform(X_transformed)
        if self.use_gpu and self.xp is not None:
            try:
                X_transformed = self.xp.asarray(X_transformed)
            except Exception as e:
                print(f"[Warning] Could not move data to GPU ({e}); continuing on CPU.")
        return X_transformed

    def fit(self, X, y):
        X_transformed = self._transform_input(X)
        self.model.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self._transform_input(X)
        preds = self.model.predict(X_transformed)
        if hasattr(preds, "get"):
            preds = preds.get()
        return preds

    def predict_proba(self, X):
        """
        Retourne les probabilités (uniquement si probability=True au fit)
        """
        X_transformed = self._transform_input(X)
        probs = self.model.predict_proba(X_transformed)
        if hasattr(probs, "get"):
            probs = probs.get()
        return probs

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
        xp_backup = self.xp
        joblib.dump(self, model_path)
        self.xp = xp_backup
        print(f"✅ Modèle sauvegardé avec succès : {model_path}")

    @staticmethod
    def load(path):
        return joblib.load(path)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['xp'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            if self._xp_module_name == 'cupy':
                import cupy as cp  # type: ignore
                self.xp = cp
            else:
                import numpy as np
                self.xp = np
        except Exception:
            import numpy as np
            self.xp = np

    def to_torch_tensor(self, X):
        return torch.tensor(X, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(X)

    def predict_torch(self, X):
        X_torch = self.to_torch_tensor(X)
        X_cpu = X_torch.cpu().numpy()
        return self.predict(X_cpu)