# check_nan.py
import numpy as np
from scripts.pipeline_backends import compute_kernel_matrix

print("Generating test data...")
X = np.random.uniform(-3, 3, (100, 10)).astype(np.float64)
W = np.random.normal(0, 0.1, (2, 10)).astype(np.float64)

print("Computing kernel with cuda_states (float64)...")
K = compute_kernel_matrix(
    X, weights=W, 
    gram_backend="cuda_states", 
    device_name="lightning.gpu",
    dtype="float64"
)

n_nans = np.isnan(K).sum()
n_infs = np.isinf(K).sum()

print("\nResult:")
print(f"   Min: {K.min():.4f}")
print(f"   Max: {K.max():.4f}")
print(f"   NaNs: {n_nans}")
print(f"   Infs: {n_infs}")

if n_nans == 0 and n_infs == 0:
    print("Success: the backend is numerically stable.")
else:
    print("Failure: numerical errors are still present.")
