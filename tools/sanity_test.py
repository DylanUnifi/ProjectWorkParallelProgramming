from models.hybrid_qcnn import HybridQCNNFeatures
import torch, numpy as np

inp = torch.randn(4, 28*28)  # ex. Fashion-MNIST flatten
m = HybridQCNNFeatures(input_size=28*28)
A = m.compute_angles(inp)          # -> [4, n_qubits]
W = m.get_entangler_weights()      # -> [n_layers, n_qubits]
print(A.shape, W.shape)
