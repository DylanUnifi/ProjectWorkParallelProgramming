import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pennylane as qml

# 1. Residual MLP Block
class ResidualMLPBlock(nn.Module):
    def __init__(self, in_features, out_features, downsample=False, dropout=0.3):
        super().__init__()
        self.downsample = None
        self.fc1 = nn.Linear(in_features, out_features)
        self.ln1 = nn.LayerNorm(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        if downsample or in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features)
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.dropout(out)
        out = self.ln2(self.fc2(out))
        out += identity
        return F.relu(out)

# 2. Picklable QNode creation (pour multiprocesseur)
def create_qnode(n_qubits, n_layers, device_name="lightning.qubit"):
    dev = qml.device(device_name, wires=n_qubits, shots=None)
    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def qnode(inputs, weights):
        if not isinstance(weights, dict) or "weights" not in weights:
            raise ValueError("Weights must be dict with key 'weights'")
        w = weights["weights"]
        if w.ndim != 2:
            raise ValueError(f"Weights tensor must be 2D, got shape {w.shape}")
        inputs = inputs.flatten()
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
            qml.RZ(inputs[i], wires=i)
        qml.templates.BasicEntanglerLayers(w, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return qnode

class QuantumCircuitBuilder:
    def __init__(self, n_qubits, n_layers=2, device_name="lightning.qubit"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_name = device_name
        self.qnode = create_qnode(n_qubits, n_layers, device_name)

    def create_layer(self):
        weight_shapes = {"weights": (self.n_layers, self.n_qubits)}
        layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        for name, param in layer.named_parameters():
            if "weights" in name:
                nn.init.normal_(param, mean=0.0, std=0.01)
        return layer

    def eval_sample(self, sample_np, weights):
        inputs = torch.tensor(sample_np, dtype=torch.float32)
        outputs = self.qnode(inputs, weights)
        return torch.tensor(outputs).detach().numpy()

class QuantumWorker:
    def __init__(self, n_qubits, n_layers, device_name="lightning.qubit"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_name = device_name

    def __call__(self, args):
        sample_np, weights = args
        builder = QuantumCircuitBuilder(self.n_qubits, self.n_layers, self.device_name)
        return builder.eval_sample(sample_np, weights)


class HybridQCNNBase(nn.Module):
    def __init__(self, input_size, hidden_sizes=None, n_qubits=4, n_layers=2,
                 dropout=0.3, parallel=True, device="cpu"):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 16, 8]
        self.block1 = ResidualMLPBlock(input_size, hidden_sizes[0], downsample=True, dropout=dropout)
        self.block2 = ResidualMLPBlock(hidden_sizes[0], hidden_sizes[1], downsample=True, dropout=dropout)
        self.block3 = ResidualMLPBlock(hidden_sizes[1], hidden_sizes[2], downsample=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.quantum_fc_input = nn.Linear(hidden_sizes[2], n_qubits)
        self.builder = QuantumCircuitBuilder(n_qubits=n_qubits, n_layers=n_layers)
        self.quantum_layer = self.builder.create_layer()
        self.bn_q = nn.LayerNorm(n_qubits)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.parallel = parallel
        self.worker = QuantumWorker(n_qubits, n_layers)

    def forward_quantum(self, x, pool=None):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        # x = self.bn_q(x)
        x = torch.tanh(self.quantum_fc_input(x)) * np.pi
        if self.parallel and pool is not None:
            weights_tensor = next(self.quantum_layer.parameters()).to(x.device)
            weights_dict = {"weights": weights_tensor}
            samples_np = x.detach().cpu().numpy()
            args_list = [(sample, weights_dict) for sample in samples_np]
            results = pool.map(self.worker, args_list)
            outputs = [torch.tensor(r, device=x.device) for r in results]
            x = torch.stack(outputs, dim=0)
        else:
            x = self.quantum_layer(x)
        x = self.bn_q(x)
        return x

class HybridQCNNFeatures(HybridQCNNBase):
    def forward(self, x, pool=None):
        return self.forward_quantum(x, pool)

class HybridQCNNBinaryClassifier(HybridQCNNBase):
    def __init__(self, input_size, hidden_sizes=None, n_qubits=4, n_layers=2,
                 dropout=0.3, parallel=True, device="cpu"):
        super().__init__(input_size, hidden_sizes, n_qubits, n_layers, dropout, parallel, device)
        self.final_fc = nn.Linear(n_qubits, 1)
    def forward(self, x, pool=None):
        x = self.forward_quantum(x,pool)
        x = self.final_fc(x)
        return torch.sigmoid(x)
