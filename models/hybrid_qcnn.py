# models/hybrid_qcnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pennylane as qml

# ------------------------------
# 1) Residual MLP Block
# ------------------------------
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

# ------------------------------
# 2) Couche quantique (TorchLayer)
#    — QNode batch-friendly, sortie vecteur
# ------------------------------
def create_quantum_layer(n_qubits, n_layers=2, backend="lightning.qubit"):
    dev = qml.device(backend, wires=n_qubits, shots=None)  # différentiable

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(inputs, weights):
        # inputs: (n_qubits,) pour UN sample (TorchLayer gère la 1ère dim = batch)
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.reshape(-1)  # rester côté Torch
        else:
            # fallback non-torch uniquement si besoin
            inputs = qml.numpy.ravel(inputs)

        # encoding simple mais riche : RY + RZ par qubit
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
            qml.RZ(inputs[i], wires=i)

        qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))

        # >>> TOUJOURS un vecteur de longueur n_qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    # init douce des poids quantiques
    for name, param in layer.named_parameters():
        if "weights" in name:
            nn.init.normal_(param, mean=0.0, std=0.01)

    return layer

# ------------------------------
# 3) Classifier hybride (si besoin end-to-end)
# ------------------------------
class HybridQCNNBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 16, 8], n_qubits=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.block1 = ResidualMLPBlock(input_size, hidden_sizes[0], downsample=True, dropout=dropout)
        self.block2 = ResidualMLPBlock(hidden_sizes[0], hidden_sizes[1], downsample=True, dropout=dropout)
        self.block3 = ResidualMLPBlock(hidden_sizes[1], hidden_sizes[2], downsample=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.quantum_fc_input = nn.Linear(hidden_sizes[2], n_qubits)
        self.quantum_layer = create_quantum_layer(n_qubits, n_layers)
        self.bn_q = nn.LayerNorm(n_qubits)
        self.final_fc = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        x = torch.tanh(self.quantum_fc_input(x)) * np.pi  # mapping [-π, π]

        # Appel par échantillon (robuste, compatible anciens runs)
        outputs = []
        for sample in x:
            q_out = self.quantum_layer(sample.unsqueeze(0))  # (1, n_qubits)
            outputs.append(q_out)
        x = torch.cat(outputs, dim=0)  # (B, n_qubits)

        x = self.bn_q(x)
        x = self.final_fc(x)
        return torch.sigmoid(x)

# ------------------------------
# 4) Extracteur de features pour ton pipeline QSVM
#     - forward = renvoie les angles (B, n_qubits)
#     - compute_angles = utilisé par ton script
#     - get_entangler_weights = retourne (n_layers, n_qubits)
# ------------------------------
class HybridQCNNFeatures(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 16, 8], n_qubits=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.block1 = ResidualMLPBlock(input_size, hidden_sizes[0], downsample=True, dropout=dropout)
        self.block2 = ResidualMLPBlock(hidden_sizes[0], hidden_sizes[1], downsample=True, dropout=dropout)
        self.block3 = ResidualMLPBlock(hidden_sizes[1], hidden_sizes[2], downsample=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)

        # Projection MLP -> n_qubits (angles)
        self.quantum_fc_input = nn.Linear(hidden_sizes[2], self.n_qubits)

        # IMPORTANT : on instancie aussi la TorchLayer pour porter les poids
        # (même si on ne l’utilise pas dans forward)
        self.quantum_layer = create_quantum_layer(self.n_qubits, self.n_layers)

        # attribut libre si ton script l’assigne
        self.use_torchlayer = True

    def _mlp_trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        return x

    def compute_angles(self, x: torch.Tensor) -> torch.Tensor:
        """MLP -> projection n_qubits -> tanh*π. Retourne [B, n_qubits]."""
        self.eval()
        with torch.no_grad():
            h = self._mlp_trunk(x)
            angles = torch.tanh(self.quantum_fc_input(h)) * np.pi
        return angles

    def get_entangler_weights(self) -> np.ndarray:
        """Retourne les poids (n_layers, n_qubits) de la TorchLayer PennyLane."""
        try:
            w = next(self.quantum_layer.parameters()).detach().cpu().numpy()
        except StopIteration:
            raise RuntimeError("quantum_layer ne contient pas de paramètres 'weights'.")
        if w.ndim != 2 or w.shape != (self.n_layers, self.n_qubits):
            raise ValueError(f"Poids inattendus: shape={w.shape}, attendu=({self.n_layers},{self.n_qubits})")
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les ANGLES uniquement (B, n_qubits) — utilisé par ton pré-train + extraction."""
        h = self._mlp_trunk(x)
        angles = torch.tanh(self.quantum_fc_input(h)) * np.pi
        return angles
