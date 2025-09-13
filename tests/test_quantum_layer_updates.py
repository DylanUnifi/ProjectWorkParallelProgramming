import torch
from models.hybrid_qcnn import HybridQCNNBinaryClassifier

def test_quantum_layer_parameters_update():
    model = HybridQCNNBinaryClassifier(
        input_size=4,
        hidden_sizes=[4, 4, 4],
        n_qubits=2,
        n_layers=1,
        parallel=False,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(2, 4)
    y = torch.ones(2, 1)

    optimizer.zero_grad()
    before = next(model.quantum_layer.parameters()).clone()
    out = model(x)
    loss = torch.nn.functional.binary_cross_entropy(out, y)
    loss.backward()
    optimizer.step()
    after = next(model.quantum_layer.parameters())
    assert not torch.allclose(before, after)