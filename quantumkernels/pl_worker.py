import os
import numpy as np
from typing import Optional, Any, Callable
from .helpers import ensure_numpy

_pl_w: Optional[np.ndarray] = None
_pl_nq: Optional[int] = None
_pl_device: Optional[str] = None
_pl_qnode: Optional[Callable] = None
_pl_float_dtype: Optional[np.dtype] = None
_pl_complex_dtype: Optional[np.dtype] = None
_pl_angle_scale: float = 1.0
_pl_re_embed: bool = False
_pl_embed_mode: str = "ryrz"


def pl_worker_init(w_local: np.ndarray, device_name: str, nq: int,
                   float_dtype_str: str = "float64",
                   angle_scale: float = 1.0,
                   re_embed_between_layers: bool = False,
                   embed_mode: str = "ryrz"):
    """Initializer called once per worker process."""
    global _pl_w, _pl_nq, _pl_device, _pl_qnode, _pl_float_dtype, _pl_complex_dtype
    global _pl_angle_scale, _pl_re_embed, _pl_embed_mode

    _pl_w = ensure_numpy(w_local, np.dtype(np.float32) if float_dtype_str == "float32" else np.dtype(np.float64))
    _pl_nq = int(nq)
    _pl_device = str(device_name)
    _pl_qnode = None
    _pl_float_dtype = np.dtype(np.float32) if float_dtype_str == "float32" else np.dtype(np.float64)
    _pl_complex_dtype = np.dtype(np.complex64) if float_dtype_str == "float32" else np.dtype(np.complex128)
    _pl_angle_scale = float(angle_scale)
    _pl_re_embed = bool(re_embed_between_layers)
    _pl_embed_mode = str(embed_mode)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    if _pl_w.ndim != 2 or _pl_w.shape[1] != _pl_nq:
        raise ValueError(f"_pl_w must have shape (L, n_qubits={_pl_nq}); got {_pl_w.shape}.")


def pl_get_qnode():
    """Create the qnode used in this process worker with embedding controls."""
    global _pl_qnode
    if _pl_qnode is None:
        import pennylane as qml
        dev = qml.device(_pl_device, wires=_pl_nq, shots=None, c_dtype=_pl_complex_dtype)

        def _embed(theta):
            s = _pl_angle_scale
            if _pl_embed_mode == "angle":
                qml.AngleEmbedding(s * theta[:_pl_nq], wires=range(_pl_nq), rotation="Y", normalize=False)
            else:
                for i in range(_pl_nq):
                    qml.RY(s * theta[i], wires=i)
                    if _pl_embed_mode == "ryrz":
                        qml.RZ(s * theta[i], wires=i)

        @qml.qnode(dev, interface=None, diff_method=None)
        def _state(theta_row):
            theta = qml.math.asarray(theta_row, dtype=_pl_float_dtype)
            if theta.shape[0] < _pl_nq:
                raise ValueError(f"theta has length {theta.shape[0]} < n_qubits={_pl_nq}")
            if _pl_re_embed:
                L = _pl_w.shape[0]
                for l in range(L):
                    _embed(theta)
                    qml.templates.BasicEntanglerLayers(_pl_w[l:l + 1], wires=range(_pl_nq))
            else:
                _embed(theta)
                qml.templates.BasicEntanglerLayers(_pl_w, wires=range(_pl_nq))
            return qml.state()

        _pl_qnode = _state
    return _pl_qnode


def pl_states_for_rows(rows: list[int], mat: np.ndarray) -> np.ndarray:
    qnode = pl_get_qnode()
    out = np.empty((len(rows), 1 << _pl_nq), dtype=_pl_complex_dtype)
    for t, idx in enumerate(rows):
        out[t] = qnode(mat[idx])
    return out
