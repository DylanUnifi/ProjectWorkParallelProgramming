import numpy as np
import torch as th
import pennylane as qml


def gram_torch_stream(a_np, b_np, *, weights_np, device_name, tile_size,
                      symmetric, float_dt, ret_dt, angle_scale,
                      re_embed_between_layers, embed_mode):
    """Compute Gram matrix with PennyLane + Torch GPU backend."""
    assert "gpu" in device_name.lower(), "torch_stream requires GPU device (e.g. lightning.gpu)"
    nq = a_np.shape[1]
    n = a_np.shape[0]
    m = n if b_np is None else b_np.shape[0]

    t_float = th.float32 if float_dt == np.float32 else th.float64
    t_complex = th.complex64 if float_dt == np.float32 else th.complex128
    t_ret = th.float32 if ret_dt == np.float32 else th.float64

    a = th.from_numpy(np.ascontiguousarray(a_np)).to("cuda", dtype=t_float)
    b = a if b_np is None else th.from_numpy(np.ascontiguousarray(b_np)).to("cuda", dtype=t_float)
    w = th.from_numpy(np.ascontiguousarray(weights_np)).to("cuda", dtype=t_float)

    dev = qml.device(device_name, wires=nq, shots=None,
                     c_dtype=(np.complex64 if float_dt == np.float32 else np.complex128))

    def _embed(v):
        s = float(angle_scale)
        if embed_mode == "angle":
            qml.AngleEmbedding(s * v[:nq], wires=range(nq), rotation="Y", normalize=False)
        else:
            for i in range(nq):
                qml.RY(s * v[i], wires=i)
                if embed_mode == "ryrz":
                    qml.RZ(s * v[i], wires=i)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def _state(theta_row):
        if re_embed_between_layers:
            L = w.shape[0]
            for l in range(L):
                _embed(theta_row)
                qml.templates.BasicEntanglerLayers(w[l:l + 1], wires=range(nq))
        else:
            _embed(theta_row)
            qml.templates.BasicEntanglerLayers(w, wires=range(nq))
        return qml.state()

    def build_states(x_block):
        try:
            from torch import vmap
            return vmap(_state)(x_block).to(dtype=t_complex)
        except Exception:
            states = [_state(x_block[t]) for t in range(x_block.shape[0])]
            return th.stack(states, dim=0).to(dtype=t_complex)

    k = th.empty((n, m), device="cuda", dtype=t_ret)
    with th.no_grad():
        for i0 in range(0, n, tile_size):
            i1 = min(i0 + tile_size, n)
            sa_x = build_states(a[i0:i1])
            for j0 in range(0 if not (symmetric and b is a) else i0, m, tile_size):
                j1 = min(j0 + tile_size, m)
                sb_x = sa_x if (b is a and j0 == i0) else build_states(b[j0:j1])
                g = sa_x @ sb_x.conj().transpose(0, 1)
                k_blk = (g.abs() ** 2).to(dtype=t_ret)
                k[i0:i1, j0:j1] = k_blk
                if symmetric and b is a and j0 > i0:
                    k[j0:j1, i0:i1] = k_blk.transpose(0, 1)
            del sa_x
            th.cuda.empty_cache()

    return k.detach().cpu().numpy().astype(ret_dt, copy=False)
