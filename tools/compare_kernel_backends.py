import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import KernelCenterer, MinMaxScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader.utils import load_dataset_by_name
from scripts.pipeline_backends import compute_kernel_matrix


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def extract_features(dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    all_features = []
    all_labels = []
    for images, labels in tqdm(loader, desc="Extracting", leave=False):
        batch_size = images.shape[0]
        all_features.append(images.view(batch_size, -1).numpy())
        all_labels.append(labels.numpy())
    X = np.vstack(all_features).astype(np.float32)
    y = np.concatenate(all_labels).astype(np.int64)
    return X, y


def select_balanced_blocks(labels, block_size, num_blocks):
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if classes.size != 2:
        raise ValueError(f"Expected exactly 2 classes, found {classes.tolist()}")

    per_class = {classes[0]: block_size // 2, classes[1]: block_size - (block_size // 2)}
    pools = {cls: np.flatnonzero(labels == cls) for cls in classes}

    blocks = []
    cursor = {cls: 0 for cls in classes}
    for block_id in range(num_blocks):
        indices = []
        for cls in classes:
            need = per_class[cls]
            start = cursor[cls]
            stop = start + need
            if stop > pools[cls].size:
                raise ValueError(
                    f"Not enough samples to build block {block_id} for class {int(cls)}: "
                    f"need {need}, have {pools[cls].size - start}"
                )
            indices.extend(pools[cls][start:stop].tolist())
            cursor[cls] = stop
        blocks.append(np.asarray(indices, dtype=np.int64))
    return blocks


def normalize_square(K):
    K = np.asarray(K, dtype=np.float64).copy()
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    K /= d[:, None] * d[None, :]
    np.fill_diagonal(K, 1.0)
    return K


def normalize_cross(K_xy, diag_x, diag_y):
    K_xy = np.asarray(K_xy, dtype=np.float64).copy()
    dx = np.sqrt(np.clip(np.asarray(diag_x, dtype=np.float64), 1e-12, None))
    dy = np.sqrt(np.clip(np.asarray(diag_y, dtype=np.float64), 1e-12, None))
    return K_xy / (dx[:, None] * dy[None, :])


def matrix_stats(K):
    K = np.asarray(K, dtype=np.float64)
    stats = {
        "shape": list(K.shape),
        "min": float(np.min(K)),
        "max": float(np.max(K)),
        "mean": float(np.mean(K)),
        "std": float(np.std(K)),
        "fro_norm": float(np.linalg.norm(K, ord="fro")),
    }
    if K.shape[0] == K.shape[1]:
        diag = np.diag(K)
        stats.update(
            {
                "diag_min": float(np.min(diag)),
                "diag_max": float(np.max(diag)),
                "diag_mean": float(np.mean(diag)),
                "diag_std": float(np.std(diag)),
                "symmetry_max_abs": float(np.max(np.abs(K - K.T))),
            }
        )
    return stats


def diff_summary(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    abs_diff = np.abs(candidate - reference)
    idx = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
    ref_norm = np.linalg.norm(reference, ord="fro")
    return {
        "max_abs": float(abs_diff[idx]),
        "mean_abs": float(np.mean(abs_diff)),
        "fro_rel": float(np.linalg.norm(candidate - reference, ord="fro") / (ref_norm + 1e-12)),
        "argmax": [int(idx[0]), int(idx[1])],
        "reference_at_argmax": float(reference[idx]),
        "candidate_at_argmax": float(candidate[idx]),
    }


def print_json(label, payload):
    print(f"{label}:")
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_backend(name, X, Y, weights, args):
    backend_params = {
        "weights": weights,
        "device_name": args.pl_device,
        "tile_size": args.torch_tile_size if name == "torch" else args.tile_size,
        "symmetric": Y is None,
        "n_workers": 0,
        "dtype": args.dtype,
        "return_dtype": args.return_dtype,
        "gram_backend": name,
        "progress": True,
        "desc": f"{name}_{'square' if Y is None else 'cross'}",
        "angle_scale": args.angle_scale,
        "re_embed_between_layers": False,
        "embed_mode": args.embed_mode,
        "normalize": False,
        "state_tile": args.state_tile,
        "autotune": args.autotune,
        "precompute_all_states": args.precompute_all_states,
        "vram_fraction": args.vram_fraction,
        "dynamic_batch": False,
        "num_streams": args.num_streams,
        "learn_tiles": args.learn_tiles,
        "use_cuda_graphs": False,
        "profile_memory": False,
        "verbose_profile": False,
        "use_pinned_memory": False,
        "use_cuda_streams": False,
        "use_amp": False,
        "use_compile": False,
    }
    return compute_kernel_matrix(X, Y=Y, **backend_params)


def run_native_normalized_backend(name, X, Y, weights, args):
    backend_params = {
        "weights": weights,
        "device_name": args.pl_device,
        "tile_size": args.torch_tile_size if name == "torch" else args.tile_size,
        "symmetric": Y is None,
        "n_workers": 0,
        "dtype": args.dtype,
        "return_dtype": args.return_dtype,
        "gram_backend": name,
        "progress": False,
        "desc": f"{name}_native_norm",
        "angle_scale": args.angle_scale,
        "re_embed_between_layers": False,
        "embed_mode": args.embed_mode,
        "normalize": True,
        "state_tile": args.state_tile,
        "autotune": args.autotune,
        "precompute_all_states": args.precompute_all_states,
        "vram_fraction": args.vram_fraction,
        "dynamic_batch": False,
        "num_streams": args.num_streams,
        "learn_tiles": args.learn_tiles,
        "use_cuda_graphs": False,
        "profile_memory": False,
        "verbose_profile": False,
        "use_pinned_memory": False,
        "use_cuda_streams": False,
        "use_amp": False,
        "use_compile": False,
    }
    return compute_kernel_matrix(X, Y=Y, **backend_params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train-subset", type=int, default=1000)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--embed-mode", type=str, default="ryrz", choices=["angle", "ry", "ryrz"])
    parser.add_argument("--angle-scale", type=float, default=0.1)
    parser.add_argument("--pl-device", type=str, default="lightning.gpu")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="float64")
    parser.add_argument("--return-dtype", type=str, default="float64")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--torch-tile-size", type=int, default=64)
    parser.add_argument("--state-tile", type=int, default=64)
    parser.add_argument("--vram-fraction", type=float, default=0.90)
    parser.add_argument("--num-streams", type=int, default=1)
    parser.add_argument("--autotune", action="store_true", default=False)
    parser.add_argument("--precompute-all-states", action="store_true", default=True)
    parser.add_argument("--no-precompute", action="store_false", dest="precompute_all_states")
    parser.add_argument("--learn-tiles", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_dataset, _ = load_dataset_by_name(
        name=config["dataset"]["name"],
        binary_classes=config.get("dataset", {}).get("binary_classes", [3, 8]),
        grayscale=config.get("dataset", {}).get("grayscale", True),
        root=config.get("dataset", {}).get("root", "./data"),
    )
    if args.train_subset:
        train_dataset = Subset(train_dataset, range(min(len(train_dataset), args.train_subset)))

    X_raw, y = extract_features(train_dataset)
    n_components = args.pca_components or config.get("svm", {}).get("pca_components", X_raw.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    X = pca.fit_transform(X_raw)

    scale_range = (0, np.pi) if args.embed_mode == "angle" else (0, 1)
    scaler = MinMaxScaler(feature_range=scale_range)
    X = scaler.fit_transform(X)

    blocks = select_balanced_blocks(y, args.block_size, num_blocks=2)
    idx_a, idx_b = blocks
    X_a, y_a = X[idx_a], y[idx_a]
    X_b, y_b = X[idx_b], y[idx_b]

    n_layers = config.get("pennylane", {}).get("layers", 2)
    rng = np.random.default_rng(SEED)
    weights = rng.normal(0, 0.1, (n_layers, n_components)).astype(
        np.float32 if args.dtype == "float32" else np.float64
    )

    print(
        f"Comparing backends on {config['dataset']['name']} with "
        f"{args.block_size}x{args.block_size} balanced blocks, "
        f"embed={args.embed_mode}, device={args.pl_device}, dtype={args.dtype}"
    )
    print(f"Block A labels: {y_a.tolist()}")
    print(f"Block B labels: {y_b.tolist()}")

    K_torch_aa = run_backend("torch", X_a, None, weights, args)
    K_cuda_aa = run_backend("cuda_states", X_a, None, weights, args)
    K_torch_bb = run_backend("torch", X_b, None, weights, args)
    K_cuda_bb = run_backend("cuda_states", X_b, None, weights, args)
    K_torch_ab = run_backend("torch", X_a, X_b, weights, args)
    K_cuda_ab = run_backend("cuda_states", X_a, X_b, weights, args)

    diag_torch_a = np.diag(K_torch_aa)
    diag_cuda_a = np.diag(K_cuda_aa)
    diag_torch_b = np.diag(K_torch_bb)
    diag_cuda_b = np.diag(K_cuda_bb)

    K_torch_aa_norm = normalize_square(K_torch_aa)
    K_cuda_aa_norm = normalize_square(K_cuda_aa)
    K_torch_bb_norm = normalize_square(K_torch_bb)
    K_cuda_bb_norm = normalize_square(K_cuda_bb)
    K_torch_ab_norm = normalize_cross(K_torch_ab, diag_torch_a, diag_torch_b)
    K_cuda_ab_norm = normalize_cross(K_cuda_ab, diag_cuda_a, diag_cuda_b)

    centerer_torch = KernelCenterer()
    K_torch_aa_centered = centerer_torch.fit_transform(K_torch_aa_norm)
    K_torch_ab_centered = centerer_torch.transform(K_torch_ab_norm)

    centerer_cuda = KernelCenterer()
    K_cuda_aa_centered = centerer_cuda.fit_transform(K_cuda_aa_norm)
    K_cuda_ab_centered = centerer_cuda.transform(K_cuda_ab_norm)

    native_torch_aa_norm = run_native_normalized_backend("torch", X_a, None, weights, args)
    native_cuda_aa_norm = run_native_normalized_backend("cuda_states", X_a, None, weights, args)
    native_torch_ab_norm = run_native_normalized_backend("torch", X_a, X_b, weights, args)
    native_cuda_ab_norm = run_native_normalized_backend("cuda_states", X_a, X_b, weights, args)

    print_json(
        "metadata",
        {
            "config": args.config,
            "train_subset": int(args.train_subset),
            "pca_components": int(n_components),
            "layers": int(n_layers),
            "block_size": int(args.block_size),
            "device": args.pl_device,
            "dtype": args.dtype,
            "return_dtype": args.return_dtype,
        },
    )

    print_json(
        "torch_stats",
        {
            "raw_square_a": matrix_stats(K_torch_aa),
            "raw_square_b": matrix_stats(K_torch_bb),
            "raw_cross_ab": matrix_stats(K_torch_ab),
            "normalized_square_a": matrix_stats(K_torch_aa_norm),
            "normalized_cross_ab": matrix_stats(K_torch_ab_norm),
            "centered_square_a": matrix_stats(K_torch_aa_centered),
            "centered_cross_ab": matrix_stats(K_torch_ab_centered),
        },
    )

    print_json(
        "cuda_stats",
        {
            "raw_square_a": matrix_stats(K_cuda_aa),
            "raw_square_b": matrix_stats(K_cuda_bb),
            "raw_cross_ab": matrix_stats(K_cuda_ab),
            "normalized_square_a": matrix_stats(K_cuda_aa_norm),
            "normalized_cross_ab": matrix_stats(K_cuda_ab_norm),
            "centered_square_a": matrix_stats(K_cuda_aa_centered),
            "centered_cross_ab": matrix_stats(K_cuda_ab_centered),
        },
    )

    print_json(
        "backend_diffs",
        {
            "raw_square_a": diff_summary(K_torch_aa, K_cuda_aa),
            "raw_square_b": diff_summary(K_torch_bb, K_cuda_bb),
            "raw_cross_ab": diff_summary(K_torch_ab, K_cuda_ab),
            "normalized_square_a": diff_summary(K_torch_aa_norm, K_cuda_aa_norm),
            "normalized_cross_ab": diff_summary(K_torch_ab_norm, K_cuda_ab_norm),
            "centered_square_a": diff_summary(K_torch_aa_centered, K_cuda_aa_centered),
            "centered_cross_ab": diff_summary(K_torch_ab_centered, K_cuda_ab_centered),
        },
    )

    print_json(
        "native_normalize_flag_checks",
        {
            "torch_square_flag_vs_raw": diff_summary(K_torch_aa, native_torch_aa_norm),
            "cuda_square_flag_vs_external": diff_summary(K_cuda_aa_norm, native_cuda_aa_norm),
            "torch_cross_flag_vs_raw": diff_summary(K_torch_ab, native_torch_ab_norm),
            "cuda_cross_flag_vs_external": diff_summary(K_cuda_ab_norm, native_cuda_ab_norm),
        },
    )


if __name__ == "__main__":
    main()
