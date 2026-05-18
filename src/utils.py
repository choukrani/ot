from __future__ import annotations

import ot
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, List, Optional, Sequence, Tuple

from core import sinkhorn_log

Array = np.ndarray


def _to_numpy_image(img: object) -> Array:
    if hasattr(img, "detach") and hasattr(img, "cpu") and hasattr(img, "numpy"):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)

    arr = np.asarray(arr, dtype=np.float64)

    # Accept (28, 28), (1, 28, 28), or (28, 28, 1)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {arr.shape}")

    return arr


def mnist_image_to_histogram(img: object, eta: float = 1e-8) -> Array:
    if eta < 0:
        raise ValueError("eta must be nonnegative.")

    arr = _to_numpy_image(img)

    # Map to [0, 1] if the image looks like uint8-style intensities.
    if arr.max() > 1.0:
        arr = arr / 255.0

    arr = arr + eta
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("Image must have strictly positive total mass after preprocessing.")

    hist = arr / total
    return hist.reshape(-1)


def batch_images_to_histograms(images: Sequence[object], eta: float = 1e-8) -> Array:
    if len(images) == 0:
        raise ValueError("images must be non-empty.")

    H = np.stack([mnist_image_to_histogram(img, eta=eta) for img in images], axis=0)
    return H


def select_examples_by_label(
    dataset: Sequence[Tuple[object, int]],
    label: int,
    k: int,
    seed: Optional[int] = None,
) -> Tuple[List[object], Array]:
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    candidate_indices: List[int] = []
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        if int(y) == int(label):
            candidate_indices.append(idx)

    if len(candidate_indices) < k:
        raise ValueError(
            f"Requested k={k} examples of label {label}, but only found {len(candidate_indices)}."
        )

    rng = np.random.default_rng(seed)
    chosen = np.array(candidate_indices, dtype=np.int64)
    if len(chosen) > k:
        chosen = rng.choice(chosen, size=k, replace=False)
        chosen = np.sort(chosen)

    images = [dataset[int(idx)][0] for idx in chosen]
    return images, chosen


def pairwise_sinkhorn_cost_matrix(
    H,
    C,
    eps=0.02,
    max_iter=1500,
    tol=1e-5,
    verbose=True,
):
    H = np.asarray(H, dtype=np.float64)
    N = H.shape[0]

    D = np.zeros((N, N), dtype=np.float64)
    iters = np.zeros((N, N), dtype=int)
    conv = np.zeros((N, N), dtype=bool)

    t0 = time.time()

    for i in range(N):
        for j in range(i, N):
            out = sinkhorn_log(H[i], H[j], C, eps=eps, max_iter=max_iter, tol=tol)

            D[i, j] = D[j, i] = out["cost"]
            iters[i, j] = iters[j, i] = out["n_iter"]
            conv[i, j] = conv[j, i] = out["converged"]

            if verbose:
                print(
                    f"[{i},{j}] cost={out['cost']:.8f} | "
                    f"iters={out['n_iter']} | converged={out['converged']}"
                )

    runtime_sec = time.time() - t0
    return D, iters, conv, runtime_sec


def euclidean_average_histogram(H):
    H = np.asarray(H, dtype=np.float64)
    avg = H.mean(axis=0)
    avg = np.maximum(avg, 0.0)
    avg /= avg.sum()
    return avg


def wasserstein_barycenter_same_support(
    H,
    C,
    eps,
    weights=None,
    numItermax=5000,
    stopThr=1e-6,
):
    H = np.asarray(H, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    if H.ndim != 2:
        raise ValueError("H must have shape (K, n).")
    K = H.shape[0]

    if weights is None:
        weights = np.ones(K, dtype=np.float64) / K
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()

    A = H.T  

    t0 = time.time()

    try:
        bary = ot.bregman.barycenter(
            A,
            C,
            reg=eps,
            weights=weights,
            method="sinkhorn_log",
            numItermax=numItermax,
            stopThr=stopThr,
            log=False,
            verbose=False,
            warn=True,
        )
    except Exception:
        try:
            bary = ot.bregman.barycenter(
                A,
                C,
                reg=eps,
                weights=weights,
                method="sinkhorn_stabilized",
                numItermax=numItermax,
                stopThr=stopThr,
                log=False,
                verbose=False,
                warn=True,
            )
        except Exception:
            bary = ot.bregman.barycenter(
                A,
                C,
                reg=eps,
                weights=weights,
                numItermax=numItermax,
                stopThr=stopThr,
                log=False,
                verbose=False,
                warn=True,
            )

    runtime_sec = time.time() - t0

    bary = np.asarray(bary, dtype=np.float64).ravel()
    bary = np.maximum(bary, 0.0)
    bary /= bary.sum()

    return bary, runtime_sec


def weighted_group_euclidean_average(H_A, H_B, lam):
    H_A = np.asarray(H_A, dtype=np.float64)
    H_B = np.asarray(H_B, dtype=np.float64)

    avg_A = euclidean_average_histogram(H_A)
    avg_B = euclidean_average_histogram(H_B)

    avg = (1.0 - lam) * avg_A + lam * avg_B
    avg = np.maximum(avg, 0.0)
    avg /= avg.sum()
    return avg


def weighted_group_wasserstein_barycenter(
    H_A,
    H_B,
    C,
    lam,
    eps,
    numItermax=5000,
    stopThr=1e-6,
):
    H_A = np.asarray(H_A, dtype=np.float64)
    H_B = np.asarray(H_B, dtype=np.float64)

    K_A = H_A.shape[0]
    K_B = H_B.shape[0]

    H = np.vstack([H_A, H_B])

    w_A = np.ones(K_A, dtype=np.float64) * ((1.0 - lam) / K_A)
    w_B = np.ones(K_B, dtype=np.float64) * (lam / K_B)
    weights = np.concatenate([w_A, w_B])

    bary, runtime_sec = wasserstein_barycenter_same_support(
        H,
        C,
        eps=eps,
        weights=weights,
        numItermax=numItermax,
        stopThr=stopThr,
    )
    return bary, runtime_sec


def mnist_image_to_unit_interval_vector(img):
    arr = np.asarray(img, dtype=np.float64)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr.reshape(-1)


def batch_images_to_unit_interval_vectors(images):
    return np.stack([mnist_image_to_unit_interval_vector(img) for img in images], axis=0)


def select_balanced_subset(dataset, per_label=10, labels=range(10), seed=0):
    """
    Select a balanced subset from a labeled dataset.
    Returns images, labels, indices.
    """
    rng = np.random.default_rng(seed)

    all_images = []
    all_labels = []
    all_indices = []

    for label in labels:
        imgs, idx = select_examples_by_label(dataset, label=label, k=per_label, seed=int(rng.integers(1e9)))
        all_images.extend(imgs)
        all_labels.extend([label] * len(imgs))
        all_indices.extend(idx.tolist())

    return all_images, np.array(all_labels, dtype=int), np.array(all_indices, dtype=int)


def euclidean_query_candidate_distances(Q, G):
    """
    Pairwise squared Euclidean distances between query vectors Q and candidate vectors G.
    Q: (Nq, d), G: (Ng, d)
    Returns D: (Nq, Ng)
    """
    Q = np.asarray(Q, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)

    q_norm = np.sum(Q**2, axis=1, keepdims=True)
    g_norm = np.sum(G**2, axis=1, keepdims=True).T
    D = q_norm + g_norm - 2.0 * (Q @ G.T)
    D = np.maximum(D, 0.0)
    return D


def sinkhorn_cost_pot(a, b, C, eps=0.01, numItermax=1000, stopThr=1e-4):
    try:
        val = ot.bregman.sinkhorn2(
            a, b, C,
            reg=eps,
            method="sinkhorn_log",
            numItermax=numItermax,
            stopThr=stopThr,
            log=False,
            warn=False,
            verbose=False,
        )
    except Exception:
        try:
            val = ot.sinkhorn2(
                a, b, C,
                reg=eps,
                method="sinkhorn_log",
                numItermax=numItermax,
                stopThr=stopThr,
                log=False,
                warn=False,
                verbose=False,
            )
        except Exception:
            val = sinkhorn_log(a, b, C, eps=eps, max_iter=numItermax, tol=stopThr)["cost"]

    if isinstance(val, tuple):
        val = val[0]

    return float(np.asarray(val))


def sinkhorn_query_candidate_distances(
    Q_hist,
    G_hist,
    C,
    eps=0.01,
    numItermax=1000,
    stopThr=1e-4,
    verbose=True,
):
    """
    Pairwise entropic OT costs between query histograms and candidate histograms.
    Returns D: (Nq, Ng)
    """
    Q_hist = np.asarray(Q_hist, dtype=np.float64)
    G_hist = np.asarray(G_hist, dtype=np.float64)

    Nq = Q_hist.shape[0]
    Ng = G_hist.shape[0]
    D = np.zeros((Nq, Ng), dtype=np.float64)

    t0 = time.time()

    for i in range(Nq):
        if verbose:
            print(f"Query {i+1}/{Nq}")
        for j in range(Ng):
            D[i, j] = sinkhorn_cost_pot(
                Q_hist[i],
                G_hist[j],
                C,
                eps=eps,
                numItermax=numItermax,
                stopThr=stopThr,
            )

    runtime_sec = time.time() - t0
    return D, runtime_sec


def retrieval_topk_indices(D, k=5):
    """
    For each query, return indices of the top-k smallest distances.
    D: (Nq, Ng)
    Returns array of shape (Nq, k)
    """
    return np.argsort(D, axis=1)[:, :k]


def topk_same_label_stats(neigh_idx, query_labels, candidate_labels, k=5):
    query_labels = np.asarray(query_labels, dtype=int)
    candidate_labels = np.asarray(candidate_labels, dtype=int)

    top1_correct = []
    topk_prop = []

    for i in range(neigh_idx.shape[0]):
        retrieved_labels = candidate_labels[neigh_idx[i, :k]]
        top1_correct.append(int(retrieved_labels[0] == query_labels[i]))
        topk_prop.append(np.mean(retrieved_labels == query_labels[i]))

    return {
        "top1_accuracy": float(np.mean(top1_correct)),
        "topk_same_label_fraction": float(np.mean(topk_prop)),
    }


def show_retrieval_panel(
    query_idx,
    query_images,
    query_labels,
    query_indices,
    cand_images,
    cand_labels,
    cand_indices,
    nn_eucl,
    nn_sink,
    k=5,
):
    fig, axes = plt.subplots(3, k + 1, figsize=(2 * (k + 1), 6))

    # Query
    axes[0, 0].imshow(np.asarray(query_images[query_idx]), cmap="gray")
    axes[0, 0].set_title(f"Query\nlabel={query_labels[query_idx]}")
    axes[0, 0].axis("off")

    axes[1, 0].axis("off")
    axes[2, 0].axis("off")

    # Euclidean row
    for r, cand_id in enumerate(nn_eucl[query_idx, :k], start=1):
        axes[1, r].imshow(np.asarray(cand_images[cand_id]), cmap="gray")
        axes[1, r].set_title(f"E{r}\nlabel={cand_labels[cand_id]}")
        axes[1, r].axis("off")

    # Sinkhorn row
    for r, cand_id in enumerate(nn_sink[query_idx, :k], start=1):
        axes[2, r].imshow(np.asarray(cand_images[cand_id]), cmap="gray")
        axes[2, r].set_title(f"S{r}\nlabel={cand_labels[cand_id]}")
        axes[2, r].axis("off")

    # Turn off unused query-row cells
    for r in range(1, k + 1):
        axes[0, r].axis("off")

    axes[1, 0].text(0.5, 0.5, "Euclidean", ha="center", va="center", fontsize=12)
    axes[2, 0].text(0.5, 0.5, "Sinkhorn", ha="center", va="center", fontsize=12)

    plt.tight_layout()
    plt.show()
