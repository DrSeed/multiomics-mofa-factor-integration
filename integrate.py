# Reusable joint matrix factorisation for multiomics blocks measured on shared samples.
import numpy as np


def standardise_block(X):
    # Centre and scale each feature so no single block dominates the shared factors.
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def joint_factorisation(blocks, n_factors=2, n_iter=200, seed=0):
    # blocks: list of arrays each shaped (n_samples, n_features_k), same sample order.
    # Returns shared factor matrix Z (n_samples, n_factors) and per-block loadings.
    rng = np.random.default_rng(seed)
    blocks = [standardise_block(np.asarray(b, dtype=float)) for b in blocks]
    n_samples = blocks[0].shape[0]
    for b in blocks:
        if b.shape[0] != n_samples:
            raise ValueError("All blocks must share the same samples (rows).")

    Z = rng.standard_normal((n_samples, n_factors))
    Ws = [rng.standard_normal((b.shape[1], n_factors)) for b in blocks]

    for _ in range(n_iter):
        # Update loadings given Z (least squares per block).
        for k, b in enumerate(blocks):
            ZtZ = Z.T @ Z + 1e-6 * np.eye(n_factors)
            Ws[k] = (b.T @ Z) @ np.linalg.inv(ZtZ)
        # Update shared Z given all loadings (stack the problem).
        A = np.zeros((n_factors, n_factors))
        Bmat = np.zeros((n_samples, n_factors))
        for k, b in enumerate(blocks):
            A += Ws[k].T @ Ws[k]
            Bmat += b @ Ws[k]
        A += 1e-6 * np.eye(n_factors)
        Z = Bmat @ np.linalg.inv(A)

    return Z, Ws, blocks


def variance_explained(blocks, Z, Ws):
    # Fraction of each block's variance captured by the shared factors.
    out = []
    for b, W in zip(blocks, Ws):
        recon = Z @ W.T
        ss_res = np.sum((b - recon) ** 2)
        ss_tot = np.sum(b ** 2)
        out.append(1.0 - ss_res / ss_tot)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    z = rng.standard_normal((60, 1))
    x1 = z @ rng.standard_normal((1, 40)) + 0.3 * rng.standard_normal((60, 40))
    x2 = z @ rng.standard_normal((1, 25)) + 0.3 * rng.standard_normal((60, 25))
    Z, Ws, std_blocks = joint_factorisation([x1, x2], n_factors=2)
    print("Variance explained per block:", variance_explained(std_blocks, Z, Ws))
