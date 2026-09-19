# Self-contained demo: simulate two omics blocks with a shared latent factor and recover it.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from integrate import joint_factorisation, variance_explained


def simulate(n_samples=80, seed=42):
    rng = np.random.default_rng(seed)
    # A true shared biological factor: two groups of samples.
    group = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    shared = (group - 0.5).reshape(-1, 1) * 2.0
    shared = shared + 0.4 * rng.standard_normal((n_samples, 1))

    # Block 1 (e.g. RNA): 50 features driven by shared factor plus private noise.
    load1 = rng.standard_normal((1, 50))
    block1 = shared @ load1 + 0.5 * rng.standard_normal((n_samples, 50))

    # Block 2 (e.g. methylation): 30 features, different scale, same shared factor.
    load2 = rng.standard_normal((1, 30)) * 5.0
    block2 = shared @ load2 + 0.5 * rng.standard_normal((n_samples, 30)) * 5.0

    return block1, block2, group


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    block1, block2, group = simulate()
    Z, Ws, std_blocks = joint_factorisation([block1, block2], n_factors=2, seed=0)
    ve = variance_explained(std_blocks, Z, Ws)

    # Save a summary table.
    summary = pd.DataFrame(
        {
            "block": ["omics_1_rna", "omics_2_methyl"],
            "n_features": [block1.shape[1], block2.shape[1]],
            "variance_explained_shared": [round(v, 3) for v in ve],
        }
    )
    summary.to_csv("results/summary.csv", index=False)

    # Plot the shared factor space, coloured by the true group.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for g, col in zip([0, 1], ["#1f77b4", "#d62728"]):
        mask = group == g
        ax[0].scatter(Z[mask, 0], Z[mask, 1], c=col, label="group " + str(g), alpha=0.8)
    ax[0].set_xlabel("Shared factor 1")
    ax[0].set_ylabel("Shared factor 2")
    ax[0].set_title("Recovered shared factor space")
    ax[0].legend()

    ax[1].bar(["RNA", "Methyl"], ve, color=["#2ca02c", "#9467bd"])
    ax[1].set_ylabel("Variance explained by shared factors")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("How much each block shares")

    fig.tight_layout()
    fig.savefig("figures/demo.png", dpi=120)
    print("Wrote figures/demo.png and results/summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
