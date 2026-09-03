

import matplotlib.pyplot as plt
import numpy as np

# Load the comparison results and all Frechet results
comp = np.load("../Data/comp.npy")
result_frechet_all = np.load("../Data/result_frechet_all.npy", allow_pickle=True)
sigma = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # Example sigma values
nu = np.array([0.1, 0.3, 0.5, 0.7])  # Example nu values
for i in range(4):
    acc_kde = comp[6 * i : 6 * (i + 1), 0]
    acc_linear = comp[6 * i : 6 * (i + 1), 1]

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(
        sigma,
        acc_kde,
        label="KDE Acc",
        linestyle="--",
        linewidth=4,
        marker="*",
        markersize=16,
    )
    plt.plot(
        sigma,
        acc_linear,
        label="Linear Acc",
        linestyle="--",
        linewidth=4,
        marker="D",
        markersize=10,
    )
    plt.title(f"Comparison of Accuracies for $\\nu_1={nu[i]}$", fontsize=20)
    plt.xlabel("$\\sigma_1$", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.savefig(f"../figs/nu_{i + 1}_comparison.pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory
