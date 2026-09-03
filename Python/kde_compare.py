

# %% ##########################################################################
# pylint: disable= redefined-outer-name,
import matplotlib.pyplot as plt
import numpy as np
from diffKDE import diffKDE  # pyright: ignore[reportMissingImports]
from scipy.interpolate import interp1d
from scipy.special import erf
from utils import kde_diff


# Define the density function f(x)
def f(x):
    """This function comes from Eq.(39) of the paper 'A diffusion-based kernel density estimator (diffKDE, version 1) with  optimal bandwidth approximation for the analysis of data in  geoscience and ecological research' by Pelz et al. 2023."""
    return (
        0.3 * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - 6) ** 2)
        + 0.6 * (1 / (0.7 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 9.5) / 0.7) ** 2)
        + 0.1 * (1 / (0.5 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 12) / 0.5) ** 2)
    )


# %%
def f_cdf_analytical(x):
    """Compute the CDF of f(x) analytically."""
    term1 = 0.3 * 0.5 * (1 + erf((x - 3) / np.sqrt(2)))
    term2 = 0.6 * 0.5 * (1 + erf((x - 6.5) / (0.7 * np.sqrt(2))))
    term3 = 0.1 * 0.5 * (1 + erf((x - 9) / (0.5 * np.sqrt(2))))
    return term1 + term2 + term3


# %%
# Compute the CDF of f(x) over a range of x values
n_all = [50, 100, 200, 400]
tvd1_all = []
tvd2_all = []
for n in n_all:
    print(f"n = {n}")
    np.random.seed(42)  # For reproducibility
    x_vals = np.linspace(0, 20, n)  # Define the range of x
    pdf_vals = f(x_vals)  # Evaluate the PDF
    cdf_vals = np.cumsum(pdf_vals) * (x_vals[1] - x_vals[0])  # Compute the CDF
    cdf_vals /= cdf_vals[-1]  # Normalize the CDF to range [0, 1]

    # Create an interpolation function for the inverse CDF
    inverse_cdf = interp1d(
        cdf_vals, x_vals, bounds_error=False, fill_value="extrapolate"
    )

    # Generate random variables using inverse transform sampling
    num_samples = n
    uniform_randoms = np.random.uniform(
        0, 1, num_samples
    )  # Generate uniform random variables
    samples = inverse_cdf(uniform_randoms)  # Map uniform randoms to samples from f(x)

    plt.hist(samples, bins=50, density=True, alpha=0.5, label="Generated Samples")
    plt.plot(x_vals, pdf_vals, label="True PDF", color="red")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title(f"Random Variables Generated from f(x) with n={n}")
    plt.show()

    tt = diffKDE.KDE(samples, n=1023)

    bandwidth, density, cdf, lqd, mesh = kde_diff(samples, n=1024)

    x = np.linspace(0, 20, 1024)
    f_den = f(x)
    cdf_f = f_cdf_analytical(x)  # CDF of f(x)
    # Interpolate tt[0] (density values) to the x grid using tt[1] (coordinates)
    interp_func = interp1d(tt[1], tt[0], bounds_error=False, fill_value=0)
    g = interp_func(x)
    cdf_g = np.cumsum(g) * (x[1] - x[0])  # CDF of g(x)
    cdf_g /= cdf_g[-1]

    interp_func = interp1d(
        np.array(mesh), np.array(density), bounds_error=False, fill_value=0
    )
    g_my = interp_func(x)
    cdf_g_my = np.cumsum(g_my) * (x[1] - x[0])  # CDF of g(x)
    cdf_g_my /= cdf_g_my[-1]

    # %
    plt.figure(figsize=(10, 6))
    plt.plot(x, g, label="diffKDE", color="blue", lw=4)  # diffKDE density estimate
    plt.plot(x, g_my, label="Our diffKDE", color="orange", lw=4)
    plt.plot(
        x_vals, pdf_vals, label="True PDF", color="red", lw=4
    )  # True density function
    plt.legend(fontsize=16)
    plt.xlabel("x", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.title("Density Estimates", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.savefig(f"../figs/kde_compare_n{n}.pdf", dpi=600)
    plt.show()

    # % Compute total variation distance
    tvd1 = 0.5 * np.trapezoid(np.abs(f_den - g), x=x)
    tvd1_all.append(tvd1)
    tvd2 = 0.5 * np.trapezoid(np.abs(f_den - g_my), x=x)
    tvd2_all.append(tvd2)
    print(f"Wasserstein distance (diffKDE): {tvd1}")
    print(f"Wasserstein distance (Our diffKDE): {tvd2}")


# %%
print(tvd1_all)
print(tvd2_all)
