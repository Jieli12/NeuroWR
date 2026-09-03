

import numpy as np
import statsmodels.api as sm
from scipy.fftpack import dct, idct
from scipy.interpolate import interp1d
from scipy.optimize import brentq, fminbound
from scipy.stats import gaussian_kde


def truncate_repeats(cdf, lqd_temp):
    if len(cdf) != len(lqd_temp):
        raise ValueError("cdf and lqd_temp must have the same length")
    seen = set()
    new_cdf = []
    new_lqd_temp = []
    for i, value in enumerate(cdf):
        if value not in seen:
            seen.add(value)
            new_cdf.append(value)
            new_lqd_temp.append(lqd_temp[i])
    return np.array(new_cdf), np.array(new_lqd_temp)


def kde_diff(data, n=None, MIN=None, MAX=None, threshold=1e-32):
    data = np.array(data)
    data = data[np.isfinite(data)]
    n = 2**14 if n is None else int(2 ** np.ceil(np.log2(n)))
    minimum = min(data)
    maximum = max(data)
    Range = maximum - minimum
    MIN = minimum - Range / 10 if MIN is None else MIN
    MAX = maximum + Range / 10 if MAX is None else MAX
    R = MAX - MIN
    N = len(np.unique(data))
    initial_data, bins = np.histogram(data, bins=n, range=(MIN, MAX))
    initial_data = initial_data / N
    initial_data = initial_data / np.sum(initial_data)
    DCTData = dct(initial_data, norm=None)
    I = np.array([i * i for i in range(1, n)])
    SqDCTData = (DCTData[1:] / 2) ** 2

    def fixed_point(t, N, I, a2):
        l = 7
        I = np.longdouble(I)
        N = np.longdouble(N)
        a2 = np.longdouble(a2)
        f = 2 * np.pi ** (2 * l) * np.sum(I**l * a2 * np.exp(-I * np.pi**2 * t))
        for s in range(l - 1, 1, -1):
            K0 = np.prod(range(1, 2 * s, 2)) / np.sqrt(2 * np.pi)
            const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
            time = (2 * const * K0 / N / f) ** (2 / (3 + 2 * s))
            f = 2 * np.pi ** (2 * s) * np.sum(I**s * a2 * np.exp(-I * np.pi**2 * time))
        return t - (2 * N * np.sqrt(np.pi) * f) ** (-2 / 5)

    def root(f, N, *args):
        N = 50 if N <= 50 else (1050 if N >= 1050 else N)
        tol = 10**-12 + 0.01 * (N - 50) / 1000
        flag = 0
        while flag == 0:
            try:
                t = brentq(lambda x: f(x, N, *args), 0, tol)
                flag = 1
            except ValueError:
                tol = min(tol * 2, 0.1)
            if tol == 0.1:
                t = fminbound(lambda x: abs(f(x, N, *args)), 0, 0.1)
                flag = 1
        return t

    try:
        t_star = root(fixed_point, N, I, SqDCTData)
    except ValueError:
        kde = gaussian_kde(data)
        bandwidth = kde.factor * np.std(data)
        mesh = [(bins[i] + bins[i + 1]) / 2 for i in range(n)]
        density = kde.evaluate(mesh)
        density[density <= 0] = threshold
        cdf = np.cumsum(density) * (mesh[1] - mesh[0])
        cdf /= cdf[-1]
        cdf = np.sort(cdf)
        mask = np.abs(density) >= threshold
        lqd_temp = -np.log(density[mask])
        cdf_temp, lqd_temp = truncate_repeats(cdf[mask], lqd_temp)
        spline = interp1d(cdf_temp, lqd_temp, kind="cubic", fill_value="extrapolate")
        lqd = spline(mesh)
        return bandwidth, density, cdf, lqd, mesh
    SmDCTData = DCTData * np.exp(-np.arange(n) ** 2 * np.pi**2 * t_star / 2)
    density = idct(SmDCTData, norm=None) * n / R
    density[density <= 0] = threshold
    mesh = [(bins[i] + bins[i + 1]) / 2 for i in range(n)]
    bandwidth = np.sqrt(t_star) * R
    density = density / np.trapz(density, mesh)
    f = 2 * np.pi**2 * np.sum(I * SqDCTData * np.exp(-I * np.pi**2 * t_star))
    t_cdf = (np.sqrt(np.pi) * f * N) ** (-2 / 3)
    a_cdf = DCTData * np.exp(-np.arange(n) ** 2 * np.pi**2 * t_cdf / 2)
    cdf = np.cumsum(idct(a_cdf, norm=None)) / (n - 1) / 2
    cdf /= cdf[-1]
    cdf = np.sort(cdf)
    mask = np.abs(density) >= threshold
    lqd_temp = -np.log(density[mask])
    cdf_temp, lqd_temp = truncate_repeats(cdf[mask], lqd_temp)
    spline = interp1d(cdf_temp, lqd_temp, kind="cubic", fill_value="extrapolate")
    lqd = spline(mesh)
    return bandwidth, density, cdf, lqd, mesh


def quadratic_Q(Qmat, t, positive_support=False):
    Qmat = np.array(Qmat)
    t = np.array(t)
    Qmat_diff = np.diff(Qmat, axis=1)
    if positive_support:
        condition = np.min(Qmat_diff) < 0 or np.min(Qmat) < 0
    else:
        condition = np.min(Qmat_diff) < 0
    if condition:
        refit_index = np.where(np.min(Qmat_diff, axis=1) < 0)[0]
        min_index = np.where(np.min(Qmat, axis=1) < 0)[0]
        union_index = np.union1d(refit_index, min_index)
        for i in union_index:
            qmat_temp = Qmat[i, :]
            n_power = np.floor(np.log10(np.max(np.abs(qmat_temp))))
            qmat_temp = qmat_temp / 10**n_power
            # 线性插值替代优化
            Qrefit = np.maximum.accumulate(qmat_temp)
            if positive_support:
                Qrefit[0] = max(Qrefit[0], 0)
            Qmat[i, :] = Qrefit * 10**n_power
    return Qmat


def wri_predict(Qmat_lm, x_pred, t, positive_support=False):
    x_pred = sm.add_constant(x_pred, has_constant="add")
    Qpred = Qmat_lm.predict(x_pred)
    Qpred = quadratic_Q(Qpred, t, positive_support=positive_support)
    return Qpred


def quantile_function(x_prob_grid, x, y, positive_support=False):
    y_unique, idx = np.unique(y, return_index=True)
    x_unique = x[idx]
    inv_cdf = interp1d(y_unique, x_unique, bounds_error=False, fill_value="extrapolate")
    quantiles = inv_cdf(x_prob_grid)
    quantiles = quadratic_Q(
        np.reshape(quantiles, (1, -1)), x_prob_grid, positive_support=positive_support
    )
    return quantiles


def frechet_regression(
    quantile_mtbi,
    cdf_mtbi,
    age_gender_mtbi,
    quantile_normal,
    cdf_normal,
    age_gender_normal,
    grid_sup,
):
    quantile_normal_align = np.zeros_like(quantile_normal)
    for i, (quant, cdf) in enumerate(zip(quantile_normal, cdf_normal)):
        quantile_normal_align[i, :] = quantile_function(
            grid_sup, quant, cdf, positive_support=True
        )
    quantile_mtbi_align = np.zeros_like(quantile_mtbi)
    for i, (quant, cdf) in enumerate(zip(quantile_mtbi, cdf_mtbi)):
        quantile_mtbi_align[i, :] = quantile_function(
            grid_sup, quant, cdf, positive_support=True
        )

    def wri_regress(Qmat, x, t, positive_support=False):
        x = sm.add_constant(x)
        Qmat_lm = sm.OLS(Qmat, x).fit()
        Qfit = Qmat_lm.fittedvalues
        Qfit = quadratic_Q(Qfit, t, positive_support=positive_support)
        return Qmat_lm, Qfit

    mtbi_lm, mtbi_fit = wri_regress(
        quantile_mtbi_align, age_gender_mtbi, grid_sup, positive_support=True
    )
    normal_lm, normal_fit = wri_regress(
        quantile_normal_align, age_gender_normal, grid_sup, positive_support=True
    )
    mtbi_fit_normal_data = wri_predict(
        mtbi_lm, age_gender_normal, grid_sup, positive_support=True
    )
    normal_fit_mtbi_data = wri_predict(
        normal_lm, age_gender_mtbi, grid_sup, positive_support=True
    )
    dmtbi_2mtbi = np.sum(np.abs(mtbi_fit - quantile_mtbi_align), axis=1) * (
        grid_sup[1] - grid_sup[0]
    )
    dmtbi_2normal = np.sum(
        np.abs(quantile_mtbi_align - normal_fit_mtbi_data), axis=1
    ) * (grid_sup[1] - grid_sup[0])
    dnormal_2normal = np.sum(np.abs(normal_fit - quantile_normal_align), axis=1) * (
        grid_sup[1] - grid_sup[0]
    )
    dnormal_2mtbi = np.sum(
        np.abs(quantile_normal_align - mtbi_fit_normal_data), axis=1
    ) * (grid_sup[1] - grid_sup[0])
    k_can = np.arange(0.001, 5, 0.001)
    f1 = []
    acc = []
    num_normal = age_gender_normal.shape[0]
    num_mtbi = age_gender_mtbi.shape[0]
    for k in k_can:
        tn = np.sum(dnormal_2normal < k * dnormal_2mtbi)
        fp = num_normal - tn
        tp = np.sum(k * dmtbi_2mtbi < dmtbi_2normal)
        fn = num_mtbi - tp
        f1_score = 2 * tp / (2 * tp + fp + fn)
        accuracy = (tn + tp) / (num_mtbi + num_normal)
        f1.append(f1_score)
        acc.append(accuracy)
    f1 = np.array(f1)
    acc = np.array(acc)
    max_f1_index = np.argmax(f1)
    max_f1 = f1[max_f1_index]
    corresponding_acc = acc[max_f1_index]
    k_opt = k_can[max_f1_index]
    normal_pred_class = dnormal_2normal >= k_opt * dnormal_2mtbi
    mtbi_pred_class = k_opt * dmtbi_2mtbi < dmtbi_2normal
    return {
        "quantile_normal": quantile_normal_align,
        "quantile_mtbi": quantile_mtbi_align,
        "mtbi_lm": mtbi_lm,
        "mtbi_fit": mtbi_fit,
        "normal_lm": normal_lm,
        "normal_fit": normal_fit,
        "dmtbi_2mtbi": dmtbi_2mtbi,
        "dmtbi_2normal": dmtbi_2normal,
        "dnormal_2normal": dnormal_2normal,
        "dnormal_2mtbi": dnormal_2mtbi,
        "k_opt": k_opt,
        "normal_pred_class": normal_pred_class,
        "mtbi_pred_class": mtbi_pred_class,
        "max_f1": max_f1,
        "corresponding_acc": corresponding_acc,
    }


def predict_test_area(data):
    selected_area = data["selected_area"]
    quantile_matrix_inn = data["quantile_matrix_inn"]
    cdf_matrix_inn = data["cdf_matrix_inn"]
    age_gender_matrix_inn = data["age_gender_matrix_inn"]
    subject_id_inn = data["subject_id_inn"]
    quantile_matrix_cam = data["quantile_matrix_cam"]
    cdf_matrix_cam = data["cdf_matrix_cam"]
    age_gender_matrix_cam = data["age_gender_matrix_cam"]
    subject_id_cam = data["subject_id_cam"]
    grid_sup = data["grid_sup"]
    mtbi_lm = data["mtbi_lm"]
    normal_lm = data["normal_lm"]
    k_opt = data["k_opt"]
    quantile_normal_align = np.zeros_like(quantile_matrix_cam)
    for i, (quant, cdf) in enumerate(zip(quantile_matrix_cam, cdf_matrix_cam)):
        quantile_normal_align[i, :] = quantile_function(
            grid_sup, quant, cdf, positive_support=True
        )
    quantile_mtbi_align = np.zeros_like(quantile_matrix_inn)
    for i, (quant, cdf) in enumerate(zip(quantile_matrix_inn, cdf_matrix_inn)):
        quantile_mtbi_align[i, :] = quantile_function(
            grid_sup, quant, cdf, positive_support=True
        )
    mtbi_fit_mtbi_data = wri_predict(
        mtbi_lm, age_gender_matrix_inn, grid_sup, positive_support=True
    )
    mtbi_fit_normal_data = wri_predict(
        mtbi_lm, age_gender_matrix_cam, grid_sup, positive_support=True
    )
    normal_fit_normal_data = wri_predict(
        normal_lm, age_gender_matrix_cam, grid_sup, positive_support=True
    )
    dmtbi_2mtbi = np.sum(np.abs(mtbi_fit_mtbi_data - quantile_mtbi_align), axis=1) * (
        grid_sup[1] - grid_sup[0]
    )
    dmtbi_2normal = np.sum(
        np.abs(quantile_mtbi_align - normal_fit_normal_data), axis=1
    ) * (grid_sup[1] - grid_sup[0])
    dnormal_2normal = np.sum(
        np.abs(normal_fit_normal_data - quantile_normal_align), axis=1
    ) * (grid_sup[1] - grid_sup[0])
    dnormal_2mtbi = np.sum(
        np.abs(quantile_normal_align - mtbi_fit_normal_data), axis=1
    ) * (grid_sup[1] - grid_sup[0])
    normal_pred_class = dnormal_2normal >= k_opt * dnormal_2mtbi
    mtbi_pred_class = k_opt * dmtbi_2mtbi < dmtbi_2normal
    num_normal = len(normal_pred_class)
    num_mtbi = len(mtbi_pred_class)
    fp = np.sum(normal_pred_class)
    tn = num_normal - fp
    tp = np.sum(mtbi_pred_class)
    fn = num_mtbi - tp
    tpr = tp / num_mtbi
    tnr = tn / num_normal
    f1_score = 2 * tp / (2 * tp + fp + fn)
    accuracy = (tn + tp) / (num_mtbi + num_normal)
    return {
        "quantile_normal": quantile_normal_align,
        "quantile_mtbi": quantile_mtbi_align,
        "mtbi_fit": mtbi_fit_mtbi_data,
        "normal_fit": normal_fit_normal_data,
        "dmtbi_2mtbi": dmtbi_2mtbi,
        "dmtbi_2normal": dmtbi_2normal,
        "dnormal_2normal": dnormal_2normal,
        "dnormal_2mtbi": dnormal_2mtbi,
        "k_opt": k_opt,
        "normal_pred_class": normal_pred_class,
        "mtbi_pred_class": mtbi_pred_class,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "tpr": tpr,
        "tnr": tnr,
        "num_mtbi": num_mtbi,
        "num_normal": num_normal,
        "subject_id_cam": subject_id_cam,
        "subject_id_inn": subject_id_inn,
        "selected_area": selected_area,
    }


def generate_normal_random_density_sample(
    mu_beta_0,
    mu_beta_1,
    mu_beta_2,
    std_beta_0,
    std_beta_1,
    std_beta_2,
    age_range,
    sample_size=1000,
    num_observations=500,
):
    beta_0 = np.random.normal(mu_beta_0, std_beta_0, size=num_observations)
    beta_1 = np.random.normal(mu_beta_1, std_beta_1, size=num_observations)
    beta_2 = np.random.normal(mu_beta_2, std_beta_2, size=num_observations)
    age = np.random.randint(age_range[0], age_range[1], size=num_observations)
    gender = np.random.randint(0, 2, size=num_observations)
    mean_model = mu_beta_0 + age * mu_beta_1 + gender * mu_beta_2
    var_model = std_beta_0**2 + (age**2) * std_beta_1**2 + (gender**2) * std_beta_2**2
    mu = beta_0 + beta_1 * age + beta_2 * gender
    std = np.sqrt(var_model)
    sample = np.array(
        [np.random.normal(mu[i], std[i], sample_size) for i in range(len(mu))]
    )
    return {
        "age": age,
        "gender": gender,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "mean_model": mean_model,
        "mu": mu,
        "std": std,
        "sample": sample,
    }
