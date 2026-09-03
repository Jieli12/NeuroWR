

# %% ##########################################################################
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import (
    frechet_regression,
    generate_normal_random_density_sample,
    kde_diff,
    predict_test_area,
)

pair_list = []
result_frechet_all = []
acc_linear_all = []
# %%
np.random.seed(42)  # For reproducibility
for mu_beta_3 in [0.1, 0.3, 0.5, 0.7]:
    for std_beta_3 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        print(f"mu_beta_3: {mu_beta_3}, std_beta_3: {std_beta_3}")
        pair_list.append((mu_beta_3, std_beta_3))
        # Generate samples for control and mTBI groups
        mu_beta_0 = 0
        mu_beta_1 = 0.1
        mu_beta_2 = 2
        mu_beta_4 = 1
        std_beta_0 = 0.5
        std_beta_1 = 0.1
        std_beta_2 = 0.5
        std_beta_4 = 0.5
        sample_size = 1000
        num_observations = 2000
        age_range = (18, 90)
        sample_ctrl = generate_normal_random_density_sample(
            mu_beta_0,
            mu_beta_1,
            mu_beta_2,
            std_beta_0,
            std_beta_1,
            std_beta_2,
            age_range=age_range,
            sample_size=sample_size,
            num_observations=num_observations,
        )
        sample_mtbi = generate_normal_random_density_sample(
            mu_beta_0,
            mu_beta_3,
            mu_beta_4,
            std_beta_0,
            std_beta_3,
            std_beta_4,
            age_range=age_range,
            sample_size=sample_size,
            num_observations=num_observations,
        )
        n = 1024
        result_ctrl = []
        for r in sample_ctrl["sample"]:
            bandwidth, density, cdf, lqd, quantile = kde_diff(r, n=n)
            res_temp = pd.DataFrame(
                {
                    "bandwidth": bandwidth,
                    "density": density,
                    "cdf": cdf,
                    "lqd": lqd,
                    "quantile": quantile,
                }
            )
            result_ctrl.append(res_temp)

        result_mtbi = []
        for r in sample_mtbi["sample"]:
            bandwidth, density, cdf, lqd, quantile = kde_diff(r, n=n)
            res_temp = pd.DataFrame(
                {
                    "bandwidth": bandwidth,
                    "density": density,
                    "cdf": cdf,
                    "lqd": lqd,
                    "quantile": quantile,
                }
            )
            result_mtbi.append(res_temp)

        quantile_matrix_ctrl = np.array([df["quantile"].values for df in result_ctrl])
        quantile_matrix_mtbi = np.array([df["quantile"].values for df in result_mtbi])
        cdf_matrix_ctrl = np.array([df["cdf"].values for df in result_ctrl])
        cdf_matrix_mtbi = np.array([df["cdf"].values for df in result_mtbi])
        age_gender_matrix_ctrl = np.column_stack(
            [sample_ctrl["age"], sample_ctrl["gender"]]
        )
        age_gender_matrix_mtbi = np.column_stack(
            [sample_mtbi["age"], sample_mtbi["gender"]]
        )
        grid_sup = np.linspace(0, 1, n)
        fig_name = "F1_Accuracy_plot.pdf"
        # split the data into training and test sets
        rate = 0.7
        n_train = int(rate * num_observations)
        quantile_matrix_mtbi_train = quantile_matrix_mtbi[:n_train, :]
        quantile_matrix_ctrl_train = quantile_matrix_ctrl[:n_train, :]
        cdf_matrix_mtbi_train = cdf_matrix_mtbi[:n_train, :]
        cdf_matrix_ctrl_train = cdf_matrix_ctrl[:n_train, :]
        age_gender_matrix_mtbi_train = age_gender_matrix_mtbi[:n_train, :]
        age_gender_matrix_ctrl_train = age_gender_matrix_ctrl[:n_train, :]
        quantile_matrix_mtbi_test = quantile_matrix_mtbi[n_train:, :]
        quantile_matrix_ctrl_test = quantile_matrix_ctrl[n_train:, :]
        cdf_matrix_mtbi_test = cdf_matrix_mtbi[n_train:, :]
        cdf_matrix_ctrl_test = cdf_matrix_ctrl[n_train:, :]
        age_gender_matrix_mtbi_test = age_gender_matrix_mtbi[n_train:, :]
        age_gender_matrix_ctrl_test = age_gender_matrix_ctrl[n_train:, :]
        result_frechet = frechet_regression(
            quantile_matrix_mtbi_train,
            cdf_matrix_mtbi_train,
            age_gender_matrix_mtbi_train,
            quantile_matrix_ctrl_train,
            cdf_matrix_ctrl_train,
            age_gender_matrix_ctrl_train,
            grid_sup,
            fig_name=fig_name,
        )
        # %%
        mtbi_lm = result_frechet["mtbi_lm"]
        normal_lm = result_frechet["normal_lm"]
        k_opt = result_frechet["k_opt"]
        data_test = {
            "selected_area": "test_area",
            "mtbi_lm": mtbi_lm,
            "normal_lm": normal_lm,
            "k_opt": k_opt,
            "grid_sup": grid_sup,
            "quantile_matrix_inn": quantile_matrix_mtbi_test,
            "cdf_matrix_inn": cdf_matrix_mtbi_test,
            "age_gender_matrix_inn": age_gender_matrix_mtbi_test,
            "quantile_matrix_cam": quantile_matrix_ctrl_test,
            "cdf_matrix_cam": cdf_matrix_ctrl_test,
            "age_gender_matrix_cam": age_gender_matrix_ctrl_test,
            "subject_id_inn": None,
            "subject_id_cam": None,
        }

        result_predict = predict_test_area(data_test)

        ave_ctrl = np.mean(sample_ctrl["sample"], axis=1)
        ave_mtbi = np.mean(sample_mtbi["sample"], axis=1)
        ave_ctrl_train = ave_ctrl[:n_train]
        ave_mtbi_train = ave_mtbi[:n_train]
        ave_ctrl_test = ave_ctrl[n_train:]
        ave_mtbi_test = ave_mtbi[n_train:]
        # build two linear models
        age_ctrl = sample_ctrl["age"]
        age_mtbi = sample_mtbi["age"]
        age_ctrl_train = age_ctrl[:n_train]
        age_mtbi_train = age_mtbi[:n_train]
        age_ctrl_test = age_ctrl[n_train:]
        age_mtbi_test = age_mtbi[n_train:]
        gender_ctrl = sample_ctrl["gender"]
        gender_mtbi = sample_mtbi["gender"]
        gender_ctrl_train = gender_ctrl[:n_train]
        gender_mtbi_train = gender_mtbi[:n_train]
        gender_ctrl_test = gender_ctrl[n_train:]
        gender_mtbi_test = gender_mtbi[n_train:]

        # Build linear model for control group
        X_ctrl_train = np.column_stack([age_ctrl_train, gender_ctrl_train])
        X_ctrl_train = sm.add_constant(X_ctrl_train)  # Add intercept
        model_ctrl = sm.OLS(ave_ctrl_train, X_ctrl_train).fit()

        # Build linear model for mTBI group
        X_mtbi_train = np.column_stack([age_mtbi_train, gender_mtbi_train])
        X_mtbi_train = sm.add_constant(X_mtbi_train)  # Add intercept
        model_mtbi = sm.OLS(ave_mtbi_train, X_mtbi_train).fit()

        # Print model summaries
        print("Control Group Model Summary:")
        print(model_ctrl.summary())
        print("\nmTBI Group Model Summary:")
        print(model_mtbi.summary())
        # Predict on test data
        X_ctrl_test = np.column_stack([age_ctrl_test, gender_ctrl_test])
        X_ctrl_test = sm.add_constant(X_ctrl_test)  # Add intercept
        predictions_ctrl = model_ctrl.predict(X_ctrl_test)

        X_mtbi_test = np.column_stack([age_mtbi_test, gender_mtbi_test])
        X_mtbi_test = sm.add_constant(X_mtbi_test)  # Add intercept
        predictions_mtbi = model_mtbi.predict(X_mtbi_test)

        predictions_ctrl_mtbi = model_ctrl.predict(X_mtbi_test)
        predictions_mtbi_ctrl = model_mtbi.predict(X_ctrl_test)

        true_control_pred = np.abs(predictions_ctrl - ave_ctrl_test) <= np.abs(
            predictions_mtbi_ctrl - ave_ctrl_test
        )

        true_mtbi_pred = np.abs(predictions_mtbi - ave_mtbi_test) <= np.abs(
            predictions_ctrl_mtbi - ave_mtbi_test
        )

        # Calculate accuracy
        accuracy = (np.sum(true_control_pred) + np.sum(true_mtbi_pred)) / (
            len(true_control_pred) + len(true_mtbi_pred)
        )
        print(f"Accuracy of the linear models: {accuracy:.2f}")
        print(result_predict["accuracy"])
        result_frechet_all.append(result_predict)
        acc_linear_all.append(accuracy)
# %% Save results
acc_kde_all = [result["accuracy"] for result in result_frechet_all]
comp = np.column_stack([acc_kde_all, acc_linear_all])
# Save the comparison results and all Frechet results
np.save("comp.npy", comp)
np.save("result_frechet_all.npy", result_frechet_all)
