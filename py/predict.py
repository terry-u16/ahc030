import json
import math
import os

import numpy as np
import optuna


def gaussian_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    t4: float,
) -> float:
    return (
        t1 * np.dot(x1, x2) + t2 * np.exp(-np.linalg.norm(x1 - x2) ** 2 / t3) + t4
    )


def calc_kernel_matrix(
    x1: np.ndarray,
    x2: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    t4: float,
) -> np.ndarray:
    n = x1.shape[0]
    m = x2.shape[0]
    k = np.zeros((n, m))

    for i in range(n):
        for j in range(i, m):
            delta = 1 if i == j else 0
            v = gaussian_kernel(x1[i, :], x2[j, :], t1, t2, t3, t4 * delta)
            k[i, j] = v
            k[j, i] = v

    return k


def predict_y(
    x: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    t4: float,
) -> float:
    y_mean = np.mean(y)
    k = calc_kernel_matrix(x, x, t1, t2, t3, t4)
    kk = calc_kernel_matrix(x, xx, t1, t2, t3, t4)
    yy = kk.transpose() @ np.linalg.solve(k, y - y_mean)
    return yy + y_mean


def calc_log_likelihood(
    x: np.ndarray,
    y: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    t4: float,
) -> float:
    y_mean = np.mean(y)
    y = y - y_mean
    k = calc_kernel_matrix(x, x, t1, t2, t3, t4)
    yy = y.transpose() @ np.linalg.solve(k, y)
    return -np.log(np.linalg.det(k)) - yy


class Objective:
    x: np.ndarray
    y: np.ndarray

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def __call__(self, trial: optuna.trial.Trial) -> float:
        t1 = trial.suggest_float("t1", 0.01, 10.0, log=True)
        t2 = trial.suggest_float("t2", 0.01, 10.0, log=True)
        t3 = trial.suggest_float("t3", 0.01, 1.0, log=True)
        t4 = trial.suggest_float("t4", 0.01, 10.0, log=True)
        return calc_log_likelihood(self.x, self.y, t1, t2, t3, t4)


def load_data():
    x_list = []
    k_list = []
    b_list = []
    r_list = []
    multi_list = []

    OPT_RESULT_DIR = "data/opt"

    files = os.listdir(OPT_RESULT_DIR)

    for file in files:
        if not file.endswith(".json"):
            continue

        with open(f"{OPT_RESULT_DIR}/{file}", "r") as f:
            data = json.load(f)
            x = []
            x.append((data["n"] - 10) / 10)
            x.append(math.sqrt(data["m"]) / 4)
            x.append(data["eps"] * 5)
            x.append(math.sqrt(data["avg"]) / 8)
            x_list.append(x)

            k_list.append(math.sqrt(data["params"]["k"]))
            b_list.append(data["params"]["b"])
            r_list.append(data["params"]["r"])
            multi_list.append(data["params"]["multi"])

    x_matrix = np.array(x_list, dtype=np.float64)
    k_array = np.array(k_list, dtype=np.float64)
    b_array = np.array(b_list, dtype=np.float64)
    r_array = np.array(r_list, dtype=np.float64)
    multi_array = np.array(multi_list, dtype=np.float64)

    return x_matrix, k_array, b_array, r_array, multi_array


def predict_one(
    x_matrix: np.ndarray, data_array: np.ndarray, new_x: np.ndarray, n_trials: int = 500
) -> float:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
    )
    objective = Objective(x_matrix, data_array)
    study.optimize(objective, n_trials=n_trials)

    print("param", study.best_trial.params)

    t1 = study.best_trial.params["t1"]
    t2 = study.best_trial.params["t2"]
    t3 = study.best_trial.params["t3"]
    t4 = study.best_trial.params["t4"]

    optuna.logging.set_verbosity(optuna.logging.INFO)

    pred = predict_y(x_matrix, data_array, new_x,  t1, t2, t3, t4)

    return pred


def predict(
    n: int, m: int, eps: float, avg: int, n_trials: int = 500
) -> tuple[float, float, float, float]:
    (x_matrix, k_array, b_array, r_array, multi_array) = load_data()

    new_x = np.array(
        [[(n - 10) / 10, math.sqrt(m) / 4, eps * 5, math.sqrt(avg) / 8]],
        dtype=np.float64,
    )

    pred_k = predict_one(x_matrix, k_array, new_x, n_trials)
    pred_k = pred_k**2
    pred_b = predict_one(x_matrix, b_array, new_x, n_trials)
    pred_r = predict_one(x_matrix, r_array, new_x, n_trials)
    pred_multi = predict_one(x_matrix, multi_array, new_x, n_trials)

    return pred_k[0], pred_b[0], pred_r[0], pred_multi[0]


if __name__ == "__main__":
    (x_matrix, k_array, b_array, r_array, multi_array) = load_data()

    new_x = np.array(
        [[(10 - 10) / 10, math.sqrt(2) / 4, 0.01 * 5, math.sqrt(25) / 8]],
        dtype=np.float64,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
    )
    objective = Objective(x_matrix, k_array)
    study.optimize(objective, n_trials=500)

    print("param_k", study.best_trial.params)

    t1 = study.best_trial.params["t1"]
    t2 = study.best_trial.params["t2"]
    t3 = study.best_trial.params["t3"]
    t4 = study.best_trial.params["t4"]

    pred_k = predict_y(x_matrix, k_array, new_x, t1, t2, t3, t4)
    pred_k = pred_k**2
    print("pred_k", pred_k)

    study = optuna.create_study(
        direction="maximize",
    )
    objective = Objective(x_matrix, b_array)
    study.optimize(objective, n_trials=500)

    print("param_b", study.best_trial.params)

    t1 = study.best_trial.params["t1"]
    t2 = study.best_trial.params["t2"]
    t3 = study.best_trial.params["t3"]
    t4 = study.best_trial.params["t4"]

    pred_b = predict_y(x_matrix, b_array, new_x, t1, t2, t3, t4)
    print("pred_b", pred_b)

    study = optuna.create_study(
        direction="maximize",
    )
    objective = Objective(x_matrix, r_array)
    study.optimize(objective, n_trials=500)

    print("param_r", study.best_trial.params)

    t1 = study.best_trial.params["t1"]
    t2 = study.best_trial.params["t2"]
    t3 = study.best_trial.params["t3"]
    t4 = study.best_trial.params["t4"]

    pred_r = predict_y(x_matrix, r_array, new_x, t1, t2, t3, t4)
    print("pred_r", pred_r)

    study = optuna.create_study(
        direction="maximize",
    )
    objective = Objective(x_matrix, multi_array)
    study.optimize(objective, n_trials=500)

    print("param_multi", study.best_trial.params)

    t1 = study.best_trial.params["t1"]
    t2 = study.best_trial.params["t2"]
    t3 = study.best_trial.params["t3"]
    t4 = study.best_trial.params["t4"]

    pred_multi = predict_y(x_matrix, multi_array, new_x, t1, t2, t3, t4)
    print("pred_multi", pred_multi)
