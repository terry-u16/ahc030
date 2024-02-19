import json
import math
import os

import numpy as np
import optuna
import pack


def gaussian_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
) -> float:
    return t1 * np.dot(x1, x2) + t2 * np.exp(-np.linalg.norm(x1 - x2) ** 2 / t3)


def calc_kernel_matrix(
    x1: np.ndarray,
    x2: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
) -> np.ndarray:
    n = x1.shape[0]
    m = x2.shape[0]
    k = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            k[i, j] = gaussian_kernel(x1[i, :], x2[j, :], t1, t2, t3)

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
    k = calc_kernel_matrix(x, x, t1, t2, t3) + t4 * np.eye(x.shape[0])
    kk = calc_kernel_matrix(x, xx, t1, t2, t3)
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
    k = calc_kernel_matrix(x, x, t1, t2, t3) + t4 * np.eye(x.shape[0])
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
        t2 = trial.suggest_float("t2", 0.01, 100.0, log=True)
        t3 = trial.suggest_float("t3", 0.01, 1.0, log=True)
        t4 = trial.suggest_float("t4", 0.01, 100.0, log=True)
        return calc_log_likelihood(self.x, self.y, t1, t2, t3, t4)


def load_data():
    x_list = []
    answer_list = []

    OPT_RESULT_DIR = "data/opt3"

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

            answer_list.append(math.sqrt(data["params"]["answer_threshold_ratio"]))

    x_matrix = np.array(x_list, dtype=np.float64)
    answer_array = np.array(answer_list, dtype=np.float64)

    return x_matrix, answer_array


def predict_one(
    x_matrix: np.ndarray, data_array: np.ndarray, new_x: np.ndarray, n_trials: int = 500
) -> tuple[float, float, float, float, float]:
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

    pred = predict_y(x_matrix, data_array, new_x, t1, t2, t3, t4)

    return pred, t1, t2, t3, t4


def predict(
    n: int, m: int, eps: float, avg: int, n_trials: int = 500
) -> tuple[float, float, float, float]:
    (x_matrix, answer_array) = load_data()

    new_x = np.array(
        [[(n - 10) / 10, math.sqrt(m) / 4, eps * 5, math.sqrt(avg) / 8]],
        dtype=np.float64,
    )

    pred_answer, _, _, _, _ = predict_one(x_matrix, answer_array, new_x, n_trials)
    pred_answer = pred_answer**2

    return pred_answer[0]


if __name__ == "__main__":
    (x_matrix, answer_array) = load_data()

    n = 10
    m = 2
    eps = 0.05
    avg = 20

    print(f"n={n}, m={m}, eps={eps}, avg={avg}")

    new_x = np.array(
        [[(n - 10) / 10, math.sqrt(m) / 4, eps * 5, math.sqrt(avg) / 8]],
        dtype=np.float64,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=== answer_threshold ===")
    pred_answer, t1, t2, t3, t4 = predict_one(x_matrix, answer_array, new_x)
    pred_answer = pred_answer**2
    param_answer = [t1, t2, t3, t4]
    print("pred_answer", pred_answer)

    PARAM_PATH = "data/params3.txt"

    with open(PARAM_PATH, "w") as f:
        params = {}

        params["len"] = len(x_matrix)

        n_vec = x_matrix[:, 0]
        m_vec = x_matrix[:, 1]
        eps_vec = x_matrix[:, 2]
        avg_vec = x_matrix[:, 3]

        f.write(f'const N3: &[u8] = b"{pack.pack_vec(n_vec)}";\n')
        f.write(f'const M3: &[u8] = b"{pack.pack_vec(m_vec)}";\n')
        f.write(f'const EPS3: &[u8] = b"{pack.pack_vec(eps_vec)}";\n')
        f.write(f'const AVG3: &[u8] = b"{pack.pack_vec(avg_vec)}";\n')

        f.write(f'const ANSWER: &[u8] = b"{pack.pack_vec(answer_array)}";\n')

        f.write(
            f'const PARAM_ANSWER: &[u8] = b"{pack.pack_vec(np.array(param_answer, dtype=np.float64))}";\n'
        )
