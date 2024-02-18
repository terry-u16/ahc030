import datetime
import json
import math
import os
import random
import shutil
import subprocess

import optimize
import optuna
import predict

subprocess.run("cargo build --release")
shutil.move("../target/release/ahc030.exe", "./ahc030.exe")

SEED_PATH = "data/seed.txt"
INPUT_PATH = "data/in"
OPT_PATH = "data/opt"
os.environ["DURATION_MUL"] = "1.5"
os.environ["AHC030_SHOW_COMMENT"] = "0"

with open(SEED_PATH, "w") as f:
    for seed in range(0, 2000):
        f.write(f"{seed}\n")

for iteration in range(1, 1000):
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"ahc030-{start_time}-group-{iteration:000}"

    n = random.randint(10, 20)
    m = random.randint(2, n * n // 20)
    eps = random.randint(1, 20) / 100
    avg_min = max(n * n // 5 // m, 4)
    avg_max = n * n // 2 // m
    avg = random.randint(avg_min, avg_max)

    cmd = (
        f"./gen.exe {SEED_PATH} -d {INPUT_PATH} --N {n} --M {m} --eps {eps} --avg {avg}"
    )
    print(cmd)

    subprocess.run(cmd).check_returncode()

    objective = optimize.Objective()

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage="mysql+pymysql://default@localhost/optuna",
        load_if_exists=True,
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )

    (k, b, r, multi) = predict.predict(n, m, eps, avg, 100)

    print(f"suggested params: k={k}, b={b}, r={r}, multi={multi}")

    k = min(max(k, 0.5), 50.0)
    b = min(max(b, 0.0), 0.1)
    r = min(max(r, 0.1), 0.9)
    multi = 1 if multi >= 0.5 else 0

    study.enqueue_trial(
        {
            "multi": multi,
            "k": k,
            "b": b,
            "r": r,
        }
    )

    timeout = 450 if m >= 12 else 300

    study.optimize(objective, timeout=timeout)

    dictionary = {
        "study_name": study_name,
        "n": n,
        "m": m,
        "eps": eps,
        "avg": avg,
        "params": study.best_trial.params,
    }

    filename = (
        "optimized_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    with open(f"{OPT_PATH}/{filename}", "w") as f:
        json.dump(dictionary, f, indent=2)

    # optuna.visualization.plot_param_importances(study).show()
    # optuna.visualization.plot_contour(study).show()
