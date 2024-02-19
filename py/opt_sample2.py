import datetime
import json
import math
import os
import random
import shutil
import subprocess

import optimize2
import optuna
import predict

subprocess.run("cargo build --release")
shutil.move("../target/release/ahc030.exe", "./ahc030.exe")

SEED_PATH = "data/seed.txt"
INPUT_PATH = "data/in"
OPT_PATH = "data/opt2"
os.environ["DURATION_MUL"] = "1.5"
os.environ["AHC030_SHOW_COMMENT"] = "0"

with open(SEED_PATH, "w") as f:
    for seed in range(0, 2000):
        f.write(f"{seed}\n")

for iteration in range(1, 10000):
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"ahc030-{start_time}-group-{iteration:000}"

    n = random.randint(10, 20)
    m = random.randint(2, n * n // 20)
    eps = random.randint(1, 20) / 100
    avg_min = max(n * n // 5 // m, 4)
    avg_max = n * n // 2 // m
    avg = random.randint(avg_min, avg_max)

    (_, _, _, multi) = predict.predict(n, m, eps, avg, 100)

    cmd = (
        f"./gen.exe {SEED_PATH} -d {INPUT_PATH} --N {n} --M {m} --eps {eps} --avg {avg}"
    )
    print(cmd)

    if multi < 0.5:
        print("skipped.")
        continue

    subprocess.run(cmd).check_returncode()

    objective = optimize2.Objective()

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage="mysql+pymysql://default@localhost/optuna",
        load_if_exists=True,
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )

    study.enqueue_trial(
        {
            "min_cut": 0.3,
            "min_cut_pow": 2.0,
            "taboo_prob": 0.5,
            "max_entropy_len": 50,
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
