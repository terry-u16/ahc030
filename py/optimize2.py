import json
import math
import re
import shutil
import subprocess

import optuna

TIME_RATIO = 1.5


class Objective:
    def __init__(self) -> None:
        pass

    def __call__(self, trial: optuna.trial.Trial) -> float:
        taboo_prob = trial.suggest_float("taboo_prob", 0.0, 1.0, log=False)
        max_entropy_len = trial.suggest_int("max_entropy_len", 20, 500, log=True)

        min_seed = 0
        max_seed = 511
        batch_size = 64
        step = 0
        score_sum = 0.0
        args = f"{taboo_prob} {max_entropy_len}"
        local_execution = f"tester.exe ahc030.exe {args}"
        cloud_execution = f"tester main {args}"
        print(f">> {local_execution}")

        for begin in range(min_seed, max_seed + 1, batch_size):
            end = begin + batch_size

            with open("runner_config_original.json", "r") as f:
                config = json.load(f)

            config["RunnerOption"]["StartSeed"] = begin
            config["RunnerOption"]["EndSeed"] = end
            config["ExecutionOption"]["LocalExecutionSteps"][0][
                "ExecutionCommand"
            ] = local_execution
            config["ExecutionOption"]["CloudExecutionSteps"][0][
                "ExecutionCommand"
            ] = cloud_execution

            with open("runner_config.json", "w") as f:
                json.dump(config, f, indent=2)

            command = "dotnet marathon run-local"
            process = subprocess.run(command, stdout=subprocess.PIPE, encoding="utf-8")

            lines = process.stdout.splitlines()
            score_pattern = r"rate:\s*(\d+.\d+)%"

            for line in lines:
                result = re.search(score_pattern, line)
                if result:
                    score = float(result.group(1))
                    if score > 0.0:
                        score = math.log10(score)
                    else:
                        score = math.log10(10.00)
                    score_sum += score

            if end < max_seed + 1:
                trial.report(score_sum, step)
                print(f"{score_sum:.5f}", end=" ", flush=True)

                if trial.should_prune():
                    print()
                    raise optuna.TrialPruned()

            step += 1

        print(f"{score_sum:.5f}")
        return score_sum


if __name__ == "__main__":
    STUDY_NAME = "ahc030-010"

    # subprocess.run("dotnet marathon compile-rust")
    subprocess.run("cargo build --release")
    shutil.move("../target/release/ahc030.exe", "./ahc030.exe")

    objective = Objective()

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage="mysql+pymysql://default@localhost/optuna",
        load_if_exists=True,
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )

    if len(study.trials) == 0:
        study.enqueue_trial(
            {
                "min_cut": 0.3,
                "min_cut_pow": 2.0,
                "taboo_prob": 0.5,
                "max_entropy_len": 50,
            }
        )

    study.optimize(objective, timeout=60)
    print(study.best_trial)

    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_slice(study, params=["min_cut", "min_cut_pow"]).show()
    optuna.visualization.plot_contour(study).show()
