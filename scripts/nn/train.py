"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-13 15:54:53
"""

import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.config import configs
from pyutils.general import ensure_dir, logger

dataset = "fdfd"
model = "cnn"
exp_name = "train_test"
root = f"log/{dataset}/{model}/{exp_name}"
script = "train_NN.py"
config_file = f"configs/{dataset}/{model}/{exp_name}/train.yml"
checkpoint_dir = f"{dataset}/{model}/{exp_name}"
configs.load(config_file, recursive=True)


def task_launcher(args):
    (
        mixup,
        device_type,
        alg,
        id,
        description,
        gpu_id,
        epochs,
        lr,
        criterion,
        criterion_weight,
        checkpt,
        bs,
    ) = args
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    pres = ["python3", script, config_file]
    suffix = f"model-{alg}_id-{id}"
    with open(os.path.join(root, f"{suffix}.log"), "w") as wfid:
        exp = [
            f"--dataset.device_type={device_type}",
            f"--dataset.processed_dir={device_type}",
            f"--dataset.num_workers={4}",
            f"--dataset.augment.prob={mixup}",
            f"--run.n_epochs={epochs}",
            f"--run.batch_size={bs}",
            f"--run.gpu_id={gpu_id}",
            f"--run.log_interval=200",
            f"--run.random_state={59}",
            f"--run.fp16={False}",
            f"--criterion.name={criterion}",
            f"--criterion.weight={criterion_weight}",
            f"--test_criterion.name={'mse'}",
            f"--test_criterion.weighted_frames={0}",
            f"--test_criterion.weight={1}",
            f"--scheduler.lr_min={lr*5e-3}",
            f"--plot.train=False",
            f"--plot.valid=False",
            f"--plot.test=False",
            f"--plot.interval=1",
            f"--plot.dir_name={model}_{exp_name}_des-{description}_id-{id}",
            f"--optimizer.lr={lr}",
            f"--model.name={alg}",
            f"--checkpoint.model_comment={suffix}",
            (
                f"--checkpoint.resume={False}"
                if checkpt == "none"
                else f"--checkpoint.resume={True}"
            ),
            f"--checkpoint.restore_checkpoint={checkpt}",
            f"--checkpoint.checkpoint_dir={checkpoint_dir}",
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        [
            0.0,
            "metacoupler",
            "NeurOLight2d",
            0,
            "test_training_flow",
            3,
            50,
            0.02,
            "mse",
            1,
            "none",
            2,
        ],
    ]

    with Pool(8) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
