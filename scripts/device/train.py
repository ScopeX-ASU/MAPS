import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.config import configs
from pyutils.general import ensure_dir, logger

device = "bending"
model = "local_search"
exp_name = "train_local_search"
root = f"log/{device}/{model}/{exp_name}"
script = "train.py"
config_file = f"configs/{device}/{model}/{exp_name}/train.yml"
configs.load(config_file, recursive=True)
checkpoint_dir = f"{device}/{model}/{exp_name}"


def task_launcher(args):

    (
        device_type,
        gpu_id,
        lr,
        init_sharp,
        final_sharp,
        res,
        test_res,
        epochs,
        id,
        comment,
    ) = args
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    pres = ["python3", script, config_file]

    with open(
        os.path.join(
            root,
            f"id-{id}_device-{device_type}_lr-{lr}_initS-{init_sharp}_finalS-{final_sharp}_com-{comment}.log",
        ),
        "w",
    ) as wfid:
        exp = [
            f"--optimizer.lr={lr}",
            f"--optimizer.name={'adam'}",
            f"--lr_scheduler.lr_min={lr*1e-2}",
            f"--sharp_scheduler.init_sharp={init_sharp}",
            f"--sharp_scheduler.final_sharp={final_sharp}",
            f"--res_scheduler.init_res={res}",
            f"--res_scheduler.final_res={res}",
            f"--res_scheduler.test_res={test_res}",
            f"--run.gpu_id={gpu_id}",
            f"--run.n_epochs={epochs}",
            f"--run.random_state={59}",
            f"--run.fp16={False}",
            f"--model.device_type={device_type}",
            f"--model.name={device_type + 'Optimization'}",
            f"--model.sim_cfg.plot_root={'./plot/' + str(device) + '_' + str(exp_name) + '_' + str(id)}",
            f"--checkpoint.comment={comment}",
            f"--plot.train={True}",
            f"--plot.test={True}",
            f"--plot.valid={True}",
            f"--plot.root={'./plot/'}",
            f"--plot.dir_name={device}_{exp_name}_{id}",
        ]
        logger.info(f"running command:\n\t{' '.join(pres + exp)}")

        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        ["Bending", 0, 0.02, 4, 200, 50, 50, 50, 0, "put_your_comment_here"],
    ]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
