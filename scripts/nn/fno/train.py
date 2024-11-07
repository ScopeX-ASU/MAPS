'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-13 15:54:53
'''
import os
import subprocess
from multiprocessing import Pool

# import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "fdfd"
model = "fno"
exp_name = "train"
root = f"log/{dataset}/{model}/{exp_name}"
script = 'train_NN.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
checkpoint_dir = f'{dataset}/{model}/{exp_name}'
configs.load(config_file, recursive=True)


def task_launcher(args):
    mixup, device_type, alg, mode1, mode2, id, description, gpu_id, epochs, lr, criterion, criterion_weight, maxwell_loss, grad_loss, s_param_loss, checkpt, bs = args
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    pres = [
            'python3',
            script,
            config_file
            ]
    suffix = f"model-{alg}_dev-{device_type}_id-{id}_dcrp-{description}"
    with open(os.path.join(root, f'{suffix}.log'), 'w') as wfid:
        exp = [
            f"--dataset.device_type={device_type}",
            f"--dataset.processed_dir={device_type}",
            f"--dataset.num_workers={4}",
            f"--dataset.augment.prob={mixup}",

            f"--run.n_epochs={epochs}",
            f"--run.batch_size={bs}",
            f"--run.use_cuda={1}",
            f"--run.gpu_id={gpu_id}",
            f"--run.log_interval={400}",
            f"--run.random_state={59}",
            f"--run.fp16={False}",

            f"--criterion.name={criterion}",
            f"--criterion.weight={criterion_weight}",

            f"--aux_criterion.maxwell_residual_loss.weight={maxwell_loss}",
            f"--aux_criterion.grad_loss.weight={grad_loss}",
            f"--aux_criterion.s_param_loss.weight={s_param_loss}",

            f"--test_criterion.name={'nmse'}",
            f"--test_criterion.weighted_frames={0}",
            f"--test_criterion.weight={1}",

            f"--scheduler.lr_min={lr*5e-3}",

            f"--plot.train={True}",
            f"--plot.valid={True}",
            f"--plot.test={True}",
            f"--plot.interval=1",
            f"--plot.dir_name={model}_{exp_name}_des-{description}_id-{id}",
            f"--optimizer.lr={lr}",

            f"--model.name={alg}",
            f"--model.in_channels={3}",
            f"--model.out_channels={2}",
            # f"--model.mode1={150}",
            # f"--model.mode2={225}",
            # f"--model.mode1={30}",
            # f"--model.mode2={45}",
            f"--model.mode1={mode1}",
            f"--model.mode2={mode2}",

            f"--checkpoint.model_comment={suffix}",
            f"--checkpoint.resume={False}" if checkpt == "none" else f"--checkpoint.resume={True}",
            f"--checkpoint.restore_checkpoint={checkpt}",
            f"--checkpoint.checkpoint_dir={checkpoint_dir}",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    # mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        # [0.0, "metacoupler", "FNO3d", 150, 225, 7, "full_mode", 0, 50, 0.002, "nmse", 1, 0, "none", 2],
        # [0.0, "metacoupler", "FNO3d", 30, 45, 8, "less_mode", 1, 50, 0.002, "nmse", 1, 0, "none", 2],
        # [0.0, "metacoupler", "FNO3d", 30, 45, 9, "less_mode_ripped_dataset", 2, 50, 0.002, "nmse", 1, 0, "none", 2],
        # [0.0, "metacoupler", "FNO3d", 30, 45, 10, "less_mode_maxwell_loss", 0, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.5, "none", 2],
        [0.0, "bending", "FNO3d", 30, 45, 0, "plain", 0, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.0, "none", 2],
        [0.0, "bending", "FNO3d", 30, 45, 1, "plain_maxwell", 1, 50, 0.002, "nmse", 1, 0.1, 0.0, 0.0, "none", 2],
        # [0.0, "bending", "FNO3d", 30, 45, 2, "plain_gradient", 2, 50, 0.002, "nmse", 1, 0.0, 0.1, 0.0, "none", 2],
        # [0.0, "bending", "FNO3d", 30, 45, 3, "plain_s_param", 3, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.1, "none", 2],
    ]

    with Pool(8) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
