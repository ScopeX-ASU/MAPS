'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-11-24 15:10:46
'''
import os
import subprocess
from multiprocessing import Pool

# import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "fdfd"
model = "unet"
exp_name = "train"
root = f"log/{dataset}/{model}/{exp_name}"
script = './core/train/examples/fwd_model_train_s_param.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train_fwd.yml'
checkpoint_dir = f'{dataset}/{model}/{exp_name}'
configs.load(config_file, recursive=True)


def task_launcher(args):
    mixup, device_type, alg, train_field, include_adjoint_NN, fourier_feature, fno_block_only, mode1, mode2, id, description, gpu_id, epochs, alm, lr, criterion, criterion_weight, H_loss, maxwell_loss, grad_loss, s_param_loss, alm_lambda, alm_mu, mu_growth, constraint_tol, checkpt_fwd, checkpt_adj, bs = args
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
            f"--run.log_interval={80}",
            f"--run.random_state={59}",
            f"--run.fp16={False}",
            f"--run.include_adjoint_NN={include_adjoint_NN}",

            f"--criterion.name={criterion}",
            f"--criterion.weight={criterion_weight}",

            f"--aux_criterion.maxwell_residual_loss.weight={maxwell_loss}",
            f"--aux_criterion.maxwell_residual_loss.using_ALM={alm}",
            f"--aux_criterion.grad_loss.weight={grad_loss}",
            f"--aux_criterion.s_param_loss.weight={s_param_loss}",
            f"--aux_criterion.Hx_loss.weight={H_loss}",
            f"--aux_criterion.Hy_loss.weight={H_loss}",

            f"--log_criterion.maxwell_residual_loss.using_ALM={alm}",

            f"--test_criterion.name={'nmse'}",
            f"--test_criterion.weighted_frames={0}",
            f"--test_criterion.weight={1}",

            f"--scheduler.lr_min={lr*5e-3}",

            f"--plot.train={False}",
            f"--plot.val={False}",
            f"--plot.test={False}",
            f"--plot.interval=1",
            f"--plot.dir_name={model}_{exp_name}_id-{id}_des-{description}",

            f"--optimizer.lr={lr}",
            f"--optimizer.ALM={alm}",
            f"--optimizer.ALM_lambda={alm_lambda}",
            f"--optimizer.ALM_mu={alm_mu}",
            f"--optimizer.ALM_mu_growth={mu_growth}",
            f"--optimizer.ALM_constraint_tol={constraint_tol}",

            f"--model_fwd.type={alg}",
            f"--model_fwd.hidden_list={[128]}",
            f"--model_fwd.mode1={mode1}",
            f"--model_fwd.mode2={mode2}",
            f"--model_fwd.mapping_size={64}",
            f"--model_fwd.fno_block_only={fno_block_only}",
            f"--model_fwd.fourier_feature={fourier_feature}",
            f"--model_fwd.s_param_only={True}",

            f"--checkpoint.model_comment={suffix}",
            f"--checkpoint.resume={False}" if checkpt_fwd == "none" else f"--checkpoint.resume={True}",
            f"--checkpoint.restore_checkpoint_fwd={checkpt_fwd}",
            f"--checkpoint.restore_checkpoint_adj={checkpt_adj}",
            f"--checkpoint.checkpoint_dir={checkpoint_dir}",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    # mlflow.set_experiment(configs.run.experiment)  # set experiments first
    tasks = [
        [0.0, "bending", "UNet", "fwd", False, "learnable", True, 33, 66, 1, "unet_S_loss_only", 0, 50, False, 0.002, "nmse", 1, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 2, 1e-4, "none", "none", 8],
    ]   

    with Pool(8) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
