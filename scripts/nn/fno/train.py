'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-11-07 17:14:07
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
    mixup, device_type, alg, train_field, include_adjoint_NN, fourier_feature, fno_block_only, err_correction, mode1, mode2, id, description, gpu_id, epochs, lr, criterion, criterion_weight, H_loss, maxwell_loss, grad_loss, s_param_loss, checkpt_fwd, checkpt_adj, bs = args
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
            f"--run.log_interval={40}",
            f"--run.random_state={59}",
            f"--run.fp16={False}",
            f"--run.include_adjoint_NN={include_adjoint_NN}",

            f"--criterion.name={criterion}",
            f"--criterion.weight={criterion_weight}",

            f"--aux_criterion.maxwell_residual_loss.weight={maxwell_loss}",
            f"--aux_criterion.grad_loss.weight={grad_loss}",
            f"--aux_criterion.s_param_loss.weight={s_param_loss}",
            f"--aux_criterion.err_corr_Ez.weight={1 if err_correction else 0}",
            f"--aux_criterion.err_corr_Hx.weight={1 if err_correction else 0}",
            f"--aux_criterion.err_corr_Hy.weight={1 if err_correction else 0}",
            f"--aux_criterion.Hx_loss.weight={H_loss}",
            f"--aux_criterion.Hy_loss.weight={H_loss}",

            f"--log_criterion.err_corr_Ez.weight={1 if err_correction else 0}",
            f"--log_criterion.err_corr_Hx.weight={1 if err_correction else 0}",
            f"--log_criterion.err_corr_Hy.weight={1 if err_correction else 0}",

            f"--test_criterion.name={'nmse'}",
            f"--test_criterion.weighted_frames={0}",
            f"--test_criterion.weight={1}",

            f"--scheduler.lr_min={lr*5e-3}",

            f"--plot.train={True}",
            f"--plot.valid={True}",
            f"--plot.test={True}",
            f"--plot.interval=1",
            f"--plot.dir_name={model}_{exp_name}_id-{id}_des-{description}",
            f"--optimizer.lr={lr}",

            f"--model.name={alg}",
            f"--model.in_channels={3}",
            f"--model.out_channels={2}",
            f"--model.hidden_list={[128]}",
            # f"--model.mode1={150}",
            # f"--model.mode2={225}",
            # f"--model.mode1={30}",
            # f"--model.mode2={45}",
            f"--model.mode1={mode1}",
            f"--model.mode2={mode2}",
            f"--model.fourier_feature={fourier_feature}",
            # f"--model.fourier_feature={'none'}",
            f"--model.mapping_size={64}",
            f"--model.err_correction={err_correction}",
            f"--model.fno_block_only={fno_block_only}",
            f"--model.train_field={train_field}",

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
        # [0.0, "metacoupler", "FNO3d", 150, 225, 7, "full_mode", 0, 50, 0.002, "nmse", 1, 0, "none", 2],
        # [0.0, "metacoupler", "FNO3d", 30, 45, 8, "less_mode", 1, 50, 0.002, "nmse", 1, 0, "none", 2],
        # [0.0, "metacoupler", "FNO3d", 30, 45, 9, "less_mode_ripped_dataset", 2, 50, 0.002, "nmse", 1, 0, "none", 2],
        # [0.0, "metacoupler", "FNO3d", 30, 45, 10, "less_mode_maxwell_loss", 0, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.5, "none", 2],
        # [0.0, "bending", "FNO3d", 33, 66, 0, "plain", 0, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.0, "none", 2],
        # [0.0, "bending", "FNO3d", 10, 20, 1, "plain_lessmodes", 0, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.0, "none", 2],
        # [0.0, "bending", "FNO3d", 33, 66, 2, "plain_lessvar", 0, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.0, "none", 2],
        # [0.0, "bending", "FNO3d", 10, 20, 3, "test_more_batchsize", 1, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.0, "none", 16],
        # [0.0, "metacoupler", "FNO3d", True, 20, 280, 5, "single_layer_coupler_less_x_mode", 0, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.0, "none", 12],
        # [0.0, "bending", "FNO3d", True, True, 33, 66, 4, "fourier_feature_pml_layer_draw", 0, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.0, "./checkpoint/fdfd/fno/train/FNO3d_model-FNO3d_dev-bending_id-4_dcrp-fourier_feature_pml_layer_err-0.0643_epoch-44.pt", 2],
        # [0.0, "bending", "FNO3d", True, True, 33, 66, 5, "LFF", 0, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.0, "none", 2],
        # [0.0, "bending", "FNO3d", True, True, 33, 66, 6, "LFF_err_corr", 0, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.0, "none", 2],
        # [0.0, "bending", "FNO3d", True, True, 33, 66, 7, "LFF_err_corr_bs_8", 1, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.0, "none", 8],
        # [0.0, "bending", "FNO3d", True, False, 33, 66, 8, "LFF_bs_8_no_corr", 2, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.0, "none", 8],
        # [0.0, "bending", "FNO3d", True, False, 33, 66, 9, "LFF_bs_8_no_corr_all_field_loss", 3, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.0, "none", 8],
        # [0.0, "bending", "FNO3d", True, False, 33, 66, 10, "LFF_bs_8_no_corr_all_field_loss_print_loss_comp", 3, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.5, "./checkpoint/fdfd/fno/train/FNO3d_model-FNO3d_dev-bending_id-9_dcrp-LFF_bs_8_no_corr_all_field_loss_err-0.0679_epoch-50.pt", 8],
        # [0.0, "bending", "FNO3d", True, False, 33, 66, 18, "LFF_bs_8_no_corr_print_loss_comp", 2, 50, 0.002, "nmse", 1, 1, 0.0, 0.0, 0.0, "./checkpoint/fdfd/fno/train/FNO3d_model-FNO3d_dev-bending_id-8_dcrp-LFF_bs_8_no_corr_err-0.0589_epoch-50.pt", 8],
        # [0.0, "bending", "FNO3d", True, False, 33, 66, 12, "no_corr_src", 0, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.0, 0.0, "none", 8],
        # [0.0, "bending", "FNO3d", True, False, 33, 66, 13, "no_corr_src_maxwell", 1, 50, 0.002, "nmse", 1, 0.0, 20, 0.0, 0.0, "none", 8],
        # [0.0, "bending", "FNO3d", True, False, 33, 66, 14, "no_corr_src_maxwell_no_regression", 0, 50, 0.002, "nmse", 1, 0.0, 20, 0.0, 0.0, "none", 8],
        # [0.0, "bending", "FNO3d", True, False, 33, 66, 15, "no_corr_src_no_pos_enc", 0, 50, 0.002, "nmse", 1, 0.0, 0.0, 0.0, 0.0, "none", 8],
        # [0.0, "bending", "FNO3d", True, False, 33, 66, 16, "LFF_lightFiled_HLoss", 0, 50, 0.002, "nmse", 1, 1, 0.0, 0.0, 0.0, "none", 8],
        # [0.0, "bending", "FNO3d", True, False, 33, 66, 17, "no_pos_enc_lightFiled_HLoss", 1, 50, 0.002, "nmse", 1, 1, 0.0, 0.0, 0.0, "none", 8],
        # [0.0, "bending", "FNO3d", "both", "learnable", True, False, 33, 66, 18, "LFF_HLoss_fwd_adj", 1, 50, 0.002, "nmse", 1, 1, 0.0, 0.0, 0.0, "none", 8],
        # [0.0, "bending", "FNO3d", "both", "learnable", True, False, 33, 66, 19, "LFF_HLoss_fwd_adj_grad_loss", 2, 50, 0.002, "nmse", 1, 1, 0.0, 0.5, 0.0, "none", 8],
        [0.0, "bending", "FNO3d", "fwd", True, "learnable", True, False, 33, 66, 20, "end2end_fwd_adj", 1, 50, 0.002, "nmse", 1, 1, 0.0, 0.0, 0.0, "none", "none", 8],
    ]

    with Pool(8) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
