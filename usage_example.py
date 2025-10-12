# This is an example script to show how to use the MAPS package to train a model and perform inverse design
# This is the goal of the MAPS package, to provide a unified framework for training models and performing inverse design
# I can write code for MAPS package following the example below

from . import maps_data, maps_invdes, maps_train


def train_model_invdes():

    benchmark_cfg = {  # self defined a dictionary of benchmark configuration
        "device": "bending",  # device name, bending or stretching or splitter etc.
        "simulation": "FDFD",  # simulation name, FDTD or FDFD etc. but I think we can only support FDFD for now, see if we can support FDTD
        "resolution": 64,  # support only training on specific resolution(s)
        "wavelength": [
            1.55
        ],  # support only training on specific waveguide length(s) multiple wavelengths can be used for multi-task learning
        # wl_cen, wl_width, n_wl
        "sharpness": 256,  # suppport only training on specific sharpness(es) # on hold
        "distribution": "random",  # can choose from random pattern or pattern sample from optimization trajectory
    }
    model_cfg = {
        "Height": 64,  # number of pixels in height
        "Width": 64,  # number of pixels in width
        "Hidden": 64,  # number of hidden units
        # so on so forth
    }  # self defined a dictionary of model configuration
    model = maps_train.build_model(  # build a model or you can use any way to build a model as long as the model is a torch.nn.Module
        pretrained_model_path=None,
        model_name="FNO2D",  # model name, FNO or FFNO or NeurOLight or PACE etc.
        model_cfg=model_cfg,  # a dictionary of model configuration
    )
    train_loader, val_loader, test_loader = maps_data.get_data_loader(
        benchmark_cfg=benchmark_cfg,  # a dictionary of data configuration
        data_augmentation_method="rotate",  # default to be none which means no data augmentation and can choose from rotate, flip, superposition etc.
    )
    optimizer = maps_train.build_optimizer(
        model=model,
        optimizer_name="Adam",  # optimizer name, Adam or SGD etc.
        optimizer_cfg=None,  # a dictionary of optimizer configuration
    )
    scheduler = maps_train.build_scheduler(
        optimizer=optimizer,
        scheduler_name="StepLR",  # scheduler name, StepLR or MultiStepLR etc.
        scheduler_cfg=None,  # a dictionary of scheduler configuration
    )
    maps_train.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    # we can directly use eval_model() to evaluate the model on given benchmark
    maps_train.eval_model(
        model=model,
        benchmark_cfg=benchmark_cfg,  # a dictionary of benchmark configuration
    )

    # after training the model, we can put it into the inverse design loop to generate new designs
    invdes = maps_invdes.InverseDesign(
        model=model,
        benchmark_cfg=benchmark_cfg,
    )
    # we can even dump new designs into existing benchmark for active learning during optimization
    invdes.run_invdes(
        num_iters=100,  # number of iterations for each sample
        lr=1e-2,  # learning rate for optimization
        optimizer=optimizer,  # optimizer for optimization
        scheduler=scheduler,  # scheduler for optimization
        save_dir="invdes_results",  # directory to save the results
        active_learning=False,  # whether to perform active learning
        dump_data_cfg=benchmark_cfg,  # a dictionary of data configuration for dumping new designs
        output_gds_path="invdes_results",  # path to save the final design
    )


if __name__ == "__main__":
    train_model_invdes(pretrained_model_path=None)
