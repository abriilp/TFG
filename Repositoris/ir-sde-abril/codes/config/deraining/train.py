import argparse
import logging
import math
import os
import random
import sys
import copy
import wandb #afegit abril
import random #afegit abril
import piq #
import torchvision.utils as vutils #

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# from IPython import embed

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr

# torch.autograd.set_detect_anomaly(True)

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    #### setup options of three networks

    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.", default='/home/apinyol/Gits/proves/repos_ref/IR-SDE-Abril/codes/config/deraining/options/train/ir-sde.yml')
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

    run = wandb.init(project=opt["wandb_proj"], name=opt["wandb_run"]) #afegit abril

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                print(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                print(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                print(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt) 
    device = model.device

    #### resume training
    if resume_state:
        print(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    scale = opt['degradation']['scale']

    #### training
    print(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )
    
    run.log({"epoch": start_epoch, "iter": current_step})

    best_psnr = 0.0
    best_iter = 0
    examples_val = []
    error = mp.Value('b', False)
    lpips_loss = piq.LPIPS()

    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)

        avg_psnr1 = 0.0
        avg_ssim1 = 0.0 
        avg_lpips1 = 0.0
        avg_mse1 = 0.0
        idx1 = 0

        for _, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                break

            LQ, GT = train_data["LQ"], train_data["GT"]
            timesteps, states = sde.generate_random_states(x0=GT, mu=LQ)

            model.feed_data(states, LQ, GT) # xt, mu, x0
            model.optimize_parameters(current_step, timesteps, sde)
            run.log({ "loss": model.log_dict['loss']})
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            if idx1 <= 1000:
                noisy_state = sde.noise_state(LQ)
                model.feed_data(noisy_state, LQ, GT) # xt, mu, x0
                model.test(sde)
                visuals = model.get_current_visuals()

                output1 = util.canvi_rang(visuals['Output'])
                input1 = util.canvi_rang(visuals['Input'])
                gt1 = util.canvi_rang(visuals['GT'])

                if idx1 <= 10:
                    grid = torch.cat((input1, output1, gt1), dim=0)
                    image_grid = vutils.make_grid(grid, nrow=4, normalize=True, scale_each=True)
                    image_grid_np = image_grid.permute(1, 2, 0).mul(255).clamp(0, 255).to('cpu', torch.uint8).numpy()
                    image1 = wandb.Image(image_grid_np, caption=f"batch {current_step}")
                    run.log({"examples_train": image1})

                # calculate PSNR
                avg_psnr1 += util.calculate_psnr(output1.numpy(), gt1.numpy())
                avg_ssim1 += piq.ssim(output1, gt1, data_range=255.)
                avg_lpips1 += lpips_loss(output1, gt1)
                avg_mse1 += ((output1 - gt1) ** 2).mean()
                idx1 += 1

            
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    print("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)
            
        if idx1 != 0:
            avg_psnr1 = avg_psnr1 / idx1
            avg_ssim1 = avg_ssim1 / idx1
            avg_lpips1 = avg_lpips1 / idx1 
            avg_mse1 = avg_mse1 / idx1
                    

        run.log({ "psnr_train": avg_psnr1, "ssim_train": avg_ssim1, "lpips_train": avg_lpips1, "mse_train": avg_mse1, "best_psnr": best_psnr, "epoch": epoch, "iter": current_step})
        
        print("TRAIN epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}, ssim: {:.6f}, lpips: {:.6f}, mse: {:.6f}".format(
            epoch, current_step, avg_psnr1, avg_ssim1, avg_lpips1, avg_mse1
                    ))
        
        # validation, to produce ker_map_list(fake)
        avg_psnr = 0.0
        avg_ssim = 0.0 
        avg_lpips = 0.0
        avg_mse = 0.0
        idx = 0
        for _, val_data in enumerate(val_loader):

            LQ, GT = val_data["LQ"], val_data["GT"]
            noisy_state = sde.noise_state(LQ)

            # valid Predictor
            model.feed_data(noisy_state, LQ, GT)
            model.test(sde)
            visuals = model.get_current_visuals()

            output = util.canvi_rang(visuals['Output'])
            input = util.canvi_rang(visuals['Input'])
            gt = util.canvi_rang(visuals['GT'])

            if idx <= 10:
                grid = torch.cat((input, output, gt), dim=0)
                image_grid = vutils.make_grid(grid, nrow=4, normalize=True, scale_each=True)
                image_grid_np = image_grid.permute(1, 2, 0).mul(255).clamp(0, 255).to('cpu', torch.uint8).numpy()
                image1 = wandb.Image(image_grid_np, caption=f"batch {current_step}")
                run.log({"examples_val": image1})

            avg_psnr += util.calculate_psnr(output.numpy(), gt.numpy())
            avg_ssim += piq.ssim(output, gt, data_range=255.)
            avg_lpips += lpips_loss(output, gt)
            avg_mse += ((output - gt) ** 2).mean()
            idx += 1

        if idx != 0:
            avg_psnr = avg_psnr / idx
            avg_ssim = avg_ssim / idx
            avg_lpips = avg_lpips / idx 
            avg_mse = avg_mse / idx
                    

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_iter = current_step
                    
        run.log({ "psnr_val": avg_psnr, "ssim_val": avg_ssim, "lpips_val": avg_lpips, "mse_val": avg_mse, "best_psnr": best_psnr, "epoch": epoch, "iter": current_step})

        print("TEST epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}, ssim: {:.6f}, lpips: {:.6f}, mse: {:.6f}".format(
            epoch, current_step, avg_psnr, avg_ssim, avg_lpips, avg_mse
                    ))

    if rank <= 0:
        print("Saving the final model.")
        model.save("latest")
        print("End of Predictor and Corrector training.")



if __name__ == "__main__":
    main()

