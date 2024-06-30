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

def main():
    #### setup options of three networks

    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
    #opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.parse("/home/apinyol/Gits/proves/repos_ref/IR-SDE-Abril/codes/config/deraining/options/test/ir-sde.yml", is_train=False)
    opt = option.dict_to_nonedict(opt)

    ###### Predictor&Corrector train ######

    run = wandb.init(project=opt["wandb_proj"], name=opt["wandb_run"]) #afegit abril

    #### create dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt["datasets"].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)

    # load pretrained model by default
    model = create_model(opt)
    device = model.device

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    scale = opt['degradation']['scale']

    for test_loader in test_loaders:
        avg_psnr = 0.0
        avg_ssim = 0.0 
        avg_lpips = 0.0
        avg_mse = 0.0
        idx = 0
        lpips_loss = piq.LPIPS()
        for _, val_data in enumerate(test_loader):

            LQ, GT = val_data["LQ"], val_data["GT"]
            noisy_state = sde.noise_state(LQ)

            # valid Predictor
            model.feed_data(noisy_state, LQ, GT)
            model.test(sde)
            visuals = model.get_current_visuals()

            output = util.canvi_rang(visuals['Output'])
            input = util.canvi_rang(visuals['Input'])
            gt = util.canvi_rang(visuals['GT'])

            
            grid = torch.cat((input, output, gt), dim=0)
            image_grid = vutils.make_grid(grid, nrow=4, normalize=True, scale_each=True)
            image_grid_np = image_grid.permute(1, 2, 0).mul(255).clamp(0, 255).to('cpu', torch.uint8).numpy()
            image1 = wandb.Image(image_grid_np, caption=f"{idx}")
            run.log({"examples_val": image1})

                        # calculate PSNR
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
                        

                        
        run.log({ "psnr_val": avg_psnr, "ssim_val": avg_ssim, "lpips_val": avg_lpips, "mse_val": avg_mse})

        print("TEST psnr: {:.6f}, ssim: {:.6f}, lpips: {:.6f}, mse: {:.6f}".format(
            avg_psnr, avg_ssim, avg_lpips, avg_mse
                        ))


if __name__ == "__main__":
    main()

