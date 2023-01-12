import argparse
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
# from compressai.losses import RateDistortionLoss
from compressai.zoo import image_models

from PIL import Image
from pytorch_msssim import ms_ssim

import math
import time
from PIL import Image

from VQLIC.models import (
    get_model,
)
from VQLIC.functions import (
    AverageMeter,
    get_mu_and_sigma,
    compute_psnr,
    compute_msssim,
    compute_bpp,
    eval_images,
    save_model,
    log_net_summery,
    # optimizer setting
    configure_optimizers,
    configure_optimizers_separate,
    # criterions
    FTAVQ_loss,
    E2E_AVQ_loss,
    # train & test proccess
    train_one_epoch,
    test_epoch,
)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    

    
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    
    # MOD start
    parser.add_argument("--cuda", default=True, help="Use cuda")
    parser.add_argument("--save", default=True, help="Save model to disk")    
    parser.add_argument("--lambda", dest="lmbda", type=float,default=3e-3, help="Bit-rate distortion parameter (default: %(default)s)",)                
    parser.add_argument("-e", "--epochs", default=2000, type=int, help="Number of epochs (default: %(default)s)",)    
    parser.add_argument("-m", "--model", default="AutoEncoder", help="Model architecture (default: %(default)s)",)
    parser.add_argument("-d", "--dataset", type=str, default="../DIV2K_HR", help="Training dataset",)    
    parser.add_argument("--fix-AE", default=False, help="fix parameters of AutoEncoder",)    
    parser.add_argument("-i","--iterations", type=int, default=1, help="Exeute iterations (default: %(default)s)",)
    parser.add_argument("--vector-dim", type=int, default=64, help="setting vector quantization",)
    parser.add_argument("-cbs","--codebook-size", type=int, default=0, help="setting vector quantization codebook",)
    parser.add_argument("-q","--quantizers", type=int, default=1, help="setting vector quantization",)
    parser.add_argument("-p","--pretrained", type=str, default="", help="Load AE Parameters")
    # MOD end

    args = parser.parse_args(argv)
    return args        
        
def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
        
    CB_size = [512, 256, 128, 64, 32, 16, 8, 4, 2]    
    for i in range(args.iterations):
        log_s = f'loss, mse, bpp, aux, psnr\n'
        with open('log_training.csv', 'a') as f:
            f.write(log_s)
        picked_model = get_model(args.model)
        
        if args.codebook_size == 0:
            CB = CB_size
        else:
            CB_index = int(9 - math.log2(args.codebook_size))
            CB = CB_size[CB_index:]
            
        version = f'{args.model}_{args.quantizers}'
        if (args.model == "AutoEncoder"):
            net = picked_model(N=128, dim=args.vector_dim)
        else:
            net = picked_model(N=128, dim=args.vector_dim, quantizers = args.quantizers, CB_size=CB[i])        
        net = net.to(device)
        if not args.pretrained == "":
            net.load_AE(args.pretrained)
        
        optimizer_list = configure_optimizers(net) if (not args.fix_AE) else  configure_optimizers_seprate(net)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_list[0], "min", eps = 1e-9, cooldown = 10, verbose = True)
        criterion = E2E_AVQ_loss(lmbda=args.lmbda) if (not args.fix_AE) else FTAVQ_loss(lmbda=args.lmbda)
        last_epoch = 0
        best_loss = float("inf")
        
        time_start = time.time()
        for epoch in range(last_epoch, args.epochs):
            # train
            train_one_epoch(epoch, net, train_dataloader, optimizer_list, criterion, args.clip_max_norm)
            loss = test_epoch(epoch, test_dataloader, net, criterion)            
            lr_scheduler.step(loss)
            # To prevent initial training oscillation from causing the best loss value to be too low
            if epoch == 10:
                lr_scheduler._reset()
            # End condition
            if optimizer_list[0].param_groups[0]['lr'] < 1e-8:
                break
        time_end = time.time()
        time_min = (time_end - time_start) / 60                
        save_model(net, version, CB[i])

        # summery
        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = net(d)
        data_ch_bpp = math.log2(CB[i]) * args.quantizers / 8 / 8        
        log_net_summery(
            args,
            epoch + 1,
            net,
            version,
            data_ch_bpp,
            time_min,
            out_net,
        )
if __name__ == "__main__":
    main(sys.argv[1:])