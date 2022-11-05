import os
import shutil

from configs import cfg
from trainer import do_train
from utils.loss import make_loss
from solver import make_optimizer, WarmupMultiStepLR, build_scheduler

from logger import setup_logger
from torch.utils.tensorboard import SummaryWriter
import torch

import argparse
from can_render import Renderer

from utils.model_utils import select_model
from utils.data_utils import select_dataset
from torch.utils.data import DataLoader


import torch
torch.manual_seed(233)
import random
random.seed(233)
import numpy as np
np.random.seed(233)


# Parse arguments
text = "This is the program to train the nerf, try to get help by using -h"
parser = argparse.ArgumentParser(description=text)
parser.add_argument(
    "-c", "--config", default="", help="set the config file path to train the network"
)
parser.add_argument(
    "-g", "--gpu", type=int, default=0, help="set gpu id to train the network"
)
parser.add_argument(
    "-r",
    "--resume",
    type=int,
    default=0,
    help="set the checkpoint number to resume training",
)
parser.add_argument(
    "-s",
    "--psnr_thres",
    type=float,
    default=100.0,
    help="set the psnr threshold to train next frame",
)
parser.add_argument(
    "-cont", "--cont", action="store_true", help="automatically continue to training"
)
parser.add_argument(
    "-noise",
    "--add_noise",
    type=float,
    default=0.0,
    help="set noise level, default is zero",
)


parser.add_argument("--exp", type=str, default="test")

args = parser.parse_args()

# Set PyTorch GPU id and settings
torch.cuda.set_device(args.gpu)
# torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

# Load config
training_config = args.config
assert os.path.exists(training_config), "training config does not exist."
cfg.merge_from_file(training_config)


cfg.freeze()
log_dir = "EXP"
log_path = os.path.join(log_dir, args.exp)
# Initialize writer and logger
output_dir = log_path
writer = SummaryWriter(log_dir=output_dir, max_queue=1)
writer.add_text("OUT_PATH", output_dir, 0)
logger = setup_logger("NERFRender", output_dir, 0)
logger.info("Running with config:\n{}".format(cfg))

# Save training config
shutil.copyfile(training_config, output_dir + "/config.yml")

# Create ray dataset

train_set, val_set = select_dataset(cfg, train_nrays=5500)

train_loader = DataLoader(
    train_set, batch_size=1, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS
)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)


model = select_model(cfg).cuda()
fine_model = None
render = Renderer(
    model, fine_net=fine_model, cfg=cfg, canonical_vertex=train_set.canonical_vertex
)
optimizer = make_optimizer(cfg, model)
scheduler = build_scheduler(
    optimizer,
    cfg.SOLVER.WARMUP_ITERS,
    cfg.SOLVER.START_ITERS,
    cfg.SOLVER.END_ITERS,
    cfg.SOLVER.LR_SCALE,
)

# Train from checkpoint with fixed iteration number
iter0 = 0
# Set loss function
loss_fn = make_loss(cfg)
# Set psnr threshold to automatically stop the program
psnr_thres = args.psnr_thres


# Train model
do_train(
    cfg,
    render,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    writer,
    resume_epoch=iter0,
    psnr_thres=psnr_thres,
    output_dir=output_dir,
)
