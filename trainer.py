import logging
logging.basicConfig(level=logging.INFO)
import torch

import numpy as np
import os, time
from utils.checkpoint import Checkpointer, PeriodicCheckpointer, CheckpointableDict
from metrics import psnr, ssim_metric
from validate import val


def do_train(
    cfg,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    swriter,
    resume_epoch=0,
    psnr_thres=100,
    output_dir="",
):
    # set local vars
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = output_dir
    max_epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("NeRF.%s.train" % cfg.OUTPUT_DIR.split("/")[-1])
    logger.setLevel(logging.DEBUG)
    logger.info("Start training")
    # load model parameters
    training_status = CheckpointableDict(epoch=0, iteration=0)
    checkpointer = Checkpointer(
        model.net,
        save_dir=output_dir,
        training_status=training_status,
        scheduler=scheduler,
        optimizer=optimizer,
    )
    if checkpointer.has_checkpoint():
        last_checkpoint = checkpointer.resume_or_load(
            "output/delta_bw_ebxyzh/checkpoint_30.pt", resume=True
        )
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, checkpoint_period, max_iter=99999999, max_epoch=max_epochs
    )

    model.train()

    # global step
    global_step = training_status.iteration
    resume_epoch = training_status.epoch
    for epoch in range(1 + resume_epoch, max_epochs):
        print("Training Epoch %d..." % epoch)
        # psnr monitor
        psnr_monitor = []

        # epoch time recording
        epoch_start = time.time()
        for batch_idx, batch in enumerate(train_loader):
            # iteration time recording
            iters_start = time.time()
            global_step = (epoch - 1) * len(train_loader) + batch_idx

            optimizer.zero_grad()

            tmp = model.render(batch)
            coarse = tmp["coarse"]
            loss1 = loss_fn(coarse, batch)
           
            loss = 0
            for key in loss1:
                loss += loss1[key]

            loss.backward()

            optimizer.step()
            scheduler.step()

            psnr_ = psnr(coarse["color"], batch["rgb"].reshape(-1, 3).cuda())
            psnr_monitor.append(psnr_.cpu().detach().numpy())

            if batch_idx % 50 == 0:
                for key in loss1:
                    swriter.add_scalar(f"Loss/{key}", loss1[key].item(), global_step)
                swriter.add_scalar("Loss/loss_sum", loss.item(), global_step)
                swriter.add_scalar("TrainPsnr", psnr_, global_step)
                swriter.add_scalar("LR", scheduler.get_lr()[0], global_step)

            if batch_idx % log_period == 0:
                for param_group in optimizer.param_groups:
                    lr = param_group["lr"]
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3e}  Psnr coarse: {:.2f} Psnr fine: {:.2f} Lr: {:.2e} Speed: {:.1f}[rays/s]".format(
                        epoch,
                        batch_idx,
                        len(train_loader),
                        loss.item(),
                        psnr_,
                        psnr_,
                        lr,
                        log_period
                        * float(cfg.SOLVER.BUNCH)
                        / (time.time() - iters_start),
                    )
                )
    
            # model saving
            # if global_step % checkpoint_period == 0:
                # ModelCheckpoint(model, optimizer, scheduler, output_dir, epoch)

        # EPOCH COMPLETED
        # ModelCheckpoint(model, optimizer, scheduler, output_dir, epoch)
        training_status.iteration = global_step
        training_status.epoch = epoch
        periodic_checkpointer.step_by_epoch(epoch=epoch)

        if epoch % 40 == 0 and epoch!=0:
            val_vis(val_loader, model, loss_fn, swriter, logger, epoch, cfg, output_dir)

        logger.info(
            "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[rays/s]".format(
                epoch,
                time.time() - epoch_start,
                len(train_loader)
                * float(cfg.SOLVER.BUNCH)
                / (time.time() - epoch_start),
            )
        )

        psnr_monitor = np.mean(psnr_monitor)

        if psnr_monitor > psnr_thres:
            logger.info(
                "The Mean Psnr of Epoch: {:.3f}, greater than threshold: {:.3f}, Training Stopped".format(
                    psnr_monitor, psnr_thres
                )
            )
            break
        else:
            logger.info(
                "The Mean Psnr of Epoch: {:.3f}, less than threshold: {:.3f}, Continue to Training".format(
                    psnr_monitor, psnr_thres
                )
            )


def val_vis(val_loader, model, loss_fn, swriter, logger, epoch, cfg, output_dir):
    res = val(val_loader, model, f"{output_dir}/vis", epoch)
    model.train()
    logger.info(
        "Validation Results - Epoch: {} psnr_wMask: {:.3f}".format(
            epoch, res["psnr_wMask"]
        )
    )
    for key in res.keys():
        swriter.add_scalar(f"Val/{key}", res[key], epoch)


def ModelCheckpoint(model, optimizer, scheduler, output_dir, epoch):
    # model,optimizer,scheduler saving
    torch.save(
        {
            "model": model.net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        os.path.join(output_dir, "checkpoint_%d.pt" % epoch),
    )


