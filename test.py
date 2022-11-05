import os
import argparse
from tqdm import tqdm
import cv2

# import ffmpeg
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs import cfg
from metrics import psnr, ssim_metric
from utils.data_utils import select_dataset
import lpips
import time
from validate import load_render, mkdir

loss_fn_alex = lpips.LPIPS(net="alex").cuda()  # best forward scores
loss_fn_vgg = lpips.LPIPS(
    net="vgg"
).cuda()  # closer to "traditional" perceptual loss, when used for optimization
loss_fn_alex.eval()
loss_fn_vgg.eval()



def myinfer(infer_dataset, render, save_dir, epoch=0):
    render.eval()
    psnr_wMask_list = []
    psnr_woMask_list = []
    ssim_list = []
    lpips_alex_list = []
    lpips_vgg_list = []

    img_dir = f"{save_dir}/{epoch}/img"
    rendering_dir = f"{save_dir}/{epoch}/rendering"
    gt_dir = f"{save_dir}/{epoch}/ground_truth"
    acc_dir = f"{save_dir}/{epoch}/acc"
    depth_dir = f"{save_dir}/{epoch}/depth"
   
    mkdir(img_dir)
    mkdir(acc_dir)
    mkdir(depth_dir)
    mkdir(gt_dir)
    mkdir(rendering_dir)

    # with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(infer_dataset)):
        # batch['frame'][...] = 0
        real_frame = batch["frame"][0]
        ## fix frame code
        # batch["frame"][...] = 59
        if "save_name" in batch:
            save_name = batch["save_name"][0]
        else:
            frame_index = batch["frame_index"].item()
            view_index = batch["cam_ind"].item()
            save_name = "frame{:04d}_view{:04d}".format(frame_index, view_index)
        # import pdb;pdb.set_trace()
        results = render.render_view(batch)
        color_img_0 = results["coarse_color"]
        color_img_0 = torch.clamp(color_img_0, min=0.0, max=1.0)
        depth_img_0 = results["coarse_depth"]
        acc_map_0 = results["coarse_acc"]
        color_gt = batch["img"][0]

        H, W = color_gt.shape[:2]

        mask_at_box = batch["mask_at_box"][0].bool().reshape(H, W)

        psnr_wMask = psnr(color_img_0, color_gt, mask_at_box)
        psnr_woMask = psnr(color_img_0, color_gt)
        ssim_ = ssim_metric(
            color_img_0.cpu().numpy(), color_gt.cpu().numpy(), mask_at_box
        )

        pred = (
            (2 * color_img_0 - 1).permute(2, 0, 1)[None].float().flip(1)
        )  ### TO RGB ,(-1,1)
        gt = (2 * color_gt - 1).permute(2, 0, 1)[None].float().flip(1)
        pred = pred.cuda()
        gt = gt.cuda()
        with torch.no_grad():
            lpips_alex = loss_fn_alex(pred, gt).squeeze().cpu()
            lpips_vgg = loss_fn_vgg(pred, gt).squeeze().cpu()

        psnr_wMask_list.append(psnr_wMask)
        psnr_woMask_list.append(psnr_woMask)
        ssim_list.append(ssim_)
        lpips_alex_list.append(lpips_alex)
        lpips_vgg_list.append(lpips_vgg)
    

        img_path = os.path.join(img_dir, f"{save_name}.png")
        rendering_path = os.path.join(rendering_dir, f"{save_name}.png")
        gt_path = os.path.join(gt_dir, f"{save_name}.png")
        acc_path = os.path.join(acc_dir, f"{save_name}.png")
        depth_path = os.path.join(depth_dir, f"{save_name}.png")


        rendering = color_img_0.numpy() * 255
        gt = batch["img"].squeeze().numpy() * 255
        cat_img = np.concatenate((rendering, gt), axis=1)
        cv2.imwrite(img_path, cat_img)
        cv2.imwrite(rendering_path, rendering)
        cv2.imwrite(gt_path, gt)
        depth_img_0 = np.repeat(depth_img_0.numpy(), 3, axis=2) * 255
        cv2.imwrite(depth_path, depth_img_0)
        acc_map_0 = np.repeat(acc_map_0.numpy(), 3, axis=2) * 255
        cv2.imwrite(acc_path, acc_map_0)

    psnr_wMask_mean = np.array(psnr_wMask_list).mean()
    psnr_woMask_mean = np.array(psnr_woMask_list).mean()
    ssim_mean = np.array(ssim_list).mean()
    lpips_alex_mean = np.array(lpips_alex_list).mean()
    lpips_vgg_mean = np.array(lpips_vgg_list).mean()

    print("epoch", epoch)
    print("psnr_wMask_mean", psnr_wMask_mean)
    print("psnr_woMask_mean", psnr_woMask_mean)
    print("ssim_mean", ssim_mean)
    print("lpips_alex_mean", lpips_alex_mean)
    print("lpips_vgg_mean", lpips_vgg_mean)

    return {
        "psnr_wMask": psnr_wMask_mean,
        "psnr_woMask": psnr_woMask_mean,
        "ssim": ssim_mean,
        "lpips_alex": lpips_alex_mean,
        "lpips_vgg": lpips_vgg_mean,
    }


def save_img(imgs, img_dir):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    for idx in range(len(imgs)):
        img_path = os.path.join(img_dir, "%06d.jpg" % idx)
        cv2.imwrite(img_path, imgs[idx])


def img2vid(img_dir, output_path):
    (
        ffmpeg.input("%s/*.jpg" % img_dir, pattern_type="glob", framerate=15)
        .output(output_path)
        .run()
    )


if __name__ == "__main__":
    save_root = "./TEST"
    parser = argparse.ArgumentParser(description="infer")
    parser.add_argument(
        "-c",
        "--config",
        default="",
        help="set the config file path to train the network",
    )
    parser.add_argument("--exp", type=str, default="test")
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    epoch = int(os.path.basename(args.ckpt).split(".")[0].split("_")[-1])

    save_dir = os.path.join(save_root, args.exp)
    # Load config
    training_config = args.config
    assert os.path.exists(training_config), "training config does not exist."
    cfg.merge_from_file(training_config)

    novel_view_dataset, novel_pose_dataset = select_dataset(cfg, formal_test=True)
    novel_view_dataloader = DataLoader(
        novel_view_dataset, batch_size=1, shuffle=False, num_workers=4
    )
    novel_pose_dataloader = DataLoader(
        novel_pose_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    render = load_render(
        args.ckpt, cfg, canonical_vertex=novel_view_dataset.canonical_vertex
    )

    # import pdb;pdb.set_trace()
    # print(cfg)
    print("novel view:")
    out1 = myinfer(
        novel_view_dataloader,
        render,
        save_dir=os.path.join(save_dir, "novel_view"),
        epoch=epoch,
    )
    print("novel pose:")

    render.net.set_light_center(
        torch.tensor(cfg.TEST.light_center)
    )  
    render.net.nerf.w = 0

    out2 = myinfer(
        novel_pose_dataloader,
        render,
        save_dir=os.path.join(save_dir, "novel_pose"),
        epoch=epoch,
    )
    # import pdb;pdb.set_trace()
    # save_img(out, os.path.join(save_dir, "imgs"))
    # img2vid(save_dir, os.path.join(save_dir, f"{args.exp}.mp4"))





