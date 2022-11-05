import os
import argparse
from tqdm import tqdm
import cv2

# import ffmpeg
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs import cfg
from can_render import Renderer
from metrics import psnr, ssim_metric
from utils.model_utils import select_model
from utils.data_utils import select_dataset


def load_render(ckpt_path, cfg, canonical_vertex):
    model = select_model(cfg)
    ckpt = torch.load(ckpt_path)
    model = model.cuda()
    fine_model = None
    render = Renderer(
        model, fine_net=fine_model, cfg=cfg, canonical_vertex=canonical_vertex
    )
    render.eval()
    render.net.load_state_dict(ckpt["model"])
    return render

def mkdir(img_dir):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir, exist_ok=True)

def val(infer_dataset, render, save_dir, epoch=0):
    render.eval()
    psnr_wMask_list = []
    psnr_woMask_list = []
    ssim_list = []
    img_dir = f"{save_dir}/{epoch}/img"
    acc_dir = f"{save_dir}/{epoch}/acc"
    depth_dir = f"{save_dir}/{epoch}/depth"
    mkdir(img_dir)
    mkdir(acc_dir)
    mkdir(depth_dir)

    for batch_idx, batch in enumerate(tqdm(infer_dataset)):
        real_frame = batch["frame"][0]
        batch["frame"][...] = 50

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

        psnr_wMask_list.append(psnr_wMask)
        psnr_woMask_list.append(psnr_woMask)
        ssim_list.append(ssim_)

        img_path = os.path.join(img_dir, f"%06d_{batch_idx}.jpg" % real_frame)
        acc_path = os.path.join(acc_dir, f"%06d_{batch_idx}.jpg" % real_frame)
        depth_path = os.path.join(depth_dir, f"%06d_{batch_idx}.jpg" % real_frame)

        rendering = color_img_0.numpy() * 255
        gt = batch["img"].squeeze().numpy() * 255
        cat_img = np.concatenate((rendering, gt), axis=1)
        cv2.imwrite(img_path, cat_img)
        depth_img_0 = np.repeat(depth_img_0.numpy(), 3, axis=2) * 255
        cv2.imwrite(depth_path, depth_img_0)
        acc_map_0 = np.repeat(acc_map_0.numpy(), 3, axis=2) * 255
        cv2.imwrite(acc_path, acc_map_0)


    psnr_wMask_mean = np.array(psnr_wMask_list).mean()
    psnr_woMask_mean = np.array(psnr_woMask_list).mean()
    ssim_mean = np.array(ssim_list).mean()
    print(epoch)
    print("psnr_wMask_mean", psnr_wMask_mean)
    print("psnr_woMask_mean", psnr_woMask_mean)
    print("ssim_mean", ssim_mean)
    return {
        "psnr_wMask": psnr_wMask_mean,
        "psnr_woMask": psnr_woMask_mean,
        "ssim": ssim_mean,
    }



def img2vid(img_dir, output_path):
    (
        ffmpeg.input("%s/*.jpg" % img_dir, pattern_type="glob", framerate=15)
        .output(output_path)
        .run()
    )


if __name__ == "__main__":
    save_root = "./vis"
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

    save_dir = os.path.join(save_root, args.exp)
    img_dir = os.path.join(save_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)
    # Load config
    training_config = args.config
    assert os.path.exists(training_config), "training config does not exist."
    cfg.merge_from_file(training_config)

    dataset = select_dataset(cfg, formal_test= True)
    
    # dataset, _ = select_dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    render = load_render(args.ckpt, cfg, canonical_vertex=dataset.canonical_vertex)
    out = val(dataloader, render, save_dir=img_dir)
    # img2vid(save_dir, os.path.join(save_dir, f"{args.exp}.mp4"))

