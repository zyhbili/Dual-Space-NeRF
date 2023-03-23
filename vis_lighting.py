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
import time


def load_render(ckpt_path, cfg, canonical_vertex):
    model = select_model(cfg)
    ckpt = torch.load(ckpt_path)
    model = model.cuda()
    fine_model = None
    render = Renderer(
        model, fine_net=fine_model, cfg=cfg, canonical_vertex=canonical_vertex
    )
    render.net.load_state_dict(ckpt["model"])
    return render


def myinfer(infer_dataset, render, save_dir, epoch=0):
    render.net.eval()
    psnr_wMask_list = []
    psnr_woMask_list = []
    ssim_list = []

    rendering_dir = f"{save_dir}/{epoch}/rendering"


    if not os.path.exists(rendering_dir):
        os.makedirs(rendering_dir, exist_ok=True)

    for angle in tqdm(range(0, 360, 36)):
        for batch_idx, batch in enumerate(infer_dataset):
            # batch['frame'][...] = 0
            real_frame = batch["frame"][0]
            ## fix frame code
            # batch["frame"][...] = 50
            if "save_name" in batch:
                save_name = batch["save_name"][0]
            else:
                frame_index = batch['frame_index'].item()
                view_index = batch['cam_ind'].item()
                save_name = 'frame{:04d}_view{:04d}'.format(frame_index, view_index)
            # [-0.5992459, -0.9855767,  1.04382  ] left hand 313
            # [ 0.18649693, -0.14180326,  1.7103844 ] head 313

            render.net.set_rot_center(torch.Tensor([ 0.18649693, -0.14180326,  1.7103844 ] )[None])
            render.net.set_rot(torch.Tensor(angle2rot(angle)))
            results = render.render_view(batch)
            color_img_0 = results["coarse_color"]        

            rendering_path = os.path.join(rendering_dir, "%05d.jpg"%angle)
            
            rendering = color_img_0.numpy() * 255
            cv2.imwrite(rendering_path, rendering)

    print("epoch", epoch)
    return


def save_img(imgs, img_dir):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    for idx in range(len(imgs)):
        img_path = os.path.join(img_dir, "%06d.jpg" % idx)
        cv2.imwrite(img_path, imgs[idx])


def img2vid(img_dir, output_path):
    (
        ffmpeg.input("%s/*.jpg" % img_dir, pattern_type="glob", framerate=15)
        .output(output_path).run()
    )


def angle2rot(angle):
    radian = np.pi * angle/180
    rot = [[np.cos(radian), -np.sin(radian)],
            [np.sin(radian), np.cos(radian)]]

    return np.array(rot)



if __name__ == "__main__":
    save_root = "./lighting_vis"
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
    # img_dir = os.path.join(save_dir, "imgs")
    # os.makedirs(img_dir, exist_ok=True)
    # Load config
    training_config = args.config
    assert os.path.exists(training_config), "training config does not exist."
    cfg.merge_from_file(training_config)

    
    from dataloader.zju_mocap_dataset import Mocap_view
    novel_pose_dataset = Mocap_view("CoreView_313",ratio = 0.5, begin=0, end = 0,train_views=[] , 
    train_max_frame=2000 ,interval=1, vis_views=[0], data_dir = cfg.DATASETS.ZJU_MOCAP_PATH)
    print('length:',len(novel_pose_dataset))


    novel_pose_dataloader = DataLoader(
        novel_pose_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    render = load_render(
        args.ckpt, cfg, canonical_vertex=novel_pose_dataset.canonical_vertex
    )

    out1 = myinfer(
        novel_pose_dataloader,
        render,
        save_dir=save_dir,
        epoch=epoch,
    )
    import ffmpeg
    img2vid(f'{save_dir}/{epoch}/rendering', os.path.join(save_dir,f"{args.exp}.mp4"))