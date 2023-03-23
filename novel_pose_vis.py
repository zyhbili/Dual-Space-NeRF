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
    render.eval()
    img_dir = f"{save_dir}/{epoch}/img"
    rendering_dir = f"{save_dir}/{epoch}/rendering"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir, exist_ok=True)
    if not os.path.exists(rendering_dir):
        os.makedirs(rendering_dir, exist_ok=True)

    # with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(infer_dataset)):
        # batch['frame'][...] = 0
        real_frame = batch["frame"][0]
        ## fix frame code
        # batch["frame"][...] = torch.randint(0, 59, [1])[0]
        batch["frame"][...] = 50
        if "save_name" in batch:
            save_name = batch["save_name"][0]
        else:
            frame_index = batch['frame_index'].item()
            view_index = batch['cam_ind'].item()
            save_name = 'frame{:04d}_view{:04d}'.format(frame_index, view_index)
        # import pdb;pdb.set_trace()
        results = render.render_view(batch)
        color_img_0 = results["coarse_color"]
        depth_img_0 = results["coarse_depth"]


        img_path = os.path.join(img_dir, "%05d.jpg"%batch_idx)
        rendering_path = os.path.join(rendering_dir, "%05d.jpg"%batch_idx)
       
        rendering = color_img_0.numpy() * 255
        gt = batch["img"].squeeze().numpy() * 255
        cat_img = np.concatenate((rendering, gt), axis=1)
        cv2.imwrite(img_path, cat_img)
        cv2.imwrite(rendering_path, rendering)

    return None


def save_img(imgs, img_dir):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    for idx in range(len(imgs)):
        img_path = os.path.join(img_dir, "%06d.jpg" % idx)
        cv2.imwrite(img_path, imgs[idx])


def img2vid(img_dir, output_path):
    (
        ffmpeg.input("%s/*.jpg" % img_dir, pattern_type="glob", framerate=60)
        .output(output_path).run()
    )


if __name__ == "__main__":
    save_root = "./motion_transfer"
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


    zju_data_dir = cfg.DATASETS.ZJU_MOCAP_PATH
    h36m_data_dir = cfg.DATASETS.H36M_PATH


    # extreme pose
    from dataloader.zju_novel_pose_dataset import Mocap_view
    performer = "CoreView_313"
    novel_pose_dataset = Mocap_view("CoreView_313",ratio = 1, begin=0, end = 100000,train_views=[] , train_max_frame=2000 ,interval=4, vis_views=[9], performer = performer
                            , zju_data_dir = zju_data_dir, h36m_data_dir = h36m_data_dir)
    novel_pose_dataset.smpl_dir = os.path.join(f"novelpose_examples/CoreView_313_op3", "new_params")
    novel_pose_dataset.vertices_dir = os.path.join(f"novelpose_examples/CoreView_313_op3", "new_vertices")
        

    # if cfg.DATASETS.TYPE == "h36m":
    #     from dataloader.zju_novel_pose_dataset import Mocap_view

    #     motion_seq = "CoreView_393"
    #     performer = cfg.DATASETS.HUMAN
    #     novel_pose_dataset = Mocap_view(motion_seq, ratio = 0.5, begin=0, end = 10000,train_views=[] , train_max_frame=2000 ,interval=2, 
    #     vis_views=[6], performer = performer, zju_data_dir = zju_data_dir, h36m_data_dir = h36m_data_dir)
    #     # Novel Pose Animation Example. Using the motion seq from zju-mocap to animate the performer of h36m
    #     # ！！Body shape is different from origin performer
    #     # ！！To ensure the correct SMPL body shape, you need to generate the "new_vertices" and specify the path like follows:
    #     # novel_pose_dataset.vertices_dir = "/group/projects/smpl_nerf/novel_poses/CoreView_313_translation/new_vertices"


    # if cfg.DATASETS.TYPE == "zju_mocap":
    #     motion_seq = "S9"
    #     from dataloader.novel_poses_dataset import get_novel_pose_dataset
    #     novel_pose_dataset = get_novel_pose_dataset(performer="CoreView_377", motion_seq = motion_seq, zju_data_dir = zju_data_dir, h36m_data_dir = h36m_data_dir)
    #     # Novel Pose Animation Example. Using the motion seq from h36m to animate the performer of zju=mocap
    #     # ！！Body shape is different from origin performer
    #     # ！！To ensure the correct SMPL body shape, you need to generate the "new_vertices" and specify the path like follows:
    #     # novel_pose_dataset.vertices_dir = "/group/projects/smpl_nerf/novel_poses/CoreView_313_translation/new_vertices"


    print('length:',len(novel_pose_dataset))
    dataloader = DataLoader(novel_pose_dataset, 1, False)

    novel_pose_dataloader = DataLoader(
        novel_pose_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    render = load_render(
        args.ckpt, cfg, canonical_vertex=novel_pose_dataset.canonical_vertex
    )
    render.net.nerf.w = 0

    render.net.set_light_center(
        torch.tensor(cfg.TEST.light_center)
    )  
    # Note that the center of the z-axis in ZJU and H36M is not always consistent (+-1), so it may be necessary to manually adjust the light center.

    try:
        out1 = myinfer(
            novel_pose_dataloader,
            render,
            save_dir=save_dir,
            epoch=epoch,
        )
    except:
        import ffmpeg
        img2vid(f'{save_dir}/{epoch}/img', os.path.join(save_dir,f"{args.exp}.mp4"))
        img2vid(f'{save_dir}/{epoch}/rendering', os.path.join(save_dir,f"{args.exp}_rendering.mp4"))
