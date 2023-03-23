from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import cv2
import glob2 as gb
import pickle
from utils import rays_utils, camera_utils
import torch


class Mocap_Base(Dataset):
    def __init__(self, human="CoreView_377", ratio=0.5, nrays=500, performer = "S8", zju_data_dir = "", h36m_data_dir = ""):
        super(Mocap_Base, self).__init__()

        
        data_root = f"{zju_data_dir}/{human}"
        self.human = human
        self.data_root = data_root
        
        use_x_pose = True
        self.use_x_pose = use_x_pose
        self.gender = "neutral"

        self.ratio = ratio

        # img_dir
        if self.human in ["CoreView_313", "CoreView_315"]:
            ann_file = os.path.join(data_root, "annots.npy")
            cams = camera_utils.load_cam(ann_file)
        else:
            cams = camera_utils.load_cameras(
                data_root
            )  # ['i']['K', 'invK', 'RT', 'R', 'T', 'P', 'dist']
        self.cams = cams
        # import pdb;pdb.set_trace()
        self.nrays = nrays
        self.smpl_dir = os.path.join(data_root, "new_params")
        self.vertices_dir = os.path.join(data_root, "new_vertices")

        if "CoreView" in performer:
            self.canonical_vertex = torch.from_numpy(
                np.load(os.path.join(f"{zju_data_dir}/{performer}", "X_smpl_vertices.npy"))
            ).squeeze()
        else:
            self.canonical_vertex = torch.from_numpy(
                np.load(os.path.join(f"{h36m_data_dir}/{performer}/Posing/lbs", "X_smpl_vertices.npy"))
            ).squeeze()

        

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.vertices_dir, f"{i}.npy")
        xyz = np.load(vertices_path).astype(np.float32).squeeze()

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)

        if self.mode == "train":
            min_xyz -= 0.1
            max_xyz += 0.1
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05

        world_bounds = np.stack([min_xyz, max_xyz], axis=0)
        # transform smpl from the world coordinate to the smpl coordinate
        param_path = os.path.join(self.smpl_dir, f"{i}.npy")
        params = np.load(param_path, allow_pickle=True).item()
        Rh = params["Rh"]
        Rh = cv2.Rodrigues(Rh)[0]
        Th = params["Th"]
        poses = params["poses"].reshape(-1, 3)

        if self.use_x_pose:
            poses[1, 2] -= 0.6
            poses[2, 2] += 0.6

        return poses, xyz, world_bounds, Rh, Th

    def __getitem__(self, idx):
        img_path = self.all_img_path[0]
        img = cv2.imread(img_path)
        if self.human in ["CoreView_313", "CoreView_315"]:
            # frame_name = int(os.path.basename(t_path).split("_")[4])
            cam_idx = img_path.split("/")[-2]
        else:
            # frame_name = int(img_path.split("/")[-1].split(".")[0])
            cam_idx = img_path.split("/")[-2]
        frame_name = idx *4

        K = np.array(self.cams[cam_idx]["K"])
        D = np.array(self.cams[cam_idx]["dist"])

        img = cv2.undistort(img, K, D)

        msk_fg, msk_cihp = self.get_mask(img_path)

        img = img * msk_fg

        if self.ratio != 1:
            K[:2] = K[:2] * self.ratio
            img = cv2.resize(
                img, (0, 0), fx=self.ratio, fy=self.ratio, interpolation=cv2.INTER_AREA
            )
            msk_fg = cv2.resize(
                msk_fg,
                (0, 0),
                fx=self.ratio,
                fy=self.ratio,
                interpolation=cv2.INTER_NEAREST,
            )
            msk_cihp = cv2.resize(
                msk_cihp,
                (0, 0),
                fx=self.ratio,
                fy=self.ratio,
                interpolation=cv2.INTER_NEAREST,
            )

        img = img / 255.0
        R = np.array(self.cams[cam_idx]["R"])
        T = np.array(self.cams[cam_idx]["T"])
        H, W = int(img.shape[0]), int(img.shape[1])
        # img = np.zeros((1024,1024,3))
        poses, xyz, world_bounds, Rh, Th = self.prepare_input(frame_name)
        
        (
            rgb,
            ray_o,
            ray_d,
            near,
            far,
            coord,
            mask_at_box,
            bound_mask,
        ) = rays_utils.my_sample_ray(img, K, R, T, world_bounds, msk_cihp, self.nrays)

        # t = mask_at_box.reshape(H,W).astype(np.uint8)
        # import pdb;pdb.set_trace()

        # cv2.imshow('1', t*255)
        # cv2.imshow('2', bound_mask*255)
        # cv2.waitKey()
        # import pdb;pdb.set_trace()

        occupancy = msk_fg[coord[:, 0], coord[:, 1]]

        # import pdb;pdb.set_trace()
        if self.human in ["CoreView_313", "CoreView_315"]:
            cam_idx = int(cam_idx.split(" ")[1].strip('()'))-1
        else:
            cam_idx = int(cam_idx.split("_")[1][1:])-1


        frame_tmp = int(frame_name)-1 if self.human in ["CoreView_313", "CoreView_315"] else int(frame_name)
        ret = {
            "img": img,
            "coord": coord,
            "rgb": rgb,
            "occupancy": occupancy,
            "ray_o": ray_o,
            "ray_d": ray_d,
            "near": near,
            "far": far,
            "mask_at_box": mask_at_box,
            "poses": poses,
            "xyz": xyz,
        }
        meta = {
            "bounds": world_bounds,
            "mybound_mask":bound_mask,
            "Rh": Rh,
            "Th": Th,
            "R": R,
            "T": T,
            "frame": frame_tmp,
            "cam_ind": cam_idx,
            'save_name': "frame%04d_view%04d"%(frame_tmp, cam_idx)
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return self.len

    def get_mask(self, img_path):
        tmp = img_path.split("/")
        tmp.insert(-2, "mask_cihp")
        camera_view = tmp[-2]
        msk_path = "/".join(tmp)[:-4] + ".png"
        msk_cihp = cv2.imread(msk_path)

        if len(msk_cihp.shape) == 3:
            msk_cihp = msk_cihp[..., 0]
        msk_fg_origin = (msk_cihp != 0).astype(np.uint8)
        msk_fg_origin = cv2.undistort(
            msk_fg_origin, self.cams[camera_view]["K"], self.cams[camera_view]["dist"]
        )
        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_fg = cv2.dilate(msk_fg_origin.copy(), kernel)

        # import pdb;pdb.set_trace()
        # cv2.imshow('1', np.repeat(msk_fg_origin[...,None],3,axis =2)*255)
        # cv2.imshow('2', np.repeat(msk_fg[...,None],3,axis =2)*255)
        # cv2.waitKey()
        return msk_fg[..., None], msk_cihp[..., None]




class Mocap_view(Mocap_Base):
    def __init__(
        self,
        human="CoreView_377",
        ratio=0.5,
        begin=0,
        end=300,
        train_views=[0, 6, 12, 18],
        train_max_frame=300,
        interval=30,
        vis_views=None,
        performer = "S8",
        zju_data_dir = "", 
        h36m_data_dir = ""
    ):
        super(Mocap_view, self).__init__(human, ratio, nrays=-1, performer = performer, zju_data_dir = zju_data_dir, h36m_data_dir = h36m_data_dir)
        views = []
        self.vis_view = vis_views
        if vis_views is None:
            if human in ["CoreView_313", "CoreView_315"]:
                for view in range(len(self.cams.keys())):
                    if view not in train_views:
                        if view in [19, 20]:
                            view += 2
                        views.append(f"Camera ({view+1})")
            else:
                for view in range(len(self.cams.keys())):
                    if view not in train_views:
                        views.append(f"Camera_B{view+1}")
        else:
            if human in ["CoreView_313", "CoreView_315"]:
                for view in vis_views:
                    if view in [19, 20]:
                        view += 2
                    views.append(f"Camera ({view+1})")
            else:
                for view in vis_views:
                    views.append(f"Camera_B{view+1}")

        all_img = []
        for view in views:
            img_view = []
            img_path_view = gb.glob(os.path.join(self.data_root, f"{view}", "*.jpg"))
            for img_path in img_path_view:
                if self.human in ["CoreView_313", "CoreView_315"]:
                    i = int(os.path.basename(img_path).split("_")[4])
                    frame_index = i - 1
                else:
                    i = int(os.path.basename(img_path)[:-4])
                    frame_index = i
                if frame_index >= begin and frame_index <= end:
                    img_view.append(img_path)
            if self.human in ["CoreView_313", "CoreView_315"]:
                img_view = sorted(img_view, key=lambda name: int(name.split("_")[6]))
            else:
                img_view = sorted(
                    img_view, key=lambda name: int(name.split("/")[-1][:-4])
                )
            all_img += img_view[::interval]
        self.all_img_path = all_img
        self.len = len(all_img)*10
        self.train_max_frame = train_max_frame
        self.mode = "infer"
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        tmp = super().__getitem__(idx)
        if self.vis_view is None:
            tmp["frame"] = torch.randint(0, self.train_max_frame, [1])[0]
        return tmp





if __name__ == "__main__":
    coreviews = [
        "CoreView_313",
        "CoreView_315",
        "CoreView_377",
        "CoreView_386",
        "CoreView_387",
        "CoreView_390",
        "CoreView_392",
        "CoreView_394",
    ]
    coreview = coreviews[2]
    # dataset = Mocap(human=coreview, ratio=0.5, nrays=500, begin=0, end=500)
    # dataset = Mocap_view(human=coreview, ratio=0.5, begin=400, end=1400)
    # dataset = Mocap_infer(human=coreview, ratio=0.5, begin=0, end=500)
    # dataset = Mocap_view("CoreView_377",ratio = 0.5, begin=0, end = 1,train_views=[] , train_max_frame=2000 ,interval=30, vis_views=[0, 6, 12, 18])
    performer = "S8"
    dataset = Mocap_view("CoreView_393",ratio = 1, begin=0, end = 10000,train_views=[] , train_max_frame=2000 ,interval=2, vis_views=[9], performer = performer)
    print(len(dataset))
    # dataloader = DataLoader(dataset, 1, False, num_workers=0)
    # for idx, data in enumerate(dataloader):
    #     print(idx, "/", len(dataloader))
    #     for key in data.keys():
    #         if type(data[key]) == list:
    #             print(key, len(data[key]))
    #         else:
    #             print(key, data[key].shape)

    #     print(data['save_name'])
        # print()
        # import pdb;pdb.set_trace()
        # break

    