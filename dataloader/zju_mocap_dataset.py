from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, sys
import cv2
import glob2 as gb
sys.path.append("..")
from utils import rays_utils, camera_utils, smpl_utils
import torch


class Mocap_Base(Dataset):
    def __init__(self, human="CoreView_377", ratio=0.5, nrays=500, data_dir = None):
        super(Mocap_Base, self).__init__()

        data_root = f"{data_dir}/{human}"
        self.human = human
        self.data_root = data_root
        self.smpl_dir = os.path.join(data_root, "new_params")
        self.vertices_dir = os.path.join(data_root, "new_vertices")
        
        use_x_pose = True
        self.use_x_pose = use_x_pose
        if use_x_pose:
            self.joints = np.load(os.path.join(data_root, "X_smpl_joints.npy"))[0]
        else:  ##T_pose
            self.joints = np.load(os.path.join(data_root, "T_smpl_joints.npy"))[0]

        # self.gender = "neutral"
        # self.smpl = smpl_utils.load_bodydata(gender=self.gender)

        # kintree_table = self.smpl["kintree_table"]
        # parents = kintree_table[0].astype(np.int64)
        # parents[0] = -1
        # self.parents = parents

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
        self.nrays = nrays
        # load canonical model: option
        self.canonical_vertex = torch.from_numpy(
            np.load(os.path.join(self.data_root, "X_smpl_vertices.npy"))
        ).squeeze()

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.vertices_dir, f"{i}.npy")
        xyz = np.load(vertices_path).astype(np.float32)
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

        A = -1 ## TODO redundant code

        return poses, xyz, world_bounds, A, Rh, Th

    def __getitem__(self, idx):
        img_path = self.all_img_path[idx]
        img = cv2.imread(img_path)
        if self.human in ["CoreView_313", "CoreView_315"]:
            frame_name = int(os.path.basename(img_path).split("_")[4])
            cam_idx = img_path.split("/")[-2]
        else:
            frame_name = int(img_path.split("/")[-1].split(".")[0])
            cam_idx = img_path.split("/")[-2]

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

        poses, xyz, world_bounds, A, Rh, Th = self.prepare_input(frame_name)
        
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
            "A": A,
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


class Mocap(Mocap_Base):
    def __init__(
        self,
        human="CoreView_377",
        ratio=0.5,
        nrays=500,
        begin=0,
        end=300,
        train_views=[0, 6, 12, 18],
        data_dir = None
    ):
        super(Mocap, self).__init__(human, ratio, nrays, data_dir)
        views = []
        if human in ["CoreView_313", "CoreView_315"]:
            for view in train_views:
                views.append(f"Camera ({view+1})")
        else:
            for view in train_views:
                views.append(f"Camera_B{view+1}")
        all_img = []
        for view in views:
            img_path = gb.glob(os.path.join(self.data_root, f"{view}", "*.jpg"))
            all_img += img_path
        img_train = []

        for img_path in all_img:
            if self.human in ["CoreView_313", "CoreView_315"]:
                i = int(os.path.basename(img_path).split("_")[4])
                frame_index = i - 1
            else:
                i = int(os.path.basename(img_path)[:-4])
                frame_index = i
            if frame_index >= begin and frame_index <= end:
                img_train.append(img_path)
        self.all_img_path = img_train
        self.len = len(img_train)
        self.mode = "train"

    def __len__(self):
        return self.len


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
        data_dir = None
    ):
        super(Mocap_view, self).__init__(human, ratio, nrays=-1, data_dir = data_dir)
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
        self.len = len(all_img)
        self.train_max_frame = train_max_frame
        self.mode = "infer"
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        tmp = super().__getitem__(idx)
        if self.vis_view is None:
            tmp["frame"] = torch.randint(0, self.train_max_frame, [1])[0]
        return tmp


class Mocap_infer(Mocap_Base):
    def __init__(
        self,
        human="CoreView_377",
        ratio=0.5,
        begin=0,
        end=300,
        train_views=[0, 6, 12, 18],
        train_max_frame=300,
        interval=30,
        eval_begin_frame = 60,
        novel_pose = False
    ):
        super(Mocap_infer, self).__init__(human, ratio, nrays=-1)
        views = []
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
        
        all_img_train = []
        all_img_val = []

        for view in views:
            img_train = []
            img_val = []
            img_path_view = gb.glob(os.path.join(self.data_root, f"{view}", "*.jpg"))
            for img_path in img_path_view:
                if self.human in ["CoreView_313", "CoreView_315"]:
                    i = int(os.path.basename(img_path).split("_")[4])
                    frame_index = i - 1
                else:
                    i = int(os.path.basename(img_path)[:-4])
                    frame_index = i

                if frame_index >= begin and frame_index < eval_begin_frame:
                    img_train.append(img_path)
                if frame_index >= eval_begin_frame and frame_index <= end:
                    img_val.append(img_path)
            if self.human in ["CoreView_313", "CoreView_315"]:
                img_train = sorted(img_train, key=lambda name: int(name.split("_")[6]))
                img_val = sorted(img_val, key=lambda name: int(name.split("_")[6]))
            else:
                img_train = sorted(img_train, key=lambda name: int(name.split("/")[-1][:-4]))
                img_val = sorted(img_val, key=lambda name: int(name.split("/")[-1][:-4]))
            all_img_train += img_train[::interval]
            all_img_val += img_val[::interval]

        if novel_pose:
            self.all_img_path =  all_img_val
        else:
            self.all_img_path = all_img_train
        self.len = len(self.all_img_path)
        self.train_max_frame = train_max_frame
        self.mode = "infer"
        self.novel_pose = novel_pose
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        tmp = super().__getitem__(idx)
        if self.novel_pose:
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

    dataset = Mocap_view("CoreView_377",ratio = 0.5, begin=0, end = 1,train_views=[] , train_max_frame=2000 ,interval=30, vis_views=[0, 6, 12, 18])
    # dataset = Mocap_view("CoreView_313",ratio = 0.5, begin=0, end = 50,train_views=[] , train_max_frame=2000 ,interval=30, vis_views=[0])

    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    for idx, data in enumerate(dataloader):


        print(idx, "/", len(dataloader))

        print((data["ray_d"]**2).sum(-1))
     

    