import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
import sys 
sys.path.append("..")
from utils import h36m_utils
import torch
# from plyfile import PlyData




class H36M_test(data.Dataset):
    def __init__(self, cfg, data_root, human, ann_file, split, nrays =2000, test_novel_pose=False, is_eval = False, is_formal = True):
        super(H36M_test, self).__init__()
        self.cfg = cfg
        self.test_novel_pose = test_novel_pose
        self.data_root = data_root
        self.human = human
        self.split = split
        self.is_eval = is_eval
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])
        if len(cfg.test_view) == 0:
            test_view = [
                i for i in range(num_cams) if i not in cfg.training_view
            ]
            if len(test_view) == 0:
                test_view = [0]
        else:
            test_view = cfg.test_view
        view = cfg.training_view if split == 'train' else test_view

        i = cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame
        self.i_intv = i_intv
        if test_novel_pose:
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
            if is_formal:
                ni = cfg.num_eval_frame
            else:
                ni = cfg.my_num_eval_frame
        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::30]
        ]).ravel()
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.num_cams = len(view)

        self.lbs_root = os.path.join(self.data_root, 'lbs')

        use_x_pose = True
        if use_x_pose:
            joints = np.load(os.path.join(self.lbs_root, 'X_smpl_joints.npy'))[0]
        else:
            joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))
        self.nrays = nrays

        self.canonical_vertex = torch.from_numpy(
            np.load(os.path.join(self.data_root, 'lbs', "X_smpl_vertices.npy"))
        ).squeeze()

    def get_mask(self, index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index])[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, self.ims[index].replace(
                'images', 'mask'))[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        if len(msk_cihp.shape) == 3:
            msk_cihp = msk_cihp[..., 0]
        tmp = msk_cihp
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp
        orig_msk = msk.copy()

        if not self.is_eval:
            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk_erode = cv2.erode(msk.copy(), kernel)
            msk_dilate = cv2.dilate(msk.copy(), kernel)
            msk[(msk_dilate - msk_erode) == 1] = 100

        return msk, orig_msk, tmp

    def prepare_input(self, i):
        # read xyz in the world coordinate system
        vertices_path = os.path.join(self.data_root, self.cfg.vertices,
                                     '{}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, self.cfg.params,
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)
        Th = params['Th'].astype(np.float32)

        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)

        # calculate the skeleton transformation
        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A = h36m_utils.get_rigid_transformation(poses, joints, parents)

        pbw = np.load(os.path.join(self.lbs_root, 'bweights/{}.npy'.format(i)))
        pbw = pbw.astype(np.float32)

        return wxyz, pxyz, A, pbw, R, Th, poses

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.ims[index])
        img = cv2.imread(img_path).astype(np.float32) / 255.
        msk, orig_msk, msk_cihp = self.get_mask(index)

        H, W = img.shape[:2]
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H),
                              interpolation=cv2.INTER_NEAREST)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)
        orig_msk = cv2.undistort(orig_msk, K, D)

        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * self.cfg.ratio), int(img.shape[1] * self.cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H),
                              interpolation=cv2.INTER_NEAREST)
        
        img[orig_msk == 0] = 0

        # cv2.imshow('1', img)
        # cv2.waitKey()
        K[:2] = K[:2] * self.cfg.ratio

        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i

        # read v_shaped
        vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        tpose = np.load(vertices_path).astype(np.float32)
        tbounds = h36m_utils.get_bounds(tpose)
        tbw = np.load(os.path.join(self.lbs_root, 'tbw.npy'))
        tbw = tbw.astype(np.float32)

        wpts, ppts, A, pbw, Rh, Th, poses = self.prepare_input(i)

        pbounds = h36m_utils.get_bounds(ppts)
        wbounds = h36m_utils.get_bounds(wpts)

        border = 10
        kernel = np.ones((border, border), np.uint8)
        msk_cihp_eroded = cv2.erode(msk_cihp.copy(), kernel)
        rgb, ray_o, ray_d, near, far, coord, mask_at_box = h36m_utils.sample_ray_h36m(
            img,  msk, msk_cihp_eroded,K, R, T, wbounds, self.nrays, self.split)

        orig_msk = h36m_utils.crop_mask_edge(orig_msk)
        msk_tmp = (orig_msk!=0).astype(np.uint8)
        occupancy = msk_tmp[coord[:, 0], coord[:, 1]]

        # cv2.imshow('msk',msk_tmp*255)
        # cv2.imshow('1',img)
        # cv2.waitKey()
        # nerf
        ret = {
            'img': img,
            "coord": coord,
            'rgb': rgb,
            'occupancy': occupancy,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box
        }

        # blend weight
        meta = {
            'A': A,
            'poses': poses,
            'xyz': wpts,
            "bounds": wbounds,
            'Rh': Rh,
            'Th': Th,
            # 'pbw': pbw,
            # 'tbw': tbw,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': tbounds
        }
        ret.update(meta)

        # transformation
        meta = {'R': R, 'T':T, 'H': H, 'W': W}
        ret.update(meta)

        latent_index = index // self.num_cams
        if self.test_novel_pose:
            latent_index = self.cfg.num_train_frame - 1
        meta = {
            'latent_index': latent_index,
            'frame_index': frame_index,
            'cam_ind': cam_ind,
            'frame': frame_index//self.i_intv
        }

        if self.test_novel_pose:
            meta['frame'] = torch.randint(0, self.cfg.num_train_frame, [1])[0]
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)


if __name__=="__main__":
    pass