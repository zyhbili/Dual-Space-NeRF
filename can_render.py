import torch
from utils.nerf_net_utils import raw2outputs
import numpy as np
from utils.render_utils import (
    post_process,
    get_closest_mesh,
    get_transparent_mask,
)
from utils.geo_utils import project_point2mesh, barycentric_map2can
from utils.pts_utils import uniform_sampling, geometry_guided_ray_marching
from utils.smpl_utils import load_bodydata


class Renderer:
    def __init__(self, net, fine_net=None, cfg=None, canonical_vertex=None):
        self.net = net
        self.cfg = cfg

        self.fine_net = fine_net
        self.canonical_vertex = canonical_vertex
        self.load_body_model(gender = "neutral", body_model = "smpl" ,  model_path = cfg.DATASETS.SMPL_PATH)
        self.sample_points_mode = cfg.MODEL.sample_points_mode
        print("sampling mode:",self.sample_points_mode)


    def train(self):
        self.net.training = True
        self.net.train()
        if self.fine_net != None:
            self.fine_net.training = True
            self.fine_net.train()

    def eval(self):
        self.net.training = False
        self.net.eval()
        if self.fine_net != None:
            self.fine_net.training = False
            self.fine_net.eval()

    def get_sampling_points(self, ray_o, ray_d, near, far, xyz, mode="GG"):
        # calculate the steps for each ray
        if mode == "uniform":
            pts, z_vals = uniform_sampling(
                ray_o,
                ray_d,
                self.cfg.MODEL.COARSE_RAY_SAMPLING,
                near,
                far,
                self.cfg.MODEL.perturb,
                self.net.training,
            )
        elif mode == "GG":
            pts, z_vals = geometry_guided_ray_marching(
                ray_o,
                ray_d,
                self.cfg.MODEL.COARSE_RAY_SAMPLING,
                near,
                far,
                xyz,
                self.cfg.MODEL.perturb,
                self.net.training,
            )
        return pts, z_vals

    def batchify_pts(
        self,
        pts,
        rays,
        z_vals,
        frame_idx,
        chunk=1024 * 32,
        net=None,
        batch_info=None,
    ):
        """Render pts in smaller minibatches to avoid OOM."""
        all_ret = {}
        if net == None:
            net = self.net
        transparent_mask = batch_info["transparent_mask"]
        for i in range(0, pts.shape[0], chunk):
            ret = self.render_rays(
                pts[i : i + chunk],
                rays[i : i + chunk],
                z_vals[i : i + chunk],
                frame_idx[i : i + chunk],
                net=net,
                transparent_mask=transparent_mask[i : i + chunk],
                batch_info=batch_info,
            )
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def render_rays(
        self, pts, rays, z_vals, frame_idx, net, transparent_mask=None, batch_info=None
    ):
        pts = pts.cuda()
        rays = rays.cuda()
        z_vals = z_vals.cuda()
        frame_idx = frame_idx.cuda()
        rays_d = rays[:, 0, :3]
        if self.net.training:
            noise_std = self.cfg.MODEL.raw_noise_std
        else:
            noise_std = 0
        if self.cfg.MODEL.TYPE == "nerf":
            B, sp = pts.shape[:2]
            pts = pts.reshape(-1, 6)  ## world xyz 3 cat canonical xyz 3
            rays = rays.reshape(-1, 6)
            rgbs, density, others = net(pts, rays, frame_idx, batch_info=batch_info)

            raw = torch.cat([rgbs, density], dim=-1)
            raw = raw.reshape(B, sp, -1)

            t = raw[transparent_mask]
            t[..., -1] = 0
            raw[transparent_mask] = t

            rgb_map, disp_map, acc_map, weights, depth_map, others = raw2outputs(
                raw, z_vals, rays_d, noise_std, False
            )
            ret = {
                "color": rgb_map,
                "disp_map": disp_map,
                "acc_map": acc_map,
                "depth_map": depth_map,
                "weights": weights,
                "z_vals": z_vals,
            }

        return ret
     

    def render(self, batch):
        ray_o = batch["ray_o"].cuda()
        ray_d = batch["ray_d"].cuda()
        near = batch["near"].cuda()
        far = batch["far"].cuda()
        pts_world, z_vals = self.get_sampling_points(
            ray_o,
            ray_d,
            near,
            far,
            batch["xyz"],
            mode=self.cfg.MODEL.sample_points_mode,
        )  # B, ray, p, 3,
        sh = pts_world.shape  # B, ray, p, 3

        pts_world_smpl_can, rays, transparent_mask = self.w2l(
            pts_world, ray_o, ray_d, batch
        )

        batch["transparent_mask"] = transparent_mask.reshape(-1, sh[2])
        batch["canonical_model"] = self.canonical_model
        batch["face_idx"] = self.face_idx
        frame_idx = batch["frame"][..., None, None].repeat(1, sh[1], sh[2])
        ret_coarse = self.batchify_pts(
            pts_world_smpl_can,
            rays,
            z_vals.reshape(-1, sh[2]),
            frame_idx.reshape(-1, sh[2]),
            batch_info=batch,
        )

        return {"coarse": ret_coarse}


    ####### _view suffix used to visualize the whole img
    def batchify_rays_view(self, ray_o, ray_d, near, far, batch, chunk=1024 * 32):
        """Render rays in smaller minibatches to avoid OOM."""
        all_ret_coarse = {}
        all_ret_fine = {}

        for i in range(0, ray_o.shape[1], chunk):
            pts_world, z_vals = self.get_sampling_points(
                ray_o[:, i : i + chunk],
                ray_d[:, i : i + chunk],
                near[:, i : i + chunk],
                far[:, i : i + chunk],
                batch["xyz"],
                mode=self.cfg.MODEL.sample_points_mode,
            )  # B, ray, p, 3
            torch.cuda.empty_cache()
            pts_smpl_can, rays, transparent_mask = self.w2l(
                pts_world,
                ray_o[:, i : i + chunk],
                ray_d[:, i : i + chunk],
                batch,
            )
            sh = pts_world.shape

            batch["transparent_mask"] = transparent_mask.reshape(-1, sh[2])
            batch["canonical_model"] = self.canonical_model
            batch["face_idx"] = self.face_idx
            frame_idx = batch["frame"][..., None, None].repeat(1, sh[1], sh[2])
            ret_coarse = self.batchify_pts(
                pts_smpl_can,
                rays,
                z_vals.reshape(-1, sh[2]),
                frame_idx.reshape(-1, sh[2]),
                batch_info=batch,
            )

            for k in ret_coarse:
                if k not in all_ret_coarse:
                    all_ret_coarse[k] = []
                all_ret_coarse[k].append(ret_coarse[k].detach().cpu())

            if self.cfg.MODEL.FINE_RAY_SAMPLING > 0:
                pts_world, z_vals = self.resampling(
                    ret_coarse["z_vals"],
                    ret_coarse["weights"],
                    ray_o[:, i : i + chunk].reshape(-1, 3),
                    ray_d[:, i : i + chunk].reshape(-1, 3),
                )
                pts_world = pts_world.reshape(sh[0], sh[1], -1, sh[3])

                pts_smpl_can, rays, transparent_mask = self.w2l(
                    pts_world,
                    ray_o[:, i : i + chunk],
                    ray_d[:, i : i + chunk],
                    batch,
                )
                ret_fine = self.batchify_pts(
                    pts_smpl_can,
                    rays,
                    z_vals,
                    frame_idx.reshape(-1, sh[2]),
                    net=self.fine_net,
                    batch_info=batch,
                )

                for k in ret_fine:
                    if k not in all_ret_fine:
                        all_ret_fine[k] = []
                    all_ret_fine[k].append(ret_fine[k].cpu())

        all_ret_coarse = {k: torch.cat(all_ret_coarse[k], 0) for k in all_ret_coarse}
        if self.cfg.MODEL.FINE_RAY_SAMPLING > 0:
            all_ret_fine = {k: torch.cat(all_ret_fine[k], 0) for k in all_ret_fine}

        return all_ret_coarse, all_ret_fine

    ####### _view suffix used to visualize the whole img
    def render_view(self, batch):
        ray_o = batch["ray_o"].cuda()
        ray_d = batch["ray_d"].cuda()
        near = batch["near"].cuda()
        far = batch["far"].cuda()
        batch["xyz"] = batch["xyz"].cuda()
        sh = ray_o.shape
        # with torch.no_grad():
        coarse, fine = self.batchify_rays_view(
            ray_o, ray_d, near, far, batch, int(1024 * 3.0)  # 3 20
        )
        _, H, W, _ = batch["img"].shape
        coarse_color = post_process(coarse["color"], batch["mask_at_box"][0], (H, W, 3))
        coarse_disp_map = post_process(
            coarse["disp_map"][..., None], batch["mask_at_box"][0], (H, W, 1)
        )
        coarse_acc_map = post_process(
            coarse["acc_map"][..., None], batch["mask_at_box"][0], (H, W, 1)
        )
        coarse_depth_map = post_process(
            coarse["depth_map"][..., None], batch["mask_at_box"][0], (H, W, 1)
        )

        tmp = {
            "coarse_color": coarse_color,
            "coarse_disp": coarse_disp_map,
            "coarse_acc": coarse_acc_map,
            "coarse_depth": coarse_depth_map,
        }
       
        return tmp

    def query_volume(self, pts, code_idx, transparent_mask=None, batch_info={}):
        """Args:
            pts: (batch, num_pts, 3)

        Output:
            density: (batch, num_pts, 1)
        """
        B, N = pts.shape[:2]
        ray_input = pts.expand([2, -1, -1])  # no need,but for compatibility
        code_idx = code_idx.cuda().reshape(B, 1).repeat(1, N)
        density = self.net(
            pts, ray_input.reshape(-1, 3), code_idx, batch_info, density_only=True
        )
        if transparent_mask != None:
            density[transparent_mask.reshape([-1, 1])] = 0
        density = density.reshape(B, N, 1)
        return density

    ## world 2 smpl local
    def w2l(self, pts_world, ray_o_W, ray_d_W, batch):
        """transform point in the world coordinate to smplX local (template) space

        Args:
            pts_world (tensor): [bz,n_rays,num_sample,3]
            ray_o_W (tensor): [bz,n_rays,3]
            ray_d_W (tensor): [bz,n_rays,3]
            batch (dict): batch dict

        Returns:
            pts_smpl_can: [bz,n_rays*num_sample,3],points in canonical space
            rays: [bz,n_rays*num_sample,6], rays information : rays position in smple pose space and rays direction in canonical space
            transparent_mask: []
        """
        B, ray, sp, _ = pts_world.shape
        ray_d_W = ray_d_W.unsqueeze(2).expand([-1, -1, sp, -1]).reshape(B, -1, 3)
        pts_smpl_can, transparent_mask, ray_d_can = self.w2l_without_lbs(
            pts_world, batch, self.canonical_model, ray_d_W=ray_d_W
        )
        pts_smpl_can = pts_smpl_can.reshape(-1, sp, 3)

       
        ray_d_W = ray_d_W.reshape(-1, 3)
        rays = torch.cat([ray_d_W, ray_d_can], dim=-1)

        rays = rays.reshape(B * ray, sp, 6)

        pts_world = pts_world.reshape(B * ray, sp, -1)
        pts_world_smplt_can = torch.cat(
            [pts_world, pts_smpl_can], dim=-1
        )  # B*ray, sp, 6

        return pts_world_smplt_can, rays, transparent_mask

    def w2l_without_lbs(
        self, pts_world, batch, canonical_model, ray_d_W=None, floor=-4, ceil=5
    ):
        """warp points in world coordinate to canonical space by projection (mesh barycentric coordinate)

        Args:
            pts_world ([type]): [bz, n_rays, samples, 3]
            batch ([type]): [description]
            canonical_model ([type]): [description]
            ray_d_W ([type], optional): [bz,n_rays,3]. Defaults to None.
            floor (int, optional): [description]. Defaults to -4.
            ceil (int, optional): [description]. Defaults to 5.

        Returns:
            pts_smpl_can: [bz*n_rays*samples, 3]
            ray_d_can: [bz*n_rays*samples, 3]
        """
        # get world meshes
        B, ray, sp, _ = pts_world.shape
        xyz = batch["xyz"]
        xyz = xyz.to(pts_world)
        pts_world = pts_world.reshape(B, -1, 3)
        meshes = xyz[:, self.face_idx]
        # calculate closest mesh from pts_world
        closest_meshes, idx = get_closest_mesh(pts_world, meshes)
        # project pts_world to closest mesh
        uv, signed_distance = project_point2mesh(
            pts_world.reshape(-1, 3), meshes=closest_meshes.reshape(-1, 3, 3)
        )
        # for the clamped points, their density should be zero
        transparent_mask = get_transparent_mask(uv, signed_distance).reshape(B, -1)
        # get the mapped canonical pts
        meshes_can = canonical_model["meshes"][idx.flatten()]
        pts_smpl_can = barycentric_map2can(uv, signed_distance, meshes_can)
        if ray_d_W != None:
            pts_ray_d_W = pts_world + ray_d_W
            uv, signed_distance = project_point2mesh(
                pts_ray_d_W.reshape(-1, 3), meshes=closest_meshes.reshape(-1, 3, 3)
            )
            pts_ray_d_can = barycentric_map2can(uv, signed_distance, meshes_can)
            ray_d_can = torch.nn.functional.normalize(
                pts_ray_d_can - pts_smpl_can, dim=-1
            )
            return pts_smpl_can, transparent_mask, ray_d_can

        else:
            return pts_smpl_can, transparent_mask


    def load_body_model(self, gender, body_model, model_path):
        if body_model == "smpl":
            tmp = load_bodydata(body_model, gender, model_path)
            kintree_table = tmp["kintree_table"]
            parents = torch.Tensor(kintree_table[0].astype(np.int64)).long()
            parents[0] = -1

            weight = tmp["weights"]
            weight = torch.Tensor(weight)[None, ...]
            self.smpl_blend_weight = weight.cuda()

            face_idx = tmp["f"]
            face_idx = torch.Tensor(face_idx.astype(np.int64)).long()
            self.face_idx = face_idx.cuda()
            self.parents = parents
            x_pose = np.zeros((1, 24, 3))
            x_pose[:, 1, 2] += 0.6
            x_pose[:, 2, 2] -= 0.6
            self.x_pose = torch.Tensor(x_pose).cuda()
            # load canonical model
            if self.canonical_vertex is not None:
                self.canonical_model = {
                    "vertex": self.canonical_vertex.cuda(),
                    "meshes": self.canonical_vertex[self.face_idx].cuda(),
                }