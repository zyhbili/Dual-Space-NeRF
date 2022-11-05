import torch
import torch.nn.functional as F
from torch import nn

from .dimension_kernel import Trigonometric_kernel, Gaussian_Kernel

from utils.render_utils import (
    compute_nn_mesh,
    post_process,
    compute_mlp_delta_weights,
    load_obj_data,
    get_closest_mesh,
)
from utils.geo_utils import project_point2mesh, barycentric_map2can
from torch.autograd import grad


class SpaceNet(nn.Module):
    def __init__(self, maxFrame=500, code_dim=8, essence_dim=3, cfg=None):
        super(SpaceNet, self).__init__()
        self.use_dir = False
        self.training = True
        self.cfg = cfg

        self.code_dim = code_dim

        self.tri_kernel_pos = Trigonometric_kernel(L=10, include_input=True)
        if self.use_dir:
            self.tri_kernel_dir = Trigonometric_kernel(L=4, include_input=True)

        self.pos_dim = self.tri_kernel_pos.calc_dim(3)
        if self.use_dir:
            self.dir_dim = self.tri_kernel_dir.calc_dim(3)
        else:
            self.dir_dim = 0

        backbone_dim = 256
        head_dim = int(backbone_dim / 2)

        if self.code_dim > 0:
            self.embedding = nn.Embedding(maxFrame, self.code_dim)
            # nn.init.constant_(self.embedding.weight, 1e-6)
            in_dim = self.pos_dim + self.code_dim + 16
        else:
            in_dim = self.pos_dim
        # 4-layer MLP for density feature
        self.stage1 = nn.Sequential(
            nn.Linear(in_dim, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(inplace=True),
        )
        # 4-layer MLP for density feature with a skipped input and stage1 output
        self.stage2 = nn.Sequential(
            nn.Linear(
                backbone_dim + self.pos_dim, backbone_dim
            ),  # ?: Seems different from the structure in the paper
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(inplace=True),
        )
        # 1-layer MLP for density
        self.density_net = nn.Sequential(
            nn.Linear(backbone_dim, 1),
            # density value should be more than zero
            # nn.ReLU(inplace=True)
        )
        # 2-layer MLP for rgb
        self.rgb_net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim + self.dir_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, essence_dim),
        )
        self.w = None

    """
    INPUT
    pos: 3D positions (N,L,c_pos) or (N,c_pos)
    rays: corresponding rays  (N,6)
    OUTPUT
    rgb: color (N,L,3) or (N,3)
    density: (N,L,1) or (N,1)
    N is the number of rays
    """

    def forward(
        self, pos, rays, idx, density_only=False, pose_feats=None
    ):

        rgbs = None
        if rays is not None and self.use_dir:

            dirs = rays[..., 3:6]
            # Normalization for a better performance when training, may not necessary
            dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        else:
            dirs = None

        # When input is [N, L, 3], it will be set to True
        bins_mode = False
        # import pdb;pdb.set_trace()
        if len(pos.size()) > 2:
            bins_mode = True
            L = pos.size(1)
            pos = pos.reshape((-1, 3))  # (N, 3)
            if rays is not None and self.use_dir:
                #     dirs = dirs.unsqueeze(1).repeat(1, L, 1)
                dirs = dirs.reshape((-1, 3))  # (N, 3)

        # Positional encoding
        pos = self.tri_kernel_pos(pos)

        if self.use_dir and dirs is not None:
            dirs = self.tri_kernel_dir(dirs)
        # 8-layer MLP for density
        if self.code_dim > 0:
            # if pose_feats is None:
            idx = idx.flatten()
            if self.w is None:
                code = self.embedding(idx)
            else:
                code = 0 * self.embedding(idx)
            x = self.stage1(torch.cat([code, pos, pose_feats], dim=1))
        else:
            x = self.stage1(pos)

        x = self.stage2(torch.cat([x, pos], dim=1))
        density = self.density_net(x)

        if density_only:
            return density

        # MLP for rgb with or without direction of ray
        x1 = 0
        if rays is not None and self.use_dir:
            x1 = torch.cat([x, dirs], dim=1)
        else:
            x1 = x.clone()

        rgbs = self.rgb_net(x1)
        return rgbs, density, 0



class LightingMLP(nn.Module):
    def __init__(self, essence_dim):
        super(LightingMLP, self).__init__()
        self.tri_kernel_normal = Trigonometric_kernel(L=0, include_input=True)
        self.tri_kernel_xyz = Trigonometric_kernel(L=0, include_input=True)
        self.tri_kernel_dir = Trigonometric_kernel(L=0, include_input=True)

        self.in_channels_normal = self.tri_kernel_normal.calc_dim(3)
        self.in_channels_xyz = self.tri_kernel_xyz.calc_dim(3)
        self.in_channels_dir = self.tri_kernel_dir.calc_dim(3)
        self.in_channels = self.in_channels_normal + self.in_channels_xyz +3
        W = 128

        self.lights_encoding = nn.Sequential(
            nn.Linear(self.in_channels, W),
            nn.ReLU(True),
            nn.Linear(W, W),
            nn.ReLU(True),
            nn.Linear(W, 1),
            nn.ELU(alpha=1.0, inplace=True),
        )

    def forward(self, normal, xyz_world, view_dir_world, essence_feature):
        normal = self.tri_kernel_normal(normal)
        xyz_world = self.tri_kernel_xyz(xyz_world)

        view_dir_world = view_dir_world / torch.norm(
            view_dir_world, dim=-1, keepdim=True
        )
        view_dir_world = self.tri_kernel_dir(view_dir_world)

        inputs = torch.cat([normal, xyz_world,view_dir_world], dim=1)

        w = self.lights_encoding(inputs) + 1
        color = w * essence_feature

        return color


class DualSpaceNeRF(nn.Module):
    def __init__(self, cfg):
        super(DualSpaceNeRF, self).__init__()

        essence_dim = 3
        self.nerf = SpaceNet(essence_dim=essence_dim, cfg=cfg)
        self.lighting_mlp = LightingMLP(essence_dim=essence_dim)

        self.pose_mlp = nn.Sequential(
            nn.Linear(23 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
        )
        self.light_center = None
        self.rot_center = None
        self.rot = None

    def forward(
        self,
        pos,
        rays,
        frame_idx=0,
        batch_info={},
        density_only=False,
    ):

        xyz_cano = pos[..., 3:]
        xyz_cano.requires_grad = True

        ### only work on batch size 1
        body_pose = batch_info["poses"][0][1:, :].float()
        if len(xyz_cano.shape) == 3:
            pose = (
                batch_rod2quat(body_pose.reshape(-1, 3))
                .reshape(1, -1)
                .repeat(xyz_cano.shape[1], 1)
            )
        else:
            pose = (
                batch_rod2quat(body_pose.reshape(-1, 3))
                .reshape(1, -1)
                .repeat(xyz_cano.shape[0], 1)
            )
        pose_feat = self.pose_mlp(pose.cuda())

        if density_only:
            return self.nerf(
                xyz_cano, rays, frame_idx, density_only,  pose_feat
            )

        essence, density, _ = self.nerf(
            xyz_cano, rays, frame_idx, density_only, pose_feat
        )


        xyz_world = pos[:, :3]
        view_dir_world = rays[:, :3]

        normal_local = gradient(xyz_cano, density)
        normal_world = normal_local2world(normal_local, xyz_cano, batch_info)

        if self.rot_center is not None and self.rot is not None:
            tmp = (xyz_world[:, :2] - self.rot_center[:, :2]).matmul(
                self.rot
            ) + self.rot_center[:, :2]
            xyz_world[:, :2] = tmp

        if self.light_center is not None:
            xyz_tmp = batch_info["Th"][0].cuda()
            xyz_bias = self.light_center - torch.mean(xyz_tmp, axis=0)
            xyz_world[:, :2] += xyz_bias[:2]

        color = self.lighting_mlp(normal_world, xyz_world, view_dir_world, essence)
        return color, density, None

    def set_rot_center(self, center):
        self.rot_center = center.cuda()

    def set_rot(self, rot):
        self.rot = rot.cuda()

    def set_light_center(self, center):
        self.light_center = center.cuda()


def normal_local2world(normal_local, xyz_cano, batch_info):
    meshes_cano = batch_info["canonical_model"]["meshes"][None]
    closest_meshes, idx = get_closest_mesh(xyz_cano[None], meshes_cano)
    uv, signed_distance = project_point2mesh(
        xyz_cano, meshes=closest_meshes.reshape(-1, 3, 3)
    )

    meshes_world = batch_info["xyz"][:, batch_info["face_idx"]][:, idx.flatten()].cuda()

    pts_world_start = barycentric_map2can(uv, signed_distance, meshes_world)

    uv, signed_distance = project_point2mesh(
        (xyz_cano + normal_local), meshes=closest_meshes.reshape(-1, 3, 3)
    )
    pts_world_end = barycentric_map2can(uv, signed_distance, meshes_world)

    normal_world = torch.nn.functional.normalize(
        pts_world_end - pts_world_start, dim=-1
    )

    return normal_world.squeeze()


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    return points_grad[0]


def batch_rod2quat(rot_vecs):
    batch_size = rot_vecs.shape[0]

    angle = torch.norm(rot_vecs + 1e-16, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.cos(angle / 2)
    sin = torch.sin(angle / 2)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)

    qx = rx * sin
    qy = ry * sin
    qz = rz * sin
    qw = cos - 1.0

    return torch.cat([qx, qy, qz, qw], dim=1)


if __name__ == "__main__":
    model = SpaceNet()