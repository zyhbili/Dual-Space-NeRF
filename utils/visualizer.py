import numpy as np
import cv2 as cv
import torch
from skimage import measure
import trimesh
import pyrender

# from utils.mise import MISE


class Visualizer3D(object):
    """Visualizer3D of 3D implicit representations."""

    def __init__(
        self,
        resolution_mc,
        resolution_render,
        mc_value,
        gradient_direction,
        uniform_grid=False,
        connected=False,
        verbose=False,
    ):
        super().__init__()

        self.resolution_mc = resolution_mc
        self.resolution_render = resolution_render
        self.mc_value = mc_value
        self.gradient_direction = gradient_direction
        self.uniform_grid = uniform_grid
        self.connected = connected
        self.verbose = verbose

    @torch.no_grad()
    def get_grid_pred_batch(self, render, points=None, batch=None, chunk=100000):

        # Generating grid
        if self.uniform_grid:
            grid = self.get_grid_uniform()
            B = 1
        else:
            grid = self.get_grid(points.reshape(-1, points.shape[-1]))
            B = points.shape[0]
        grid_pts = grid["grid_pts"].unsqueeze(0).repeat(B, 1, 1)  # (Batch, Num_pts, 3)
        pts = grid_pts.clone()
        if batch is not None:
            pts_smpl_can, transparent_mask = render.w2l_without_lbs(
                pts.unsqueeze(-2).cuda(), batch, render.canonical_model
            )
            pts_smpl_can = pts_smpl_can.unsqueeze(0)  # (Batch, Num_pts, 3)
        # code_idx = torch.randperm(render.net.latent_codes.num_embeddings)[:B]
        code_idx = torch.randperm(300)[:B]
        print("Code_idx:", code_idx)

        # Evaluating points
        grid_pred = []

        # import pdb;pdb.set_trace()
        pts_world_smplt_can = torch.cat([pts.cuda(), pts_smpl_can], dim = -1) # B, ray, sp, 6
        for i in range(0, pts.shape[1], chunk):
            val = (
                render.query_volume(
                    pts_world_smplt_can[:, i : i + chunk].cuda(),
                    code_idx.cuda(),
                    transparent_mask[:, i : i + chunk],
                    batch
                )
                .detach()
                .cpu()
                .numpy()
            )
            grid_pred.append(val)
        # for i, pnts in enumerate(torch.split(pts, 100000, dim=1)):
        #     # if self.verbose:
        #     #     print ('%.1f' % (i/(can_pts.shape[0] // 100000) * 100))
        #     val = (
        #         render.query_volume(
        #             pnts.cuda(),
        #             code_idx.cuda().reshape(B, 1).repeat(1, pnts.shape[1]),
        #             transparent_mask,
        #         )
        #         .detach()
        #         .cpu()
        #         .numpy()
        #     )
        #     grid_pred.append(val)
        grid_pred = np.concatenate(grid_pred, axis=1)

        # Collecting results
        grid_pred = grid_pred.reshape(
            B,
            grid["xyz"][0].shape[0],
            grid["xyz"][1].shape[0],
            grid["xyz"][2].shape[0],
            1,
        )  # (B, X, Y, Z, 1)
        grid_pts = (
            grid_pts.reshape(
                B,
                grid["xyz"][0].shape[0],
                grid["xyz"][1].shape[0],
                grid["xyz"][2].shape[0],
                3,
            )
            .detach()
            .cpu()
            .numpy()
        )  # (B, X, Y, Z, 3)

        return grid_pts, grid_pred

    def get_mesh_from_grid(self, grid_pts, grid_pred):
        if not (np.min(grid_pred) > self.mc_value or np.max(grid_pred) < self.mc_value):
            # Marching cubes
            verts, faces, vertex_normals, values = measure.marching_cubes(
                volume=grid_pred.squeeze(-1),
                level=self.mc_value,
                spacing=(
                    grid_pts[1, 0, 0, 0] - grid_pts[0, 0, 0, 0],
                    grid_pts[0, 1, 0, 1] - grid_pts[0, 0, 0, 1],
                    grid_pts[0, 0, 1, 2] - grid_pts[0, 0, 0, 2],
                ),
                gradient_direction=self.gradient_direction,
            )
            verts = verts + grid_pts[None, 0, 0, 0]

            # Constructing Trimesh
            # mesh = trimesh.Trimesh(verts, faces, vertex_normals=vertex_normals, vertex_colors=values)
            mesh = trimesh.Trimesh(verts, faces)
            if self.connected:
                connected_comp = mesh.split(only_watertight=False)
                max_area = 0
                max_comp = None
                for comp in connected_comp:
                    if comp.area > max_area:
                        max_area = comp.area
                        max_comp = comp
                mesh = max_comp

            return mesh
        else:
            return None

    def render_mesh(self, mesh):
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

        camera_pose = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        scene.add(camera, pose=camera_pose)
        light = pyrender.SpotLight(
            color=np.ones(3),
            intensity=30.0,
            innerConeAngle=np.pi / 16.0,
            outerConeAngle=np.pi / 6.0,
        )
        scene.add(light, pose=camera_pose)
        r = pyrender.OffscreenRenderer(self.resolution_render, self.resolution_render)
        color, depth = r.render(scene)
        return color

    def get_grid(self, points):
        eps = 0.0
        input_min = torch.min(points, dim=0)[0].squeeze().cpu().numpy()
        input_max = torch.max(points, dim=0)[0].squeeze().cpu().numpy()
        bounding_box = input_max - input_min
        shortest_axis = np.argmin(bounding_box)
        if shortest_axis == 0:
            x = np.linspace(
                input_min[shortest_axis] - eps,
                input_max[shortest_axis] + eps,
                self.resolution_mc,
            )
            length = np.max(x) - np.min(x)
            y = np.arange(
                input_min[1] - eps,
                input_max[1] + length / (x.shape[0] - 1) + eps,
                length / (x.shape[0] - 1),
            )
            z = np.arange(
                input_min[2] - eps,
                input_max[2] + length / (x.shape[0] - 1) + eps,
                length / (x.shape[0] - 1),
            )
        elif shortest_axis == 1:
            y = np.linspace(
                input_min[shortest_axis] - eps,
                input_max[shortest_axis] + eps,
                self.resolution_mc,
            )
            length = np.max(y) - np.min(y)
            x = np.arange(
                input_min[0] - eps,
                input_max[0] + length / (y.shape[0] - 1) + eps,
                length / (y.shape[0] - 1),
            )
            z = np.arange(
                input_min[2] - eps,
                input_max[2] + length / (y.shape[0] - 1) + eps,
                length / (y.shape[0] - 1),
            )
        elif shortest_axis == 2:
            z = np.linspace(
                input_min[shortest_axis] - eps,
                input_max[shortest_axis] + eps,
                self.resolution_mc,
            )
            length = np.max(z) - np.min(z)
            x = np.arange(
                input_min[0] - eps,
                input_max[0] + length / (z.shape[0] - 1) + eps,
                length / (z.shape[0] - 1),
            )
            y = np.arange(
                input_min[1] - eps,
                input_max[1] + length / (z.shape[0] - 1) + eps,
                length / (z.shape[0] - 1),
            )

        xx, yy, zz = torch.meshgrid(torch.tensor(x), torch.tensor(y), torch.tensor(z))
        grid_points = torch.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T.float()

        return {
            "grid_pts": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis,
        }

    def get_grid_uniform(self):
        x = np.linspace(-1.2, 1.2, self.resolution_mc)
        y = x
        z = x

        xx, yy, zz = torch.meshgrid(torch.tensor(x), torch.tensor(y), torch.tensor(z))
        grid_points = torch.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T.float()

        return {
            "grid_pts": grid_points,
            "shortest_axis_length": 2.4,
            "xyz": [x, y, z],
            "shortest_axis_index": 0,
        }


if __name__ == "__main__":

    vis = Visualizer3D(
        resolution_mc=512,
        resolution_render=1024,
        mc_value=0.5,
        gradient_direction="ascent",
    )

    x = torch.rand([100, 3]).cuda()
    x = model(x)

    vis.plot_surface(model)
