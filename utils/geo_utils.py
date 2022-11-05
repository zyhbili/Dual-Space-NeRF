import torch

### pts (B, pts, meshes, 3)
### meshes(B, meshes, 3, 3)
### uv (B, pts, meshes, 2)
# u corresponding to edge 0,2
# v corresponding to edge 0,1
def computeBarycentricCoordinates(pts_proj, meshes):
    v0 = meshes[..., 2, :] - meshes[..., 0, :]
    v1 = meshes[..., 1, :] - meshes[..., 0, :]
    v2 = pts_proj - meshes[..., 0, :][:, None]

    dot00 = (v0 * v0).sum(-1)
    dot01 = (v0 * v1).sum(-1)
    dot02 = (v0[:, None] * v2).sum(-1)
    dot11 = (v1 * v1).sum(-1)
    dot12 = (v1[:, None] * v2).sum(-1)

    inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11[:, None] * dot02 - dot01[:, None] * dot12) * inverDeno[:, None]
    v = (dot00[:, None] * dot12 - dot01[:, None] * dot02) * inverDeno[:, None]

    uv = torch.stack([u, v], dim=-1)
    return uv


def check_inside(uv):
    mask1 = torch.logical_and(uv >= 0, uv <= 1).all(dim=-1)
    mask2 = torch.sum(uv, dim=-1) <= 1
    return torch.logical_and(mask1, mask2)


### pts (B, pts, 3)
### edges (B,edges,3)
def project_pts2edge(pts, node0, node1):
    dir = node1 - node0
    dir_unit = dir / torch.norm(dir, dim=-1, keepdim=True)
    tmp = pts[:, :, None] - node0[:, None]

    proj_len = (dir_unit * tmp).sum(-1)
    proj_pts = node0[:, None] + proj_len[..., None] * dir_unit[:, None]

    return proj_pts


### pts (B, pts, 3)
### meshes(B,meshes,3,3)
def project_pts2mesh(pts, meshes):
    """project any pts to all meshes

    Args:
        pts ([type]): [description]
        meshes ([type]): [description]

    Returns:
        [type]: [description]
    """
    B, n_pts, _ = pts.shape
    _, nF, _, _ = meshes.shape

    Vf = meshes.reshape((B * nF, 3, 3))
    v10 = Vf[:, 1] - Vf[:, 0]
    v20 = Vf[:, 2] - Vf[:, 0]
    # import pdb;pdb.set_trace()

    normal_f = torch.cross(v10, v20).reshape(B, nF, 3)
    normal_f = normal_f / torch.norm(normal_f, dim=-1, keepdim=True)

    # tmp (B, pts, meshes, 3)
    tmp = pts[:, :, None] - meshes[:, :, 0][:, None]
    # signed_distance (B, pts, meshes)
    signed_distance = torch.einsum("bpfs,bfs->bpf", tmp, normal_f)

    # (B, pts, meshes, 3)
    pts_proj = pts[..., None, :] - torch.einsum(
        "bfs,bpf->bpfs", normal_f, signed_distance
    )

    uv = computeBarycentricCoordinates(pts_proj, meshes)
    inside_mask = check_inside(uv)  # shape(B, pts, meshes)
    outside_mask = torch.logical_not(inside_mask)

    # done_mask = inside_mask.any(dim = -1)
    signed_distance[outside_mask] = 999
    # get the closest mesh
    min_dist, idx = torch.abs(signed_distance).min(-1)

    uv = uv.reshape(B * n_pts, -1, 2)[torch.arange(B * n_pts), idx.flatten()].reshape(
        [B, n_pts, 2]
    )

    return uv, idx, min_dist, inside_mask.any(dim=-1)


def get_barycentric_coordinates(pts_proj, meshes):
    v0 = meshes[..., 2, :] - meshes[..., 0, :]
    v1 = meshes[..., 1, :] - meshes[..., 0, :]
    v2 = pts_proj - meshes[..., 0, :]

    dot00 = (v0 * v0).sum(-1)
    dot01 = (v0 * v1).sum(-1)
    dot02 = (v0 * v2).sum(-1)
    dot11 = (v1 * v1).sum(-1)
    dot12 = (v1 * v2).sum(-1)

    inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno

    uv = torch.stack([u, v], dim=-1)
    return uv


def get_barycentric_coordinates_in_center(pts_proj, meshes):

    mesh_centroid = meshes.mean(dim=-2)
    v0 = torch.nn.functional.normalize(meshes[..., 0, :] - mesh_centroid, dim=-1)
    v1 = torch.nn.functional.normalize(meshes[..., 1, :] - mesh_centroid, dim=-1)
    v2 = pts_proj - mesh_centroid

    dot00 = (v0 * v0).sum(-1)
    dot01 = (v0 * v1).sum(-1)
    dot02 = (v0 * v2).sum(-1)
    dot11 = (v1 * v1).sum(-1)
    dot12 = (v1 * v2).sum(-1)

    inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno

    uv = torch.stack([u, v], dim=-1)
    return uv


def barycentric_map2can(uv, signed_distance, meshes_can):
    """map points to canonical space by "uv and distance" coordinate

    Args:
        uv ([type]): [description]
        signed_distance ([type]): [description]
        meshes_can ([type]): [description]

    Returns:
        [type]: [description]
    """
    v2 = meshes_can[..., 2, :] - meshes_can[..., 0, :]
    v1 = meshes_can[..., 1, :] - meshes_can[..., 0, :]
    normal_f = torch.cross(v1, v2)
    normal_f = normal_f / torch.norm(normal_f, dim=-1, keepdim=True)
    offset_vec = signed_distance.unsqueeze(-1) * normal_f
    pts_proj_can = meshes_can[..., 0, :] + uv[:, 0, None] * v2 + uv[:, 1, None] * v1
    pts_smpl_can = pts_proj_can + offset_vec
    return pts_smpl_can


def barycentric_map2can_in_center(uv, signed_distance, meshes_can):
    """map points to canonical space by "uv and distance" coordinate

    Args:
        uv ([type]): [description]
        signed_distance ([type]): [description]
        meshes_can ([type]): [description]

    Returns:
        [type]: [description]
    """
    mesh_centroid = meshes_can.mean(dim=-2)
    v0 = torch.nn.functional.normalize(meshes_can[..., 0, :] - mesh_centroid, dim=-1)
    v1 = torch.nn.functional.normalize(meshes_can[..., 1, :] - mesh_centroid, dim=-1)
    normal_f = torch.cross(v1, v0)
    normal_f = normal_f / torch.norm(normal_f, dim=-1, keepdim=True)
    offset_vec = signed_distance.unsqueeze(-1) * normal_f
    pts_proj_can = mesh_centroid + uv[:, 0, None] * v0 + uv[:, 1, None] * v1
    pts_smpl_can = pts_proj_can + offset_vec
    return pts_smpl_can


def project_point2mesh(pts, meshes):
    """project points to corresponding meshes,points number must be the same as mesh number

    Args:
        pts (tensor): [n,3]
        meshes (tensor): [n,3,3]
    """
    assert (
        pts.shape[0] == meshes.shape[0]
    ), "points number must be the same as mesh number"
    v10 = meshes[:, 1] - meshes[:, 0]
    v20 = meshes[:, 2] - meshes[:, 0]
    normal_f = torch.cross(v10, v20)
    normal_f = normal_f / torch.norm(normal_f, dim=-1, keepdim=True)
    tmp = pts - meshes[:, 0]
    signed_distance = torch.einsum("ij,ij->i", tmp, normal_f)
    pts_proj = pts - normal_f * signed_distance.unsqueeze(-1)
    uv = get_barycentric_coordinates(pts_proj=pts_proj, meshes=meshes)

    return uv, signed_distance


def project_point2mesh_in_center(pts, meshes):
    """project points to corresponding meshes,points number must be the same as mesh number

    Args:
        pts (tensor): [n,3]
        meshes (tensor): [n,3,3]
    """
    assert (
        pts.shape[0] == meshes.shape[0]
    ), "points number must be the same as mesh number"
    mesh_centroid = meshes.mean(dim=-2)
    v2 = torch.nn.functional.normalize(meshes[..., 0, :] - mesh_centroid, dim=-1)
    v1 = torch.nn.functional.normalize(meshes[..., 1, :] - mesh_centroid, dim=-1)
    normal_f = torch.cross(v1, v2)
    normal_f = normal_f / torch.norm(normal_f, dim=-1, keepdim=True)
    tmp = pts - mesh_centroid
    signed_distance = torch.einsum("ij,ij->i", tmp, normal_f)
    pts_proj = pts - normal_f * signed_distance.unsqueeze(-1)
    uv = get_barycentric_coordinates_in_center(pts_proj=pts_proj, meshes=meshes)

    return uv, signed_distance


def compute_uvfh(pts, meshes):
    B, n_pts = pts.shape[:2]

    uv = torch.zeros(B, n_pts, 2).to(pts)
    f = torch.zeros(B, n_pts).to(pts).long()
    h = torch.zeros(B, n_pts).to(pts)

    uv_mesh, f_mesh, dist_mesh, mesh_mask = project_pts2mesh(pts, meshes)
    uv[mesh_mask] = uv_mesh[mesh_mask]
    f[mesh_mask] = f_mesh[mesh_mask]
    h[mesh_mask] = dist_mesh[mesh_mask]

    ret = {
        "uvfh": torch.cat([uv, f[..., None], h[..., None]], dim=-1),
        "mask": mesh_mask,
    }
    return ret


def compute_edge_uvfh(pts, meshes):
    x_a = project_pts2edge(pts, meshes[..., 2, :], meshes[..., 0, :])
    x_b = project_pts2edge(pts, meshes[..., 1, :], meshes[..., 0, :])
    x_c = project_pts2edge(pts, meshes[..., 2, :], meshes[..., 1, :])

    x = torch.stack([x_a, x_b, x_c], dim=-2)

    dist_edge = torch.norm(x - pts[:, :, None, None], dim=-1)

    min_dist_edge, edge_idx = dist_edge.min(-1)

    B, n_pts, n_meshes = x.shape[:3]
    pts_proj_nearest = x.reshape([B, n_pts * n_meshes, 3, 3])[
        :, torch.arange(n_pts * n_meshes), edge_idx.flatten()
    ].reshape([B, n_pts, n_meshes, 3])
    min_dist_mesh, mesh_idx = min_dist_edge.min(-1)  ###3条边里最小的那条

    uv = computeBarycentricCoordinates(pts_proj_nearest, meshes)
    uv = uv[:, torch.arange(n_pts), mesh_idx.flatten()].reshape(
        [B, n_pts, 2]
    )  ###选择最近面的uv
    return uv, mesh_idx, min_dist_mesh


if __name__ == "__main__":
    A = torch.Tensor([0, 0, 0])[None, None, None]
    B = torch.Tensor([1, 0, 0])[None, None, None]
    C = torch.Tensor([1, 0, 1])[None, None, None]
    D = torch.Tensor([1, 1, 1])[None, None, None]

    AB = B - A
    AC = C - A

    mesh1 = torch.cat([A, B, C], dim=2)
    mesh2 = torch.cat([A, B, D], dim=2)
    meshes = torch.cat([mesh1, mesh2], dim=1)
    # print(AB.shape)
    # print(AC.shape)
    print("mesh", meshes.shape)
    pts1 = torch.Tensor([1, 1, 1]) / 2
    pts2 = torch.Tensor([10, 1, 1]) / 2
    pts3 = torch.Tensor([5, 5, 10])
    pts = torch.stack([pts1, pts2, pts3], dim=0)[None]
    print("pts", pts.shape)
    # print(pts)

    compute_uvfh(pts, meshes)

