from pytorch3d.ops import knn_gather, knn_points
import torch
from torch.functional import meshgrid
from .geo_utils import compute_uvfh, project_point2mesh, barycentric_map2can
import time
import numpy as np


def load_obj_data(obj_file):
    with open(obj_file, "r") as fp:
        verts = []
        faces = []
        vts = []
        vns = []
        faces_vts = []
        faces_vns = []

        for line in fp:
            line = line.rstrip()
            line_splits = line.split()
            prefix = line_splits[0]

            if prefix == "v":
                verts.append(
                    np.array(
                        [line_splits[1], line_splits[2], line_splits[3]],
                        dtype=np.float32,
                    )
                )

            elif prefix == "vn":
                vns.append(
                    np.array(
                        [line_splits[1], line_splits[2], line_splits[3]],
                        dtype=np.float32,
                    )
                )

            elif prefix == "vt":
                vts.append(np.array([line_splits[1], line_splits[2]], dtype=np.float32))

            elif prefix == "f":
                f = []
                f_vt = []
                f_vn = []
                for p_str in line_splits[1:4]:
                    p_split = p_str.split("/")
                    f.append(p_split[0])

                    if len(p_split) > 1:
                        f_vt.append(p_split[1])
                        f_vn.append(p_split[2])

                # index from 0
                faces.append(np.array(f, dtype=np.int32) - 1)
                faces_vts.append(np.array(f_vt, dtype=np.int32) - 1)
                faces_vns.append(np.array(f_vn, dtype=np.int32) - 1)

            elif prefix == "g" or prefix == "s":
                continue

            else:
                # raise ValueError(prefix)
                pass

        obj_dict = {
            "vertices": np.array(verts, dtype=np.float32),
            "faces": np.array(faces, dtype=np.int32),
            "vts": np.array(vts, dtype=np.float32),
            "vns": np.array(vns, dtype=np.float32),
            "faces_vts": np.array(faces_vts, dtype=np.int32),
            "faces_vns": np.array(faces_vns, dtype=np.int32),
        }
        # calculate the mesh_centroid_uv_map
        uv = obj_dict["vts"]
        faces_vts = obj_dict["faces_vts"]
        mesh_centroid_uv_map = (
            uv[faces_vts[:, 0]] + uv[faces_vts[:, 1]] + uv[faces_vts[:, 2]]
        ) / 3
        obj_dict["mesh_centroid_uv_map"] = mesh_centroid_uv_map
        return obj_dict


def get_closest_mesh(vsrc, meshes):
    """get closest mesh by barycentric points of each mesh

    Args:
        vsrc ([type]): [description]
        meshes ([type]): [description]

    Returns:
        [type]: [description]
    """
    mesh_centroid = meshes.mean(dim=-2)
    dist, idx, Vnn = knn_points(vsrc, mesh_centroid, K=1, return_nn=True)
    closest_meshes = torch.gather(
        meshes, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, 3, 3)
    )
    return closest_meshes, idx


# def get_transparent_mask(uv, signed_distance, floor=-2, ceil=3, max_dist=0.05):
def get_transparent_mask(uv, signed_distance, floor=-4, ceil=5, max_dist=0.1):
    clamped_uv_mask = torch.logical_or(uv > ceil, uv < floor)
    transparent_mask = torch.logical_or(clamped_uv_mask[:, 0], clamped_uv_mask[:, 1])
    transparent_mask = torch.logical_or(
        transparent_mask, signed_distance.abs() > max_dist
    )
    return transparent_mask


def get_base_blending_weights(
    vsrc,
    meshes,
    closest_face_idx,
    smpl_blending_weights,
    face_idx=None,
    bw_type="",
):
    """calculate the blending weights of vsrc(any point in R^3) from the closest mesh

    Args:
        vsrc (tensor): [B,n_pts,3]
        meshes (tensor): [B,n_pts,3,3]
        closest_face_idx (tensor): [B,n_pts,1]
        smpl_blending_weights (tensor): [1,num_vertex,24]
        face_idx (tensor, optional): [1,num_vertex,3]. Defaults to None.
        bw_type (str, optional): [description]. Defaults to "".

    Raises:
        ValueError: [description]

    Returns:
        base_blending_weights: [B,n_pts,24]
    """
    # bz, n_pts, _ = closest_face_idx.shape
    # _, num_vertexs, num_joints = smpl_blending_weights.shape

    vertexs_idxs_in_mesh = face_idx[
        closest_face_idx.squeeze(-1)
    ].long()  # shape: [B,n_pts,3]
    if bw_type == "rigid_interp":

        # vertex_in_mesh: vertex_coordinate of each mesh
        vertex_in_mesh = torch.gather(
            meshes,
            dim=1,
            index=closest_face_idx.unsqueeze(-1).expand([-1, -1, 3, 3]),
        )
        dist_vec = vertex_in_mesh - vsrc.unsqueeze(2).expand([-1, -1, 3, -1])
        dist = torch.norm(dist_vec, dim=-1)
        w_ = torch.nn.functional.softmax(dist, dim=-1)
        vertex_blending_weights = smpl_blending_weights.squeeze()[vertexs_idxs_in_mesh]
        base_blending_weights = (w_.unsqueeze(-1) * vertex_blending_weights).sum(-2)

    elif bw_type == "rigid_center":
        base_blending_weights = smpl_blending_weights.squeeze()[
            vertexs_idxs_in_mesh
        ].mean(dim=2)
    else:
        raise ValueError("unsupport value: bw_type")
    # normalize blending weights
    # base_blending_weights = torch.nn.functional.softmax(base_blending_weights, dim=-1)
    return base_blending_weights


def compute_knn_v_feat(vsrc, vtar, vfeat, K=3):
    dist, idx, Vnn = knn_points(vsrc, vtar, K=K, return_nn=True)
    B = idx.shape[0]
    if vfeat.shape[0] != B:
        vfeat = vfeat.expand(B, -1, -1)
    return knn_gather(vfeat, idx).mean(-2)


def get_uv_from_closest_plane(vsrc, meshes, uv_map, faces_vts):
    """calculate the uv coordinate of vsrc from closest mesh

    Args:
        vsrc (tensor): [B,n_pts,3]
        meshes (tensor): [B,num_faces,3,3]
        uv_map (tensor): [vertex_num,2]
        faces_vts (tensor): [num_faces,3]

    Returns:
        uv (tensor): [B,n_pts,2]
        h (tensor): [B, n_pts]
    """
    bz, n_pts, _ = vsrc.shape
    # get nearest face
    uvfh, mask_mesh = batchify_compute_uvfh(vsrc, meshes).values()
    # offset_uv (B, n_pts, 2)
    offset_uv = uvfh[..., :2]
    face_idxs = uvfh[
        ..., 2
    ]  # shape is [B,n_pts], the range of element in face_idxs is [0,len(meshs)-1]
    h = uvfh[..., 3]
    #
    vertex_uv_idx = faces_vts[face_idxs.flatten().long()].long()  # shape [B*n_pts,3]
    u_vector = (
        uv_map[vertex_uv_idx[..., 2]] - uv_map[vertex_uv_idx[..., 0]]
    )  # shape [B*n_pts,2]
    v_vector = uv_map[vertex_uv_idx[..., 1]] - uv_map[vertex_uv_idx[..., 0]]
    uv = (
        uv_map[vertex_uv_idx[..., 0]]
        + offset_uv[..., 0].reshape(-1, 1) * u_vector
        + offset_uv[..., 1].reshape(-1, 1) * v_vector
    ).reshape(bz, n_pts, 2)
    return uv, h, offset_uv, face_idxs, mask_mesh


def get_ebuvh_from_closest_mesh(vsrc, meshes, embeddings, floor=-4, ceil=5):
    bz, n_pts, _ = vsrc.shape
    closest_meshes, idx = get_closest_mesh(vsrc=vsrc, meshes=meshes)
    # project pts_world to closest mesh
    uv, signed_distance = project_point2mesh(
        vsrc.reshape(-1, 3), meshes=closest_meshes.reshape(-1, 3, 3)
    )
    # for the clamped points, their density should be zero
    transparent_mask = get_transparent_mask(uv, signed_distance).reshape(bz, n_pts)
    # clamp uv
    uv = torch.clamp(uv, min=floor, max=ceil)
    uvh = torch.cat((uv, signed_distance.unsqueeze(-1)), dim=-1).reshape(bz, n_pts, 3)
    embedding = embeddings(idx.squeeze(-1))
    return embedding, uvh, idx, transparent_mask


def get_uvuvh_from_closest_mesh(vsrc, meshes, mesh_centroid_uv_map, floor=-4, ceil=5):
    bz, n_pts, _ = vsrc.shape
    closest_meshes, idx = get_closest_mesh(vsrc=vsrc, meshes=meshes)
    # project pts_world to closest mesh
    uv, signed_distance = project_point2mesh(
        vsrc.reshape(-1, 3), meshes=closest_meshes.reshape(-1, 3, 3)
    )
    # for the clamped points, their density should be zero
    transparent_mask = get_transparent_mask(uv, signed_distance).reshape(bz, n_pts)
    # clamp uv
    uv = torch.clamp(uv, min=floor, max=ceil)
    uvh = torch.cat((uv, signed_distance.unsqueeze(-1)), dim=-1).reshape(bz, n_pts, 3)
    uv_texture_coord = mesh_centroid_uv_map[idx.flatten()].reshape(bz, n_pts, 2)
    return uv_texture_coord, uvh, idx, transparent_mask


def compute_mlp_delta_weights(
    vsrc,
    target,
    net,
    uv_data,
    blending_weights,
    smpl_face_idx=None,
    feature_type="uvxyzh",
    base_bw_type="rigid_interp",
):
    """calculate blending weights from the closest mesh or face

    Args:
        vsrc (tensor): [B,n_pts,3]
        target (tensor): [B,num_faces,3,3]
        net (nn.module): network to predict delta blending weights
        uv_data (dict): uv data for mapping
        blending_weights (tensor): [1,num_vertexs,24]
        smpl_face_idx (tensor): [1,num_faces,3]
        feature_type (str, optional): feature type. Defaults to "uvxyz".
        base_bw_type (str,option): base blending weight type. Default to "rigid_interp".

    Raises:
        ValueError: [description]

    Returns:
        [tensor]: [description]
    """
    # bz, n_pts, _ = vsrc.shape
    # _, num_vertexs, num_joints = blending_weights.shape
    transparent_mask = None
    if feature_type == "uvuvh":
        (
            uv_texture_coord,
            uvh,
            face_idxs,
            transparent_mask,
        ) = get_uvuvh_from_closest_mesh(
            vsrc=vsrc,
            meshes=target,
            mesh_centroid_uv_map=uv_data["mesh_centroid_uv_map"],
        )
        feat = torch.cat((uv_texture_coord, uvh), dim=-1).permute([0, 2, 1])
    elif feature_type == "eb_uvh":
        """[embedding,uvh]: uv in Barycentric space of mesh"""
        embedding, uvh, face_idxs, transparent_mask = get_ebuvh_from_closest_mesh(
            vsrc=vsrc, meshes=target, embeddings=net.face_embeddings
        )
        feat = torch.cat((embedding, uvh), dim=-1).permute([0, 2, 1])
    else:
        raise ValueError("unsupport value: feature_type")

    # calculate base blending weights
    # shape:[B,n_pts,24]
    base_blending_weights = get_base_blending_weights(
        vsrc=vsrc,
        meshes=target,
        closest_face_idx=face_idxs,
        smpl_blending_weights=blending_weights,
        face_idx=smpl_face_idx,
        bw_type=base_bw_type,
    )
    # forward get delta bw
    delta_blending_weights = net(feat).permute([0, 2, 1])

    weights = delta_blending_weights + torch.log(base_blending_weights + 1e-9)
    weights = torch.nn.functional.softmax(weights, dim=-1)
    torch.cuda.empty_cache()
    return weights, transparent_mask


def compute_knn_feat(vsrc, vtar, vfeat, K=1):
    """Each source points queries the k-nearest neighbors of the target points, and gathers their features with an
    attentional reduction.

    Args:
        vsrc (torch.Tensor): [n_batch, num_points_1, 3], source points;
        vtar (torch.Tensor): [n_batch, num_points_2, 3], target points;
        vfeat (torch.Tensor): [n_batch, num_points_2, dim], target features;
        K (int): the k-nearest neighours.

    Returns:
        fused_feat (torch.Tensor): [n_batch, num_points_1, dim], the fused k-nearest neighours's features.

    """

    # dist is [n_batch, num_points_1, K]
    # idx is [n_batch, num_points_1, K]
    dist, idx, _ = knn_points(vsrc, vtar, K=K, return_nn=False)
    # nearest points with higher scores, [n_batch, num_points_1, K]
    score = torch.softmax(
        -dist / (torch.sum(dist, dim=-1, keepdim=True) + 1e-6), dim=-1
    )
    # import pdb;pdb.set_trace()

    B = idx.shape[0]
    if vfeat.shape[0] != B:
        vfeat = vfeat.expand(B, -1, -1)
    # [n_batch, num_points_1, K, dim]
    knn_feat = knn_gather(vfeat, idx)

    # fusion, [n_batch, num_points_1, K, 1] * [n_batch, num_points_1, K, dim] = [n_batch, num_points_1, K, dim]
    # sum([n_batch, num_points_1, K, dim], dim=-2) = [n_batch, num_points_1, dim]
    fused_feat = torch.sum(score.unsqueeze(-1) * knn_feat, dim=-2)
    # import pdb;pdb.set_trace()

    return fused_feat


def compute_nn_mesh(
    pts,
    meshes,
    smpl_blend_weight,
    face_idx,
    base_bw_type="",
    ray_d=None,
    canonical_model=None,
):

    """directly calculate blending weights from the closest mesh without lbsnet
        if ray_d is not None, map the ray direction to canonical space

    Args:
        pts (tensor): [B,n_pts,3]
        meshes (tensor): [B,num_faces,3,3]
        smpl_blend_weight (tensor): [1,num_vertexs,24]
        face_idx ([type]): [description]
        base_bw_type (str, optional): [blending weights type]. Defaults to "".
        ray_d: [B,n_pts,3]
    """
    bz, n_pts, _ = pts.shape
    closest_meshes, idx = get_closest_mesh(vsrc=pts, meshes=meshes)
    # project pts_world to closest mesh
    uv, signed_distance = project_point2mesh(
        pts.reshape(-1, 3), meshes=closest_meshes.reshape(-1, 3, 3)
    )
    # for the clamped points, their density should be zero
    transparent_mask = get_transparent_mask(uv, signed_distance).reshape(bz, n_pts)

    weights = get_base_blending_weights(
        vsrc=pts,
        meshes=meshes,
        closest_face_idx=idx,
        smpl_blending_weights=smpl_blend_weight,
        face_idx=face_idx,
        bw_type=base_bw_type,
    )
    if ray_d != None and canonical_model != None:
        meshes_can = canonical_model["meshes"][idx.flatten()]
        pts_smpl_can = barycentric_map2can(uv, signed_distance, meshes_can)
        # map offset points in the ray to canonical space
        pts_ray_d = pts + ray_d
        uv, signed_distance = project_point2mesh(
            pts_ray_d.reshape(-1, 3), meshes=closest_meshes.reshape(-1, 3, 3)
        )
        pts_ray_d_can = barycentric_map2can(uv, signed_distance, meshes_can)
        ray_d_can = torch.nn.functional.normalize(pts_ray_d_can - pts_smpl_can, dim=-1)
        return weights, transparent_mask, ray_d_can

    else:
        return weights, transparent_mask


def trans_w(vsrc, vtar, vfeat, net, K=16):
    dist, idx, _ = knn_points(vsrc, vtar, K=K, return_nn=False)
    # nearest points with higher scores, [n_batch, num_points_1, K]
    score = torch.softmax(
        -dist / (torch.sum(dist, dim=-1, keepdim=True) + 1e-6), dim=-1
    )
    # import pdb;pdb.set_trace()

    B = idx.shape[0]
    if vfeat.shape[0] != B:
        vfeat = vfeat.expand(B, -1, -1)
    # [n_batch, num_points_1, K, dim]
    knn_feat = knn_gather(vfeat, idx)

    # fusion, [n_batch, num_points_1, K, 1] * [n_batch, num_points_1, K, dim] = [n_batch, num_points_1, K, dim]
    # sum([n_batch, num_points_1, K, dim], dim=-2) = [n_batch, num_points_1, dim]
    w_base = torch.sum(score.unsqueeze(-1) * knn_feat, dim=-2)

    # import pdb;pdb.set_trace()
    if B == 1:
        w_delta = batchify_idx(idx, net)
    else:
        w_delta = net(idx)

    w = torch.softmax(w_base + w_delta, dim=-1)
    # import pdb;pdb.set_trace()
    return w


def batchify_compute_uvfh(pts, meshes):
    chunk = int(5000 / pts.shape[0])
    all_ret = {}
    # import pdb;pdb.set_trace()
    # tic = time.time()
    for i in range(0, pts.shape[1], chunk):
        tmp = compute_uvfh(pts[:, i : i + chunk], meshes)
        for k in tmp:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(tmp[k])

    all_ret = {k: torch.cat(all_ret[k], 1) for k in all_ret}

    # print(time.time()-tic)

    return all_ret


def batchify_idx(idx, net, chunk=76800):
    ret = []
    for i in range(0, idx.shape[1], chunk):
        tmp = net(idx[:, i : i + chunk])
        ret.append(tmp)

    return torch.cat(ret, dim=1)

    # import pdb;pdb.set_trace()


## 把ray得到的结果放到指定位置
def post_process(source, mask, tgt_size):
    H, W, C = tgt_size
    tmp = torch.zeros(H * W, C)
    tmp[mask] = source.cpu()
    tmp = tmp.reshape(H, W, C)

    return tmp
