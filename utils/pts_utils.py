import torch

def uniform_sampling(ray_o, ray_d, n_pts, near, far, perturb, is_training):
    t_vals = torch.linspace(0., 1., steps=n_pts).to(near)
    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals
    if perturb > 0. and is_training:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(upper)
        z_vals = lower + (upper - lower) * t_rand
    pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

    return pts, z_vals

def geometry_guided_ray_marching(ray_o, ray_d, n_pts, near, far, xyz, perturb, is_training, gamma = 0.05):
    ## ray_o,ray_d (B, nrays, 3)
    ## near, far (B, nrays)
    ## xyz (B,10475,3)
    norm_ray = torch.norm(ray_d,dim=-1)
    ray_d_unit = ray_d/norm_ray[...,None]

    xyz = xyz.to(ray_o)
    B, n_rays = ray_o.shape[:2]
    z_min = torch.ones(B, xyz.shape[1], n_rays).to(ray_o) *99999
    z_max = -z_min

    # tic = time.time()
    z_0 = torch.einsum('bvs,brs->bvrs',xyz - ray_o[:,0:1], ray_d_unit).sum(-1)

    tmp = ((xyz - ray_o[:,0:1])**2).sum(-1, keepdim=True) - z_0**2 ##(B, v, ray)
    inside = tmp < gamma**2
    delta_z = (gamma**2 - tmp[inside]) ** 0.5

    z_min[inside] = z_0[inside] - delta_z
    z_max[inside] = z_0[inside] + delta_z

    z_min = z_min.min(dim=1)[0]
    z_max = z_max.max(dim=1)[0]

    z_min = z_min/norm_ray 
    z_max = z_max/norm_ray
    
    mask1 = inside.any(dim=1)
    mask2 = z_min<z_max
    mask = torch.logical_and(mask1, mask2)
    
    # import pdb;pdb.set_trace()

    near[mask] = z_min[mask]
    far[mask] = z_max[mask]

    pts, z_vals = uniform_sampling(ray_o, ray_d, n_pts, near, far, perturb, is_training)
    # print(time.time()-tic)
    
    return pts, z_vals
    # for v_idx in range(xyz.shape[1]):
    #     tic = time.time()
    #     z_0 = ((xyz[:, v_idx: v_idx+1, :] - ray_o) * ray_d)

    #     import pdb;pdb.set_trace()

    #     tmp = ((xyz[:, v_idx: v_idx+1, :] - ray_o)**2).sum(-1) - z_0**2


    #     mask = tmp< gamma**2
    #     delta_z = (gamma**2 - tmp[mask]) ** 0.5

    #     cond1 = z_0[mask] + delta_z 
    #     cond2 = z_0[mask] - delta_z
    #     mask_sub1 = cond1 > z_max[mask]
    #     mask_sub2 = cond2 < z_min[mask]

    #     t_max = z_max[mask].clone()
    #     t_min = z_min[mask].clone()
    #     t_max[mask_sub1] = cond1[mask_sub1]
    #     t_min[mask_sub2] = cond2[mask_sub2]

    #     z_max[mask] = t_max
    #     z_min[mask] = t_min 
    #     print(time.time()-tic)
    # miss_mask = z_min>z_max
    # import pdb;pdb.set_trace()
