from easymocap.smplmodel import load_model
import os, json
import numpy as np
import trimesh
from copy import deepcopy

def parse_json(path):
    with open(path,'r') as f:
        content = f.read()
    data = json.loads(content)[0]
    return data

def smplh2smpl(pose):
    assert len(pose) == 156
    # import pdb;pdb.set_trace()

    pose = pose.reshape(-1,3)
    ret = np.zeros((24,3))
    ret[0:22,:] = pose[0:22]
    ret[23,:] = pose[22]
    ret[22,:] = pose[37]
    # ret[0,:] = 0
    return ret.flatten()

if __name__ == "__main__":
    body_model = load_model(model_type="smpl")

 
    # for zju performer
    data_base = '/group/h36m/processed/h36m'
    save_base = '/group/projects/smpl_nerf/novel_poses_3dv'
    # data_base = '/storage/group/gaoshh/h36m/processed/h36m'
    Humans = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]

    for performer in ["CoreView_313", "CoreView_315","CoreView_377","CoreView_386","CoreView_387","CoreView_392","CoreView_393", "CoreView_394"]:
        performer = "CoreView_313"
        src_param_path = f"/group/zju_mocap/{performer}/new_params/1.npy"
        src_param = np.load(src_param_path, allow_pickle = True).item()
        src_shapes = src_param['shapes']
        src_Th0 = src_param['Th']
        src_Rh0 = src_param['Rh']

        coreview = "S9"

        tgt_Th0 = None

        savedir_vertices = f'{save_base}/{performer}_{coreview}/new_vertices'
        savedir_params = f'{save_base}/{performer}_{coreview}/new_params'

        if not os.path.exists(savedir_vertices):
            os.makedirs(savedir_vertices, exist_ok=True)
        if not os.path.exists(savedir_params):
            os.makedirs(savedir_params, exist_ok=True)

        for fn in os.listdir(f'{data_base}/{coreview}/Posing/new_params'):
            save_path_vertices = f'{savedir_vertices}/{fn}'
            save_path_params = f'{savedir_params}/{fn}'
            param_path = f'{data_base}/{coreview}/Posing/new_params/{fn}'
            print(param_path)
            param = np.load(param_path, allow_pickle = True).item()
            param['shapes'] = src_shapes

            if tgt_Th0 is None:
                tgt_Th0 = param['Th']
            
            param['Th'] = src_Th0 
            param['Rh'] = src_Rh0
            vertices = body_model(return_verts=True, return_tensor=False, **param)
            np.save(save_path_vertices, vertices)
            np.save(save_path_params, param)
        break


    # for h36m performer
    # data_base = '/group/zju_mocap/'

    # for performer in ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']:
    # # performer = "S5"
    #     src_param_path = f'/group/h36m/processed/h36m/{performer}/Posing/new_params/0.npy'
    #     src_param = np.load(src_param_path, allow_pickle = True).item()
    #     src_shapes = src_param['shapes']
    #     src_Th0 = src_param['Th']
    #     coreview = "CoreView_393"
    #     tgt_Th0 = None

    #     savedir_vertices = f'{save_base}/{performer}_{coreview}/new_vertices'
    #     savedir_params = f'{save_base}/{performer}_{coreview}/new_params'

    #     if not os.path.exists(savedir_vertices):
    #         os.makedirs(savedir_vertices, exist_ok=True)
    #     if not os.path.exists(savedir_params):
    #         os.makedirs(savedir_params, exist_ok=True)
        
    #     for fn in os.listdir(f'{data_base}/{coreview}/new_params'):
    #         save_path_vertices = f'{savedir_vertices}/{fn}'
    #         save_path_params = f'{savedir_params}/{fn}'

    #         param_path = f'{data_base}/{coreview}/new_params/{fn}'
    #         print(param_path)
    #         param = np.load(param_path, allow_pickle = True).item()
    #         param['shapes'] = src_shapes

    #         if tgt_Th0 is None:
    #             tgt_Th0 = param['Th']

    #         # param['Th'] =  (param['Th'] - tgt_Th0) + src_Th0 
    #         vertices = body_model(return_verts=True, return_tensor=False, **param)
    #         np.save(save_path_vertices, vertices)
    #         np.save(save_path_params, param)


 
   


    