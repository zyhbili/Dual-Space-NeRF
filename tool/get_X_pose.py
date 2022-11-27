from easymocap.smplmodel import load_model
import os, json
import numpy as np

def parse_json(path):
    with open(path,'r') as f:
        content = f.read()
    data = json.loads(content)[0]
    return data

if __name__ == "__main__":
    data_base = '/group/zju_mocap/' #TODO
    coreviews = ['CoreView_313', 'CoreView_377', 'CoreView_386', 'CoreView_387', 'CoreView_390', 'CoreView_392', 'CoreView_393', 'CoreView_394']
    body_model = load_model(model_type="smpl")
 
    # Get X_smpl_vertices with new params
    for coreview in coreviews:
        save_path = f'{data_base}/{coreview}/X_smpl_vertices.npy'
        for fn in os.listdir(f'{data_base}/{coreview}/new_params'):
            param_path = f'{data_base}/{coreview}/new_params/{fn}'
            print(coreview)
            param = np.load(param_path, allow_pickle = True).item()
            param['Rh'] *= 0
            param['Th'] *= 0
            x_pose = np.zeros((1,24,3))
            x_pose[:,1,2] += 0.6
            x_pose[:,2,2] -= 0.6
            param['poses'] = x_pose.reshape(1, 72)
            vertices = body_model(return_verts=True, return_tensor=False, **param)
            np.save(save_path, vertices)
            break

    
    ## Get X_smpl_vertices with new params for h36m
    # data_base = '/group/h36m/processed/h36m' #TODO
    # Humans = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]

    # for coreview in Humans:
    #     save_path = f'{data_base}/{coreview}/Posing/lbs/X_smpl_vertices.npy'
    #     for fn in os.listdir(f'{data_base}/{coreview}/Posing/new_params'):
    #         param_path = f'{data_base}/{coreview}/Posing/new_params/{fn}'
    #         print(coreview)
    #         param = np.load(param_path, allow_pickle = True).item()
    #         param['Rh'] *= 0
    #         param['Th'] *= 0
    #         x_pose = np.zeros((1,24,3))
    #         x_pose[:,1,2] += 0.6
    #         x_pose[:,2,2] -= 0.6
    #         param['poses'] = x_pose.reshape(1, 72)
    #         vertices = body_model(return_verts=True, return_tensor=False, **param)
    #         np.save(save_path, vertices)
    #         break


