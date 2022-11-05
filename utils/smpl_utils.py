import pickle, os

def load_bodydata(model_type = "smpl", gender="neutral", model_path="group/zju_mocap/models/smpl/smpl/models"):
    if os.path.isdir(model_path):
        model_fn = '{}_{}.{ext}'.format(model_type.upper(), gender.upper(), ext='pkl')
        smpl_path = os.path.join(model_path, model_fn)
    else:
        smpl_path = model_path
    assert os.path.exists(smpl_path), 'Path {} does not exist!'.format(
        smpl_path)

    with open(smpl_path, 'rb') as smpl_file:
        data = pickle.load(smpl_file, encoding='latin1')
    return data