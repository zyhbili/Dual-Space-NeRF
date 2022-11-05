import sys

sys.path.append("../")
from dataloader import Mocap, Mocap_view, Mocap_infer,H36M, H36M_test
# from zju_mocap_dataset import Mocap, Mocap_view, Mocap_infer
# from h36m_dataset import 
import yaml
from torch.utils.data import DataLoader
# from h36m_dataset_test import H36M_test


class MyCfg:
    def __init__(self) -> None:
        pass


def set_my_cfg(mycfg, data_config):
    if type(data_config) == dict:
        for key in data_config.keys():
            if type(data_config[key]) == dict:
                tmp = MyCfg()
                set_my_cfg(tmp, data_config[key])
                setattr(mycfg, key, tmp)
            else:
                setattr(mycfg, key, data_config[key])
    return mycfg


def select_dataset(cfg, train_nrays=2000, formal_test=False):
    yaml_path = f"data_configs/{cfg.DATASETS.TYPE}/{cfg.DATASETS.HUMAN}.yml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        cont = f.read()
    data_config = yaml.load(cont)

    if cfg.DATASETS.TYPE == "zju_mocap":
        if formal_test:
            tmp = data_config["Train"]
            train_max_frame = tmp["end"] - tmp["begin"] + 1
            tmp = data_config["Test"]
            test_set_novel_view = Mocap_infer(cfg.DATASETS.HUMAN,tmp["ratio"],tmp["begin"], tmp["end"],data_config["Train"]["views"],train_max_frame, tmp["intv"], tmp["novel_pose_begin"], novel_pose=False)
            test_set_novel_pose = Mocap_infer(cfg.DATASETS.HUMAN,tmp["ratio"],tmp["begin"], tmp["end"],data_config["Train"]["views"],train_max_frame, tmp["intv"], tmp["novel_pose_begin"], novel_pose=True)

            print("novel view length", len(test_set_novel_view))
            print("novel pose length", len(test_set_novel_pose))
            return test_set_novel_view, test_set_novel_pose

        tmp = data_config["Train"]
        train_max_frame = tmp["end"] - tmp["begin"] + 1
        train_set = Mocap(
            cfg.DATASETS.HUMAN,
            tmp["ratio"],
            train_nrays,
            tmp["begin"],
            tmp["end"],
            tmp["views"],
            data_dir = cfg.DATASETS.ZJU_MOCAP_PATH
        )
        tmp = data_config["Val"]
        val_set = Mocap_view(
            cfg.DATASETS.HUMAN,
            tmp["ratio"],
            tmp["begin"],
            tmp["end"],
            data_config["Train"]["views"],
            train_max_frame,
            interval=tmp["intv"],
            data_dir = cfg.DATASETS.ZJU_MOCAP_PATH
        )

    elif cfg.DATASETS.TYPE == "h36m":
        mycfg = MyCfg()
        mycfg = set_my_cfg(mycfg, data_config)
        data_dir = cfg.DATASETS.H36M_PATH
        data_root = f"{data_dir}/{cfg.DATASETS.HUMAN}/Posing"
        ann_file = f"{data_dir}/{cfg.DATASETS.HUMAN}/Posing/annots.npy"

        if formal_test:
            test_set_novel_view = H36M_test(mycfg, data_root, cfg.DATASETS.HUMAN, ann_file, "test", train_nrays, test_novel_pose=False, is_eval=True, is_formal=True,)
            test_set_novel_pose = H36M_test(mycfg, data_root, cfg.DATASETS.HUMAN, ann_file, "test", train_nrays, test_novel_pose=True, is_eval=True, is_formal=True,)

            print("novel view length", len(test_set_novel_view))
            print("novel pose length", len(test_set_novel_pose))
            return test_set_novel_view, test_set_novel_pose
        train_set = H36M(
            mycfg,
            data_root,
            cfg.DATASETS.HUMAN,
            ann_file,
            "train",
            train_nrays,
            test_novel_pose=False,
            is_eval=False,
        )
        val_set = H36M(
            mycfg,
            data_root,
            cfg.DATASETS.HUMAN,
            ann_file,
            "test",
            train_nrays,
            test_novel_pose=True,
            is_eval=True,
            is_formal=False,
        )
    print("len train:", len(train_set), ", len val:", len(val_set))

    return train_set, val_set


def load_yml_as_cfg(yml_path):
    with open(yml_path, "r", encoding="utf-8") as f:
        cont = f.read()
    data_config = yaml.load(cont)
    mycfg = MyCfg()
    mycfg = set_my_cfg(mycfg, data_config)

    return mycfg

