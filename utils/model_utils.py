
import sys
sys.path.append('../')
from model import DualSpaceNeRF


def select_model(cfg):
    if cfg.MODEL.TYPE == "nerf":
        return DualSpaceNeRF(cfg)
    else:
        raise Exception("error")
    
