import torch.nn as nn
import torch.nn.functional as F

def make_loss(cfg):
    if cfg.MODEL.LOSS == 'L2':
        return MSELoss(cfg)
    elif cfg.MODEL.LOSS == 'L1':
        return SmoothL1Loss(cfg)


class MSELoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cfg = cfg
    def forward(self, inputs, batch):
        loss_rgb = self.mse(inputs['color'], batch["rgb"].reshape(-1, 3).cuda())
        ret =  {
            "loss_rgb": loss_rgb,
        }
        if self.cfg.MODEL.LOSSwMask:
            acc_map = inputs['acc_map']
            occupancy = batch['occupancy'].reshape(-1).cuda()
            acc_map[occupancy==1] = 1
            loss_mask = F.l1_loss(acc_map, occupancy)
            ret.update({
                'loss_mask':0.1 * loss_mask
            })
        return ret

class SmoothL1Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.l1 = nn.SmoothL1Loss()
        self.cfg = cfg
    def forward(self, inputs, batch):
        loss_rgb = self.l1(inputs['color'], batch["rgb"].reshape(-1, 3).cuda())
        ret =  {
            "loss_rgb": loss_rgb,
        }
        if self.cfg.MODEL.LOSSwMask:
            acc_map = inputs['acc_map']
            occupancy = batch['occupancy'].reshape(-1).cuda()
            acc_map[occupancy==1] = 1
            loss_mask = F.l1_loss(acc_map, occupancy)
            ret.update({
                'loss_mask': 0.1*loss_mask
            })
        return ret