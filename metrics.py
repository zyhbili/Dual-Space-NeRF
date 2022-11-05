import torch
from kornia.losses import ssim as dssim
import numpy as np
import cv2
# from skimage.metrics import structural_similarity
from skimage.measure import compare_ssim

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value=(image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def mae(image_pred, image_gt):
    value=torch.abs(image_pred-image_gt)
    return  torch.mean(value)

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim_metric(rgb_pred, rgb_gt, mask_at_box):
    H, W = rgb_gt.shape[:2]
    mask_at_box = mask_at_box.reshape(H, W).cpu().numpy()
    # convert the pixels into an image
    img_pred = np.zeros((H, W, 3))
    img_pred[mask_at_box] = rgb_pred[mask_at_box]
    img_gt = np.zeros((H, W, 3))
    img_gt[mask_at_box] = rgb_gt[mask_at_box]
    # crop the object region
    x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
    img_pred = img_pred[y:y + h, x:x + w]
    img_gt = img_gt[y:y + h, x:x + w]
    # compute the ssim
    # ssim = structural_similarity(img_pred, img_gt, multichannel=True)
    ssim = compare_ssim(img_pred, img_gt, multichannel=True)
    return ssim