import numpy as np
import torchvision

import torch.nn.functional as F
import cv2

def split(a, n):
    k, m = divmod(len(a), n)
    return [ a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n) ]

def vis_images(batch):

    images = []
    for key in batch:
        img = torchvision.utils.make_grid( batch[key], normalize=True, range=(-1,1), nrow=8).permute(1,2,0).data.cpu().numpy()
        images.append(img)
    return np.concatenate(images, axis=0)
    
def select_dict(dict, keys):
    return {key:dict[key] for key in dict if key in keys}

def mask_dict(dict, mask):

    dict_new = {}
    for key in dict:
        dict_new[key] = dict[key][mask]

    return dict_new

def index_dict(dict, start, end):

    for key in dict:
        dict[key] = dict[key][start:end]

    return dict


def grid_sample_feat(feat_map, x):

    n_batch, n_point, _ = x.shape

    if feat_map.ndim == 4:
        x = x[:,:,None,:2]
    elif feat_map.ndim == 5:
        x = x[:,:,None,None,:3]

    feats = F.grid_sample(feat_map, x, align_corners=True, mode='bilinear',padding_mode='zeros')

    return feats.reshape(n_batch, -1, n_point).transpose(1,2)

def expand_cond(cond, x):

    cond = cond[:, None]
    new_shape = list(cond.shape)
    new_shape[0] = x.shape[0]
    new_shape[1] = x.shape[1]
    
    return cond.expand(new_shape)

def rectify_pose(pose, rot):
    """
    Rectify AMASS pose in global coord adapted from https://github.com/akanazawa/hmr/issues/50.
 
    Args:
        pose (72,): Pose.
    Returns:
        Rotated pose.
    """
    pose = pose.copy()
    R_rot = cv2.Rodrigues(rot)[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    # new_root = np.linalg.inv(R_abs).dot(R_root)
    new_root = R_rot.dot(R_root)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose


class Dict2Class(object):
    def __init__(self, dict):
        for key in dict:
            setattr(self, key, dict[key])
  
