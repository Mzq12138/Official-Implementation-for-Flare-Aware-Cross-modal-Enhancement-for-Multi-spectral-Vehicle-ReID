import torch
import torch.nn as nn

class MMIC_loss(nn.Module):
    def __init__(self) -> None:
        pass

    def forward(self, feat_list, target):
        pass

def normalize(feat, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      feat: pytorch Variable
    Returns:
      feat: pytorch Variable, same shape as input
    """
    feat = 1. * feat / (torch.norm(feat, 2, axis, keepdim=True).expand_as(feat) + 1e-12)
    return feat

def hetero_loss(f_list, label, margin=0.1):
    label_num = len(label.unique())
    chunk_f_list = [f.chunk(label_num, 0) for f in f_list]
    l = len(f_list)

    dist = 0
    dist_func = nn.MSELoss(reduction='sum')
    center_list = []
    for l_idx in range(label_num):
        for f in chunk_f_list:
            center_list.append(torch.mean(f[l_idx], 0))
        for i in range(l):
            for j in range(i+1, l):
                dist += max(margin, dist_func(center_list[i], center_list[j]))
        
    return 2*dist/(l*(l-1))

def MMIC_LOSS(f_list, label, margin):
    label_num = len(label.unique())
    chunk_f_list = [f.chunk(label_num, 0) for f in f_list]
    l = len(f_list)

    dist = 0
    for i in range(l):
        f = [chunk_f_list[n][i] for n in range(l)]
        dist += MultiModalIdConsistLoss(f, margin=margin)
    return dist

def MultiModalIdConsistLoss(f_list, margin=0.1):
    # feat should belong to same class
    N_f = f_list[0].shape[0]
    N_m = len(f_list)

    center_list = []
    for i in range(N_f):
        center_f = 0
        for n in range(N_m):
            center_f += f_list[n][i]
        center_f = center_f/N_m
        center_list.append(center_f)
    
    l = len(center_list)
    assert l >= 2
    dist_func = nn.MSELoss(reduction='sum')
    dist = 0
    for i in range(l):
        for j in range(i+1, l):
            dist += max(margin, dist_func(center_list[i], center_list[j]))
    
    return 2*dist/(l*(l-1))

def cdc(f_list, label, margin=0.1):
    return hetero_loss(f_list, label, margin=0.1) + MMIC_LOSS(f_list, label, margin= 0.1)