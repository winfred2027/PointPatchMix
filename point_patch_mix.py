import numpy as np
import torch
import time
import cv2
import os
from emd import emd_module as emd

correspond_type = 'center'
mix_level = 'patch'
target_type = 'score'


def points_correspond(points_a, points_b, data_b, data_c, remd):
    # voted by all points of each group
    B, num_group, group_size, _ = points_a.size()
    points_a = points_a.reshape(B, -1, 3)
    points_b = points_b.reshape(B, -1, 3)

    dis, ind = remd(points_a, points_b, 0.005, 300)
    for b in range(B):
        idx = (ind[b] / group_size).long().reshape(num_group, group_size)
        data_c[b, :, :, :] = data_b[b, true_id, :, :]
    return data_c


def centers_correspond(centers_a, centers_b, data_b, data_c, remd):
    # decided by the center of each group
    B = centers_a.size(0)
    centers_a = centers_a.reshape(B, -1, 3)  # [B,G,3]
    centers_b = centers_b.reshape(B, -1, 3)

    dis, ind = remd(centers_a, centers_b, 0.005, 300)
    for b in range(B):
        true_id = ind[b].long()
        data_c[b, :, :, :] = data_b[b, true_id, :, :]
    return data_c


def block_mix(lam, center):
    B, num_group, _ = center.size()
    b_group = int(num_group * lam)
    b_group = max(1, b_group)
    block_group_mask = torch.zeros([B, num_group])

    select_center = torch.from_numpy(np.random.choice(num_group, B, replace=False, p=None))
    ind1 = torch.tensor(range(B))
    query = center[ind1, select_center].view(B, 1, 3)  
    dist = torch.sqrt(torch.sum((center - query.repeat(1, num_group, 1)) ** 2, 2))
    idxs = dist.topk(b_group, dim=1, largest=False, sorted=True).indices

    for b in range(B):
        block_group_mask[b, idxs[b]] = 1 
    block_group_mask = block_group_mask.to(torch.bool)  # [B, G]
    return block_group_mask, b_group


def patch_mix(lam, center):
    B, num_group, _ = center.size()
    b_group = int(num_group * lam)
    b_group = max(1, b_group)
    for b in range(B):
        i = torch.randperm(num_group)
        a_group_mask = torch.zeros([num_group - b_group])
        b_group_mask = torch.ones([b_group])
        mask = torch.cat((a_group_mask, b_group_mask), dim=0)[i].unsqueeze(0)
        if (b == 0):
            random_group_mask = mask
        else:
            random_group_mask = torch.cat((random_group_mask, mask), dim=0)
    random_group_mask = random_group_mask.to(torch.bool)  # [B, G]
    return random_group_mask, b_group




def point_patch_mix(data, label, beta):

    lam = np.random.beta(beta, beta)

    neighborhood = data[:, :, 1:, 0:3]
    center = data[:, :, 0, 0:3]
    points = neighborhood + center.unsqueeze(2)  # [B, G, M, 3]
    B, num_group, group_size, _ = points.size()
    scores = data[:, :, 0, 3]  # [B, G]

    '''Determine paired two point clouds'''
    rand_index = torch.randperm(B).cuda()

    data_a = data.clone()
    data_b = data[rand_index]
    data_c = data[rand_index].clone()
    label_a = label
    label_b = label[rand_index]

    centers_a = center
    centers_b = center[rand_index]

    points_a = points
    points_b = points[rand_index]
    scores_a = scores.clone()

    data_a, data_b, data_c = data_a.to('cuda'), data_b.to('cuda'), data_c.to('cuda')
    points_a, points_b = points_a.to('cuda'), points_b.to('cuda')


    '''Two different ways of patch assignment(Centers/Points)'''
    remd = emd.emdModule()
    remd = remd.cuda()
    if correspond_type == 'center':
        data_c = centers_correspond(centers_a, centers_b, data_b, data_c, remd)  # [B, G, 1+M, 3+1]
    else:
        data_c = points_correspond(points_a, points_b, data_b, data_c, remd)
    
    scores_c = data_c[:, :, 0, 3]  # [B, G, 1+M]


    '''Two different Mixing levels(Patch/Block)'''
    if mix_level == 'patch':
        group_mask, b_group = patch_mix(lam, center)
    else:
        group_mask, b_group = block_mix(lam, center)
    
    data_a[group_mask] = data_c[group_mask]  # [B,G,1+M,3+1]
    

    '''Two different targets for generation'''
    if target_type == 'score':
        rate_a = torch.sum(scores_a[~group_mask].reshape(B, -1), dim=1)  # [B]
        rate_b = torch.sum(scores_c[group_mask].reshape(B, -1), dim=1)
    else:
        rate_a = (num_group - b_group) * 1.0 / num_group
        rate_b = 1 - rate_a
        rate_a = torch.tensor([rate_a]).repeat(B).to('cuda')
        rate_b = torch.tensor([rate_b]).repeat(B).to('cuda')


    return data_a, label_a, label_b, rate_a, rate_b


if __name__ == '__main__':
    B = 32
    G = 64
    M = 32
    center = torch.rand(B, G, 3)
    neighborhood = torch.rand(B, G, M, 3)
    score = torch.rand(B, G)
    points_data = torch.cat((center.unsqueeze(2), neighborhood), dim=2)
    data = torch.cat((points_data, score.unsqueeze(2).unsqueeze(3).repeat(1, 1, 1+M, 1)), dim=3)
    label = torch.rand(B)
    beta = 1.5

    # point_patch_mix
    new_points, label_a, label_b, rate_a, rate_b = point_patch_mix(data, label, beta, idx)
    print(new_points)