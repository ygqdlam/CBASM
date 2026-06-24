import numpy as np
import random
import torch
import scipy.stats as stats


# # # # # # # # # # # # # # # # # # # # # 
# # 0. random box
# # # # # # # # # # # # # # # # # # # # # 
def rand_bbox(size, lam=None):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)


    return bbx1, bby1, bbx2, bby2


# # # # # # # # # # # # # # # # # # # # # 
# # 1. cutmix for 2d
# # # # # # # # # # # # # # # # # # # # # 
# def cut_mix(unlabeled_image, unlabeled_mask, unlabeled_logits):
#     mix_unlabeled_image = unlabeled_image.clone()
#     mix_unlabeled_target = unlabeled_mask.clone()
#     mix_unlabeled_logits = unlabeled_logits.clone()
    
#     # get the random mixing objects
#     u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
#     # print(u_rand_index)
    
#     # get box
#     u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
#     # cut & paste
#     for i in range(0, mix_unlabeled_image.shape[0]):
#         mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
#         # label is of 3 dimensions
# #         mix_unlabeled_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
# #             unlabeled_mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
#         mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
#         mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

#     del unlabeled_image, unlabeled_mask, unlabeled_logits

#     return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits
def cut_mix_2d_co(unlabeled_image, unlabeled_mask_1, unlabeled_logits_1, unlabeled_mask_2, unlabeled_logits_2, target_ulb, unlabeled_conflict=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target_1 = unlabeled_mask_1.clone()
    mix_unlabeled_logits_1 = unlabeled_logits_1.clone()
    mix_unlabeled_target_2 = unlabeled_mask_2.clone()
    mix_unlabeled_logits_2 = unlabeled_logits_2.clone()
    mix_target_ulb = target_ulb.clone()
    if unlabeled_conflict is not None:
        mix_unlabeled_conflict = unlabeled_conflict.clone()

    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    # print(u_rand_index)

    # get box
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))

    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        # label is of 3 dimensions
        #         mix_unlabeled_target_1[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
        #             unlabeled_mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        mix_unlabeled_target_1[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask_1[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_logits_2[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits_2[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        mix_unlabeled_target_2[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask_2[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_target_ulb[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_target_ulb[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_logits_1[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits_1[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        if unlabeled_conflict is not None:
            mix_unlabeled_conflict[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                unlabeled_conflict[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    if unlabeled_conflict is not None:
        del unlabeled_image, unlabeled_mask_1, unlabeled_logits_1, unlabeled_conflict
        return mix_unlabeled_image, mix_unlabeled_target_1, mix_unlabeled_logits_1, mix_unlabeled_conflict

    del unlabeled_image, unlabeled_mask_1, unlabeled_logits_1,unlabeled_mask_2, unlabeled_logits_2,target_ulb
    return mix_unlabeled_image, mix_unlabeled_target_1, mix_unlabeled_logits_1,mix_unlabeled_target_2, mix_unlabeled_logits_2,mix_target_ulb


def cut_mix_array(unlabeled_image, unlabel_array):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_array = []
    for arr in unlabel_array:
        mix_arr = arr.clone()
        mix_array.append(mix_arr)
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

    # get box
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))

    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        for k in range(len(mix_array)):
            mix_array[k][i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                unlabel_array[k][u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del unlabeled_image, unlabel_array
    return mix_unlabeled_image, mix_array
def cut_mix(unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    if unlabeled_conflict is not None:
        mix_unlabeled_conflict = unlabeled_conflict.clone()
    
    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    # print(u_rand_index)
    
    # get box
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        # label is of 3 dimensions
#         mix_unlabeled_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        if unlabeled_conflict is not None:
            mix_unlabeled_conflict[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                unlabeled_conflict[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    if unlabeled_conflict is not None:
        del unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict
        return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, mix_unlabeled_conflict

    del unlabeled_image, unlabeled_mask, unlabeled_logits
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits


# # # # # # # # # # # # # # # # # # # # # 
# # 2. cutmix for 3d
# # # # # # # # # # # # # # # # # # # # # 

def rand_bbox_3d(size, lam=None):
    # img: B x C x H x W x D, lb: B x H x W x D
    if len(size) == 5:
        W = size[2]
        H = size[3]
        D = size[4]
    elif len(size) == 4:
        W = size[1]
        H = size[2]
        D = size[3]
    else:
        raise Exception
    B = size[0]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cut_d = int(D * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)
    cz = np.random.randint(size=[B, ], low=int(D/8), high=D)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbz1 = np.clip(cz - cut_d // 2, 0, D)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbz2 = np.clip(cz + cut_d // 2, 0, D)


    return bbx1, bby1, bbz1, bbx2, bby2, bbz2

def cut_mix_3d_t(unlabeled_image, unlabeled_mask, unlabeled_logits, GT,unlabeled_conflict=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_gt = GT.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    if unlabeled_conflict is not None:
        mix_unlabeled_conflict = unlabeled_conflict.clone()

    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

    # get box
    # img: B x C x H x W x D, lb: B x H x W x D
    u_bbx1, u_bby1, u_bbz1, u_bbx2, u_bby2, u_bbz2 = rand_bbox_3d(unlabeled_image.size(), lam=np.random.beta(4, 4))

    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        mix_gt[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            GT[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        if unlabeled_conflict is not None:
            mix_unlabeled_conflict[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                unlabeled_conflict[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

    if unlabeled_conflict is not None:
        del unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict,GT
        return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, mix_gt,mix_unlabeled_conflict

    del unlabeled_image, unlabeled_mask, unlabeled_logits,GT
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits,mix_gt

def cut_mix_3d(unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    if unlabeled_conflict is not None:
        mix_unlabeled_conflict = unlabeled_conflict.clone()
    
    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

    # get box
    # img: B x C x H x W x D, lb: B x H x W x D
    u_bbx1, u_bby1, u_bbz1, u_bbx2, u_bby2, u_bbz2 = rand_bbox_3d(unlabeled_image.size(), lam=np.random.beta(4, 4))

    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
        if unlabeled_conflict is not None:
            mix_unlabeled_conflict[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                unlabeled_conflict[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
    if unlabeled_conflict is not None:
        del unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict
        return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, mix_unlabeled_conflict

    del unlabeled_image, unlabeled_mask, unlabeled_logits
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits

def cut_mix_3d_array(unlabeled_image, unlabel_array):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_array = []
    for arr in unlabel_array:
        mix_arr = arr.clone()
        mix_array.append(mix_arr)
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

    # get box
    u_bbx1, u_bby1, u_bbz1, u_bbx2, u_bby2, u_bbz2 = rand_bbox_3d(unlabeled_image.size(), lam=np.random.beta(4, 4))

    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        for k in range(len(mix_array)):
            mix_array[k][i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                unlabel_array[k][u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

    del unlabeled_image, unlabel_array
    return mix_unlabeled_image, mix_array
def cut_mix_3d_co_aut(unlabeled_image, unlabeled_mask_1, unlabeled_logits_1, unlabeled_mask_2=None, unlabeled_logits_2=None,target_ulb=None, unlabeled_conflict=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target_1 = unlabeled_mask_1.clone()
    mix_unlabeled_logits_1 = unlabeled_logits_1.clone()
    if unlabeled_mask_2 is not None:
        mix_unlabeled_target_2 = unlabeled_mask_2.clone()
        mix_unlabeled_logits_2 = unlabeled_logits_2.clone()
    if target_ulb is not None:
        mix_target_ulb = target_ulb.clone()
    if unlabeled_conflict is not None:
        mix_unlabeled_conflict = unlabeled_conflict.clone()

    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

    # get box
    # img: B x C x H x W x D, lb: B x H x W x D
    u_bbx1, u_bby1, u_bbz1, u_bbx2, u_bby2, u_bbz2 = rand_bbox_3d(unlabeled_image.size(), lam=np.random.beta(4, 4))

    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        mix_unlabeled_target_1[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_mask_1[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        mix_unlabeled_logits_1[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_logits_1[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        if unlabeled_mask_2 is not None:
            mix_unlabeled_target_2[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                unlabeled_mask_2[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

            mix_unlabeled_logits_2[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                unlabeled_logits_2[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        if target_ulb is not None:
            mix_target_ulb[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                target_ulb[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        if unlabeled_conflict is not None:
            mix_unlabeled_conflict[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                unlabeled_conflict[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

    if unlabeled_conflict is not None:
        del unlabeled_image, unlabeled_mask_1, unlabeled_logits_1, unlabeled_conflict
        return mix_unlabeled_image, mix_unlabeled_target_1, mix_unlabeled_logits_1, mix_unlabeled_conflict
    if unlabeled_mask_2 is not None:
        del unlabeled_image, unlabeled_mask_1, unlabeled_logits_1, unlabeled_mask_2, unlabeled_logits_2
        return mix_unlabeled_image, mix_unlabeled_target_1, mix_unlabeled_logits_1,mix_unlabeled_target_2, mix_unlabeled_logits_2,mix_target_ulb
    del unlabeled_image, unlabeled_mask_1, unlabeled_logits_1, unlabeled_mask_2, unlabeled_logits_2
    return mix_unlabeled_image, mix_unlabeled_target_1, mix_unlabeled_logits_1,mix_target_ulb

def cut_mix_3d_co_aut_3(unlabeled_image, unlabeled_mask_1, unlabeled_logits_1, unlabeled_mask_2=None, unlabeled_logits_2=None,unlabeled_mask_3=None, unlabeled_logits_3=None,target_ulb=None, unlabeled_conflict=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target_1 = unlabeled_mask_1.clone()
    mix_unlabeled_logits_1 = unlabeled_logits_1.clone()
    if unlabeled_mask_2 is not None:
        mix_unlabeled_target_2 = unlabeled_mask_2.clone()
        mix_unlabeled_logits_2 = unlabeled_logits_2.clone()
        mix_unlabeled_target_3 = unlabeled_mask_3.clone()
        mix_unlabeled_logits_3 = unlabeled_logits_3.clone()
    if target_ulb is not None:
        mix_target_ulb = target_ulb.clone()
    if unlabeled_conflict is not None:
        mix_unlabeled_conflict = unlabeled_conflict.clone()

    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

    # get box
    # img: B x C x H x W x D, lb: B x H x W x D
    u_bbx1, u_bby1, u_bbz1, u_bbx2, u_bby2, u_bbz2 = rand_bbox_3d(unlabeled_image.size(), lam=np.random.beta(4, 4))

    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        mix_unlabeled_target_1[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_mask_1[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        mix_unlabeled_logits_1[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_logits_1[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        if unlabeled_mask_2 is not None:
            mix_unlabeled_target_2[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                unlabeled_mask_2[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

            mix_unlabeled_logits_2[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                unlabeled_logits_2[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
            mix_unlabeled_target_3[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                unlabeled_mask_3[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

            mix_unlabeled_logits_3[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                unlabeled_logits_3[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        if target_ulb is not None:
            mix_target_ulb[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                target_ulb[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        if unlabeled_conflict is not None:
            mix_unlabeled_conflict[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                unlabeled_conflict[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

    if unlabeled_conflict is not None:
        del unlabeled_image, unlabeled_mask_1, unlabeled_logits_1, unlabeled_conflict
        return mix_unlabeled_image, mix_unlabeled_target_1, mix_unlabeled_logits_1, mix_unlabeled_conflict
    if unlabeled_mask_2 is not None:
        del unlabeled_image, unlabeled_mask_1, unlabeled_logits_1, unlabeled_mask_2, unlabeled_logits_2, unlabeled_mask_3, unlabeled_logits_3
        return mix_unlabeled_image, mix_unlabeled_target_1, mix_unlabeled_logits_1,mix_unlabeled_target_2, mix_unlabeled_logits_2,mix_unlabeled_target_3, mix_unlabeled_logits_3,mix_target_ulb
    del unlabeled_image, unlabeled_mask_1, unlabeled_logits_1, unlabeled_mask_2, unlabeled_logits_2
    return mix_unlabeled_image, mix_unlabeled_target_1, mix_unlabeled_logits_1,mix_target_ulb
def cut_mix_3d_co(unlabeled_image, unlabeled_label_s, unlabeled_label_ema):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target_s = unlabeled_label_s.clone()
    mix_unlabeled_target_ema = unlabeled_label_ema.clone()

    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

    # get box
    # img: B x C x H x W x D, lb: B x H x W x D
    u_bbx1, u_bby1, u_bbz1, u_bbx2, u_bby2, u_bbz2 = rand_bbox_3d(unlabeled_image.size(), lam=np.random.beta(4, 4))

    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        mix_unlabeled_target_s[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_label_s[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        mix_unlabeled_target_ema[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_label_ema[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]


    del unlabeled_image, unlabeled_label_s, unlabeled_label_ema
    return mix_unlabeled_image, mix_unlabeled_target_s, mix_unlabeled_target_ema


def cut_mix_ldata(labeled_image, labeled_mask):
    mix_labeled_image = labeled_image.clone()
    mix_labeled_target = labeled_mask.clone()

    # get the random mixing objects
    u_rand_index = torch.randperm(labeled_image.size()[0])[:labeled_image.size()[0]]

    # get box
    # img: B x C x H x W x D, lb: B x H x W x D
    u_bbx1, u_bby1, u_bbz1, u_bbx2, u_bby2, u_bbz2 = rand_bbox_3d(labeled_image.size(), lam=np.random.beta(4, 4))

    # cut & paste
    for i in range(0, mix_labeled_image.shape[0]):
        mix_labeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            labeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        mix_labeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            labeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

    del labeled_image, labeled_mask
    return mix_labeled_image, mix_labeled_target

def context_mask3d(img, mask_ratio):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    w = np.random.randint(0, img_x - patch_pixel_x)#随机生成一个int型数值
    h = np.random.randint(0, img_y - patch_pixel_y)
    z = np.random.randint(0, img_z - patch_pixel_z)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0#w h z的作用是将要裁剪的mask随机移动一点距离，使其不靠近边界
    loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0#loss含有batch，得到batch中的每个图片对应的mask
    return mask.long(), loss_mask.long()
def context_mask2d(img, mask_ratio):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * mask_ratio), int(img_y * mask_ratio)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()

def cut_mix_bcp(unlabel_list,label_list,sub_bs,is3D=True):
    mix_unlabel_list_a = []
    mix_label_list_a = []
    mix_unlabel_list_b = []
    mix_label_list_b = []
    for item in unlabel_list:
        mix_item = item.clone()
        mix_unlabel_list_a.append(mix_item[:sub_bs])
        mix_unlabel_list_b.append(mix_item[sub_bs:])
    for item in label_list:
        mix_item = item.clone()
        mix_label_list_a.append(mix_item[:sub_bs])
        mix_label_list_b.append(mix_item[sub_bs:])
    if is3D:
        img_mask, loss_mask = context_mask3d(mix_label_list_a[0], 2/3)
    else:
        img_mask, loss_mask = context_mask2d(mix_label_list_a[0], 2/3)
    mix_bgl_list = []
    mix_bgu_list = []
    for i in range(len(mix_label_list_a)):
        mix_bgl = mix_label_list_a[i]* img_mask+mix_unlabel_list_a[i]*(1-img_mask)
        mix_bgl_list.append(mix_bgl)
    for i in range(len(mix_label_list_b)):
        mix_bgu = mix_label_list_b[i]* (1-img_mask)+mix_unlabel_list_b[i]*img_mask
        mix_bgu_list.append(mix_bgu)
    mix_img_targ_log_gt = []
    for i in range(len(mix_bgl_list)):
        mix_img_targ_log_gt.append(torch.cat((mix_bgl_list[i],mix_bgu_list[i])))
    del unlabel_list,label_list
    # del mix_unlabel_list_a,mix_label_list_a,mix_label_list_b,mix_unlabel_list_b
    return mix_img_targ_log_gt

def bcp_3d(args,weak_batch):
    num_lb = args["labeled_bs"]
    batch_size = args["batch_size"]
    num_ulb = batch_size - num_lb
    if args.get("flag_sampling_based_on_lb", False):
        sub_bs = num_lb // 2
        img_a, img_b = weak_batch[:sub_bs].clone(), weak_batch[sub_bs:num_lb].clone()
        unimg_a, unimg_b = weak_batch[num_lb:num_lb + sub_bs].clone(), weak_batch[num_lb + sub_bs:].clone()
    else:
        sub_bs = num_ulb // 2
        img_a, img_b = weak_batch[num_ulb:num_ulb + sub_bs].clone(), weak_batch[num_ulb + sub_bs:].clone()
        unimg_a, unimg_b = weak_batch[:sub_bs].clone(), weak_batch[sub_bs:num_ulb].clone()

    img_mask, loss_mask = context_mask(img_a,
                                       args["mask_ratio"])
    mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)  # 有标签数据作为背景
    mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)  # 无标签数据作为前景
    mix_unlabeled_image = torch.cat((mixl_img,mixu_img))
    return mix_unlabeled_image, loss_mask
