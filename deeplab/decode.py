from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from utils import _gather_feat, _tranpose_and_gather_feat
import numpy as np
def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def match_rel_box(rel, bbox, K_rel, K_bbox):
    rel  = rel.view(K_rel,1)
    rel = rel.repeat(1, K_bbox)
    bbox = bbox.view(1, K_bbox)
    bbox = bbox.repeat(K_rel, 1)
    dis_mat = abs(rel - bbox)
    return dis_mat

def hoidet_decode(heat_obj, K_obj=5):
    batch, cat_obj, height, width = heat_obj.size() #1, 1, 119, 119
    heat_obj = _nms(heat_obj)

    scores_obj, inds_obj, clses_obj, ys_obj, xs_obj = _topk(heat_obj, K=K_obj)
    scores_obj = scores_obj.view(batch, K_obj, 1)
    clses_obj = clses_obj.view(batch, K_obj, 1).float()

    return scores_obj, xs_obj, ys_obj
