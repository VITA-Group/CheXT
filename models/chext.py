import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
import numpy as np
from functools import partial

import os
import sys
import argparse
import cv2
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp, Block
from .radiomic import extract_radiomic_features

class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion',
               'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']

avg_size = np.array([[411.8, 512.5, 219.0, 139.1], [348.5, 392.3, 479.8, 381.1],
					 [396.5, 415.8, 221.6, 318.0], [394.5, 389.1, 294.0, 297.4],
					 [434.3, 366.7, 168.7, 189.8], [502.4, 458.7, 71.9, 70.4],
					 [378.7, 416.7, 276.5, 304.5], [369.3, 209.4, 198.9, 246.0]])

def get_IoU(truth_coords, pred_coords):
    # coords of intersection rectangle
    x1 = max(truth_coords[0], pred_coords[0])
    y1 = max(truth_coords[1], pred_coords[1])
    x2 = min(truth_coords[2], pred_coords[2])
    y2 = min(truth_coords[3], pred_coords[3])
    # area of intersection rectangle
    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    # area of prediction and truth rectangles
    boxTruthArea = (truth_coords[2] - truth_coords[0] + 1) * (truth_coords[3] - truth_coords[1] + 1)
    boxPredArea = (pred_coords[2] - pred_coords[0] + 1) * (pred_coords[3] - pred_coords[1] + 1)
    # intersection over union 
    iou = interArea / float(boxTruthArea + boxPredArea - interArea)
    return iou

def get_box(list_BB):
    avg_intensities = [np.mean(bb) for bb in list_BB]

    idx = np.argmax(avg_intensities)

    bb = list_BB[idx]

    # if bb is entire image, clip it
    if bb == [0, 0, 224, 224]:
        bb = [24, 24, 200, 200]

    return bb

def generate_mask(attention):
    blur = cv2.GaussianBlur(np.uint8(255 * attention),(5,5),0)
    T, mask = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask, 1, 2)

    list_BB = []
    for item in range(len(contours)):
        cnt = contours[item]
        x,y,w,h = cv2.boundingRect(cnt)
        poly_coords = [cnt]

        list_BB.append([x, y, x+w, y+h])
    bbox_coords = get_box(list_BB)
    bbox = np.zeros(mask.shape)
    bbox = cv2.rectangle(bbox, (bbox_coords[0], bbox_coords[1]), (bbox_coords[2], bbox_coords[3]), 255, -1)
    return bbox

class CheXT(nn.Module):
    def __init__(self, ViT, RaMLP, num_branches, num_classes, embed_dim, norm_layer=nn.LayerNorm, unsupervised=False) -> None:
        super().__init__()
        self.unsupervised = unsupervised

        self.ViT = ViT
        self.RaMLP = RaMLP
        self.norm = nn.ModuleList([norm_layer(embed_dim[i])
                                  for i in range(num_branches)])
        self.proj = nn.ModuleList([Mlp(in_features=embed_dim[i], hidden_features=32,
                                  out_features=64, act_layer=nn.GELU) for i in range(num_branches)])

        if not self.unsupervised:
            self.image_head = nn.Linear(embed_dim[0], num_classes)
            self.rad_head = nn.Linear(embed_dim[1], num_classes)
            self.combine_head = nn.Linear(embed_dim[0]+embed_dim[1], num_classes)

        self.apply(self._init_weights)

    # Froze some unused Classifier Layer: rad_head and combine_head
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_feature(self, x, threshold=None):
        B, C, H, W = x.shape
        input = x
        img_out, attn = self.ViT(input)
        nh = attn.shape[1] # number of heads

        batch_attentions = attn[:, :, 0, 1:].reshape(B, nh, 14, 14)
        batch_attentions = nn.functional.interpolate(batch_attentions, scale_factor=16, mode="nearest").cpu().detach().numpy()
        batch_attentions = batch_attentions.mean(axis=1)  # mean along nh dimension

        if threshold is not None:
            batch_percentiles = np.percentile(batch_attentions, threshold, axis=0)
            batch_attentions = [np.clip(batch_attentions[i], a_min=batch_percentiles[i], a_max=None) for i in range(B)]

        normalize = lambda x: (x - x.min()) / (x.max() - x.min())
        batch_attentions = np.array(list(map(normalize, batch_attentions)))
        masks = np.array(list(map(generate_mask, batch_attentions)))

        radiomic_features = extract_radiomic_features(input.data.cpu().numpy(), masks)
        radiomic_features = radiomic_features.float().to(input.device)
        rad_out = self.RaMLP(radiomic_features)
        img_out = self.norm[0](img_out)
        rad_out = self.norm[1](rad_out)
        return img_out, rad_out

    def forward(self, x, threshold=None):
        img_embedding, rad_embedding = self.forward_feature(x, threshold)
        z_img = self.proj[0](img_embedding)
        z_rad = self.proj[1](rad_embedding)
        zs = [z_img, z_rad]

        if self.unsupervised:
            return None, None, None, zs

        i_logits = self.image_head(img_embedding)
        r_logits = self.rad_head(rad_embedding)
        f_logits = self.combine_head(torch.concat([img_embedding, rad_embedding], dim=1))
        return i_logits, r_logits, f_logits, zs