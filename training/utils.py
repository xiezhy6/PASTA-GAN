# coding:utf-8

import os
import cv2
import json
import numpy as np
import pycocotools.mask as maskUtils
import math

def get_mask_from_kps(kps, height, width):
    rles = maskUtils.frPyObjects(kps, height, width)
    rle = maskUtils.merge(rles)
    mask = maskUtils.decode(rle)[...,np.newaxis].astype(np.float32)
    mask = mask * 255.0
    return mask

def get_rectangle_mask(a, b, c, d, height, width):
    x1 = a + (b-d)/4
    y1 = b + (c-a)/4
    x2 = a - (b-d)/4
    y2 = b - (c-a)/4

    x3 = c + (b-d)/4
    y3 = d + (c-a)/4
    x4 = c - (b-d)/4
    y4 = d - (c-a)/4
    kps = [x1,y1,x2,y2]

    v0_x = c-a
    v0_y = d-b
    v1_x = x3-x1
    v1_y = y3-y1
    v2_x = x4-x1
    v2_y = y4-y1

    cos1 = (v0_x*v1_x+v0_y*v1_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v1_x*v1_x+v1_y*v1_y))
    cos2 = (v0_x*v2_x+v0_y*v2_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v2_x*v2_x+v2_y*v2_y))

    if cos1<cos2:
        kps.extend([x3,y3,x4,y4])
    else:
        kps.extend([x4,y4,x3,y3])

    kps = np.array(kps).reshape(1,-1).tolist()
    mask = get_mask_from_kps(kps, height, width)

    return mask


def get_hand_mask(hand_keypoints, height, width):
    # 肩，肘，腕
    s_x,s_y,s_c = hand_keypoints[0]
    e_x,e_y,e_c = hand_keypoints[1]
    w_x,w_y,w_c = hand_keypoints[2]

    up_mask = np.ones((height, width, 1))
    bottom_mask = np.ones((height, width, 1))
    if s_c > 0.1 and e_c > 0.1:
        up_mask = get_rectangle_mask(s_x, s_y, e_x, e_y, height, width)
        # 对上半部分进行膨胀操作，消除两部分之间的空隙
        kernel = np.ones((20,20),np.uint8)  
        up_mask = cv2.dilate(up_mask,kernel,iterations = 1)
        up_mask = (up_mask > 0).astype(np.float32)[...,np.newaxis]
    if e_c > 0.1 and w_c > 0.1:
        bottom_mask = get_rectangle_mask(e_x, e_y, w_x, w_y, height, width)
        bottom_mask = (bottom_mask > 0).astype(np.float32)

    return up_mask, bottom_mask


def get_palm_mask(hand_mask, hand_up_mask, hand_bottom_mask):
    inter_up_mask = (hand_mask + hand_up_mask == 2).astype(np.float32)
    inter_bottom_mask = (hand_mask + hand_bottom_mask == 2).astype(np.float32)
    palm_mask = hand_mask - inter_up_mask - inter_bottom_mask

    return palm_mask