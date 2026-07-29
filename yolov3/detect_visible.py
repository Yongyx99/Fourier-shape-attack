
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from yolov3.models.common import DetectMultiBackend
from yolov3.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov3.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov3.utils.plots import Annotator, colors, save_one_box
from yolov3.utils.torch_utils import select_device, time_sync
from yolov3.utils.augmentations import letterbox
import numpy as np
import torch.nn as nn
import PIL.Image as Image
from torchvision import transforms
from ipdb import set_trace as st

device = torch.device("cuda")
inputsize = [416,416]
trans = transforms.Compose([
    transforms.ToTensor(),
])

def load_visible_model():
    weights = "./yolov3/weights/visible.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model

def detect_visible(model,img):
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    H = len(img[0][0])
    W = len(img[0][0][0])
    img = nn.functional.interpolate(img, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
    img = img.cuda()
    pred = model(img)
    conf_thres=0.0001 # confidence threshold
    iou_thres=0.45
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
    if len(pred[0]) == 0:
        return None,0
    left = max(int(pred[0][0][0].item()),0)
    up = max(int(pred[0][0][1].item()),0)
    right = min(int(pred[0][0][2].item()),inputsize[0])
    below = min(int(pred[0][0][3].item()),inputsize[1])
    left = int(left*W/inputsize[0])
    up = int(up*H/inputsize[1])
    right = int(right*W/inputsize[0])
    below = int(below*H/inputsize[1])
    return [left,up,right,below],pred[0][0][4].clone().detach()


def detect_all(model,img):
    """对图片做检测
        model: yolov3模型
        img: 输入图片张量，shape为(N,C,H,W)
    """
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    orig_height = img.shape[2]
    orig_width = img.shape[3]
    img = nn.functional.interpolate(img, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
    img = img.cuda()
    pred = model(img)
    conf_thres=0.0001 # confidence threshold
    iou_thres=0.45
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
    if len(pred[0]) == 0:
        return None,0
    
    boxes, confs = [], []
    for detection in pred[0]:
        left_model = max(int(detection[0].item()),0)
        up_model = max(int(detection[1].item()),0)
        right_model = min(int(detection[2].item()),inputsize[0])
        below_model = min(int(detection[3].item()),inputsize[1])
        
        # 坐标缩放回原始图像尺寸
        left = int(left_model * orig_width / inputsize[1])
        up = int(up_model * orig_height / inputsize[0])
        right = int(right_model * orig_width / inputsize[1])
        below = int(below_model * orig_height / inputsize[0])

        # 获取置信度
        confidence = detection[4].item()

        # 添加结果到列表
        boxes.append((left, up, right, below))
        confs.append(confidence)


    return boxes,confs


def detect_visible_batch(model,img):
    """
    model: yolov3模型
    img: 输入图片张量，shape为(N,C,H,W)
    图像尺寸为416*416，满足yolov3输入要求
    """
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    batch_size = img.shape[0]
    orig_height = img.shape[2]
    orig_width = img.shape[3]
    conf_thres=0.01 # confidence threshold
    iou_thres=0.45

    img = nn.functional.interpolate(img, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
    img = img.detach().cuda()
    pred = model(img)
    
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)

    
    batch_boxes, batch_confs = [], []
    for i in range(batch_size):
        image_pred = pred[i]

        if image_pred is None or len(image_pred) == 0:
            batch_boxes.append([])
            batch_confs.append([])
            continue
        
        boxes, confs = [], []
        for detection in image_pred:
            # 获取模型输入尺寸下的物体坐标
            left_model = max(int(detection[0].item()),0)
            up_model = max(int(detection[1].item()),0)
            right_model = min(int(detection[2].item()),inputsize[0])
            below_model = min(int(detection[3].item()),inputsize[1])
            
            # 坐标缩放回原始图像尺寸
            left = int(left_model * orig_width / inputsize[1])
            up = int(up_model * orig_height / inputsize[0])
            right = int(right_model * orig_width / inputsize[1])
            below = int(below_model * orig_height / inputsize[0])

            # 获取置信度
            confidence = detection[4].item()

            # 添加结果到列表
            boxes.append((left, up, right, below))
            confs.append(confidence)
        
        # 当前结果放入批次结果中
        batch_boxes.append(boxes)
        batch_confs.append(confs)
    

    return batch_boxes,batch_confs


def focal_loss_for_zero(confidences, gamma=2.0, alpha=0.9, eps=1e-8):
    """
    约束向量值趋近于0的Focal Loss实现
    参数:
        confidences (Tensor): 待约束的向量 (值域[0,1])
        gamma (float): Focal Loss的调节因子，增大gamma会降低易分类样本的权重
        alpha (float): 正负样本平衡权重 (None表示自动设置)
        eps (float): 数值稳定因子
    
    返回:
        loss (Tensor): 计算得到的loss标量
    """
    # 确保输入值在合法范围内
    confidences = torch.clamp(confidences, min=eps, max=1-eps)
    
    # 目标标签：全0向量 (我们希望所有值趋近于0)
    targets = torch.zeros_like(confidences)
    
    # 自动设置alpha权重：当未指定时，根据非零元素比例动态调整
    if alpha is None:
        # 计算非零元素比例作为正样本权重
        nonzero_ratio = (confidences > 0.01).float().mean().item()
        alpha = max(0.2, min(0.8, nonzero_ratio))  # 限制在[0.2,0.8]范围内
    
    # 计算二元交叉熵损失 (不进行reduction)
    bce_loss = F.binary_cross_entropy(confidences, targets, reduction='none')
    
    # 计算概率因子pt
    # 对于目标为0的情况：pt = 1 - p (p为预测值)
    pt = torch.exp(-bce_loss)
    
    # 计算Focal Loss
    focal_term = (1 - pt) ** gamma
    
    # 应用alpha权重 (负样本权重 = 1-alpha)
    loss = focal_term * bce_loss * (alpha * targets + (1 - alpha) * (1 - targets))
    

    high_conf_mask = (confidences > 0.05).float()
    weighted_loss = loss * (1.0 + 50.0 * high_conf_mask)  # 大幅提高≈1样本的权重
    
    return weighted_loss.mean()


def calculate_confs_visible(model,img):
    """
    计算带梯度的置信度值
    Parameters:
        model: yolov3模型
        img: 输入图片张量，shape为(N,C,H,W)
    Returns:
        confs: 批次中每张图片的置信度列表,张量、带梯度
    """

    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    batch_size, orig_height, orig_width = img.shape[0], img.shape[2], img.shape[3]
    conf_thres=0.01 # confidence threshold
    iou_thres=0.45
    img = nn.functional.interpolate(img, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
    img = img.cuda()

    pred = model(img)       # torch.Size([1, box_number, 6]),其中第二维度6为[x1, y1, x2, y2, conf, cls]
    # 不经过NMS


    # pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
    confidences = pred[0,:, 4]
    confidence_mask = confidences > 0.1         # confidence_mask: tensor([False, False, False,  ..., False, False, False], device='cuda:0')
    if confidence_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    loss = confidences[confidence_mask].mean()       # 开始这里是.mean()
    return loss

