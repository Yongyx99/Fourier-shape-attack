
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

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
def load_infrared_model():
    """
    load yolov3 infrared model
    """
    weights = "./yolov3/weights/infrared.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model

def load_uap_defense_infrared_model():
    weights = "/home/Adversarial/advshape/case_study/defence/AdvAugmentation/YOLOv3Defense_uap/exp/weights/best.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model

def load_infp_defense_infrared_model():
    weights = "/home/Adversarial/advshape/case_study/defence/AdvAugmentation/YOLOv3Defense_infp/exp/weights/best.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model

def load_defp_defense_infrared_model():
    weights = "/home/Adversarial/advshape/case_study/defence/AdvAugmentation/YOLOv3Defense_defp/exp/weights/best.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model

def load_visible_model():
    """
    load yolov3 visible model
    """
    weights = "./yolov3/weights/visible.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model

def detect(model,img):
    """
    predict image detection results. suitable for both infrared and visible model. 
    * obtain 1 object with highest confidence. 
    Args:
        model: yolov3
        img: input image, shape: (1, C, H, W). only 1 image each time.
    Returns:
        results: [x1, y1, x2, y2], confidences
    """
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
    # right = int(pred[0][0][2].item())
    # below = int(pred[0][0][3].item())
    right = min(int(pred[0][0][2].item()),inputsize[0])
    below = min(int(pred[0][0][3].item()),inputsize[1])
    left = int(left*W/inputsize[0])
    up = int(up*H/inputsize[1])
    right = int(right*W/inputsize[0])
    below = int(below*H/inputsize[1])
    return [left,up,right,below],pred[0][0][4].clone().detach()

def detect_all(model,img):
    """
    predict image detection results. suitable for both infrared and visible model. 
    * obtain all objects on 1 image.
    Args:
        model: yolov3 model
        img: input image tensor, shape: (1, C, H, W)
    Returns:
        results: list([x1, y1, x2, y2], ...), list(confidence1, ...)
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
        # coordinates rescale to origin image
        left = int(left_model * orig_width / inputsize[1])
        up = int(up_model * orig_height / inputsize[0])
        right = int(right_model * orig_width / inputsize[1])
        below = int(below_model * orig_height / inputsize[0])
        # obtain confidence
        confidence = detection[4].item()
        # add to list
        boxes.append((left, up, right, below))
        confs.append(confidence)
    return boxes,confs

def detect_AP(model, img, conf_thr=0.01):
    """
    calculating AP. return all results including low confidence results
    Args:
        model: yolov3 model
        img: input image tensor, shape: (1, C, H, W)
    Returns:
        results
    """
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    H = len(img[0][0])
    W = len(img[0][0][0])
    img = nn.functional.interpolate(img, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
    img = img.detach().cuda()       # 要加一个detach,否则CUDA out of memory
    pred = model(img)
    iou_thres=0.45
    pred = non_max_suppression(pred, conf_thr, iou_thres, None, False, max_det=1000)
    return pred

def detect_batch(model, imgs):
    """
    predict batch images detection results. suitable for both infrared and visible model. 
    * obtain all objects on batch images.
    Args:
        model: yolov3 model
        imgs: input images tensor, shape: (N, C, H, W)
    Returns:
        batch_boxes: list([[x1, y1, x2, y2], ...], ...)
        batch_confs: list([[confidence1, ...], ...)
    """
    if len(imgs.shape) == 3:
        imgs = imgs[None]  # expand for batch dim
    batch_size = imgs.shape[0]
    orig_height, orig_width = imgs.shape[2], imgs.shape[3]
    conf_thres=0.01 # confidence threshold
    iou_thres=0.45
    imgs = nn.functional.interpolate(imgs, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
    imgs = imgs.detach().cuda()
    pred = model(imgs)
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
            # obtain resized coordinates. On [416, 416] images
            left_model = max(int(detection[0].item()),0)
            up_model = max(int(detection[1].item()),0)
            right_model = min(int(detection[2].item()),inputsize[0])
            below_model = min(int(detection[3].item()),inputsize[1])
            # coordinates rescale to origin image
            left = int(left_model * orig_width / inputsize[1])
            up = int(up_model * orig_height / inputsize[0])
            right = int(right_model * orig_width / inputsize[1])
            below = int(below_model * orig_height / inputsize[0])
            # obtain confidences
            confidence = detection[4].item()
            # add results to list
            boxes.append((left, up, right, below))
            confs.append(confidence)
        # add results to batch list
        batch_boxes.append(boxes)
        batch_confs.append(confs)
    return batch_boxes,batch_confs

def calculate_confs(model, imgs, thresh=0.01):
    """
    calculate confidence loss with gradient
    Args:
        model: yolov3
        imgs: input images tensor, shape: (N, C, H, W)
    Returns:
        conf_loss: confidence loss based on predicted results
    """
    if len(imgs.shape) == 3:
        imgs = imgs[None]  # expand for batch dim
    batch_size, orig_height, orig_width = imgs.shape[0], imgs.shape[2], imgs.shape[3]
    imgs = nn.functional.interpolate(imgs, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
    imgs = imgs.cuda()
    preds = model(imgs)     # here input imgs should range from 0~1
    confidences = preds[0,:, 4]
    confidence_mask = confidences > thresh         # confidence_mask: tensor([False, False, False,  ..., False, False, False], device='cuda:0')
    if confidence_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    loss = confidences[confidence_mask].mean()
    return loss


def calculate_confs_v2(model, imgs, q=0.3):
    """
    calculate confidence loss with gradient.
    use new version of calculating mean. giving a ratio q, when calculating mean, only use confidences > confidences.max() * q
    Args:
        model: yolov3
        imgs: input images tensor, shape: (N, C, H, W)
    Returns:
        conf_loss: confidence loss based on predicted results
    """
    if len(imgs.shape) == 3:
        imgs = imgs[None]  # expand for batch dim
    batch_size, orig_height, orig_width = imgs.shape[0], imgs.shape[2], imgs.shape[3]
    imgs = nn.functional.interpolate(imgs, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
    imgs = imgs.cuda()
    preds = model(imgs)     # here input imgs should range from 0~1
    confidences = preds[0,:, 4]
    # new version of calculating confidence loss
    select_conf_thresh = confidences.max() * q
    mask = confidences >= select_conf_thresh        # confidence_mask: tensor([False, False, False,  ..., False, False, False], device='cuda:0')
    confidences_filted = confidences[mask]
    if len(confidences_filted) == 0:
        conf_loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        conf_loss = confidences_filted.mean()
    return conf_loss





def detect_train(model,img):
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    img_ = nn.functional.interpolate(img, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False).cuda()
    pred = model(img_)
    sorted, rank = torch.sort(pred[0,:,4],descending=True)
    loss = torch.sum(pred[0][rank[:100]],dim=0)[4]
    pred = non_max_suppression(pred, 0.05, 0.45, None, False, max_det=1000)
    return -loss