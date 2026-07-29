
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
    weights = "/home/Adversarial/Repository/Cross-modal_Patch_Attack-main/yolov3/weights/infrared.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model

def load_advshape_defense_infrared_model():
    weights = "/home/Adversarial/advshape/case_study/defence/AdvAugmentation/YOLOv3Defense/exp/weights/best.pt"
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
    
def load_randomshape_aug_infrared_model():
    weights = "/home/Adversarial/advshape/yolov3/runs/train/randomshape_aug/weights/best.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model

def load_fouriershape_aug_infrared_model():
    """使用fourier shape进行1个epoch的微调"""
    weights = "/home/Adversarial/advshape/yolov3/runs/train/fouriershape_aug/weights/best.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model


def detect_infrared(model,img):
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

def detect_AP(model, img, conf_thr=0.01):
    """
    用于计算AP，返回NMS之后的全部检测结果，包括低置信度结果
    """
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    H = len(img[0][0])
    W = len(img[0][0][0])
    img = nn.functional.interpolate(img, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
    # 要加一个detach,否则CUDA out of memory
    img = img.detach().cuda()
    pred = model(img)
    iou_thres=0.45
    pred = non_max_suppression(pred, conf_thr, iou_thres, None, False, max_det=1000)
    return pred

def detect_infrared_batch(model,img,conf_thres=0.1):
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
    # conf_thres=0.5
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

def calculate_confs_infrared(model,img):
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
    pred = model(img)
    # 不经过NMS
    # pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
    confidences = pred[0,:, 4]
    confidence_mask = confidences > 0.1         # confidence_mask: tensor([False, False, False,  ..., False, False, False], device='cuda:0')
    if confidence_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    loss = confidences[confidence_mask].mean()       # 开始这里是.mean()
    return loss
    


if __name__ == '__main__':
    threat_infrared_model = load_infrared_model()
    img_size = (1024, 1280)   # (H, W)
    img_h, img_w = img_size[0], img_size[1]
    vis_conf_threshold = 0.1

    # 标签设置
    box_color = (0, 255, 0)
    text_color = (255, 255, 255)

    # 确保输出文件夹存在
    intput_folder = '/home/AdvDatasets/LLVIP/infrared/test'
    output_folder = '/home/AdvDatasets/LLVIP/infrared_detection_results'
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, f'*.{ext}')))
    
    print(f"Found {len(image_paths)} images in {input_folder}")
    
    # 图像预处理转换
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
    ])
    
    for img_path in image_paths:
        # 读取图像
        img = Image.open(img_path).convert('RGB')
        orig_img = np.array(img)  # 保存原始图像用于可视化
        
        # 预处理
        img_tensor = preprocess(img).unsqueeze(0)  # 添加batch维度 [1, C, H, W]
        
        # 模型推理
        batch_boxes, batch_confs = detect_infrared_batch(model, img_tensor)
        
        # 获取当前图像的检测结果
        boxes = batch_boxes[0]
        confs = batch_confs[0]
        
        # 创建可视化图像 (使用原始图像尺寸)
        vis_img = orig_img.copy()
        
        # 绘制检测结果
        for j in range(len(boxes)):
            box, conf = boxes[j], confs[j]
            if conf < vis_conf_threshold:
                continue
                
            left, up, right, below = box
            # 绘制边界框
            cv2.rectangle(vis_img, (left, up), (right, below), box_color, 2)
            # 绘制置信度标签
            label = f"Conf: {conf:.2f}"
            cv2.putText(vis_img, label, (left, up - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        # 保存可视化结果
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        print(f"Processed: {filename} -> Detected {len(boxes)} objects")