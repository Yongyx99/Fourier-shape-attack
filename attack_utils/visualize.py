"""
Functions for Fourier shape
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace as st
import time
from torchvision import transforms
import random
import cv2
from PIL import Image, ImageDraw, ImageFont

# drawing parameters
box_color = (0, 255, 0)  # 绿色
text_color = (255, 255, 255)  # 黑色

def vis_image_boxes(images: torch.Tensor, boxes: torch.Tensor, confs: torch.Tensor,
                    vis_conf_threshold: float, save_path: str):
    """
    Visualization of predicted boxes and confidences. Used for Wei. proposed YOLOv3.
    Parames:
        images: image tensor for visualization. torch.Tensor([1, C, H, W])
        boxes: predicted boxes for yolov3 model on images.
        confs: predicted confidences for yolov3 model on images.
        vis_conf_threshold: threshold confidence for visualization.
        save_path: path to save images.
    """
    for i in range(len(images)):
        image = images[i]
        box_i = boxes[i]
        conf_i = confs[i]
        image = image * 255.0
        image_np = image.clone().detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        image_np = np.ascontiguousarray(image_np)
        
        for j in range(len(box_i)):
            box, conf = box_i[j], conf_i[j]
            if conf < vis_conf_threshold:
                continue
            left, up, right, below = box
            cv2.rectangle(image_np, (left, up), (right, below), box_color, 2)
            label = f"Confidence: {conf:.2f}"
            cv2.putText(image_np, label, (left, up + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        # bgr_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(save_path, bgr_image)

        img_np = Image.fromarray(image_np)
        img_np.save(save_path)

def vis_image_boxes_mmdet(image_file: str, boxes: torch.Tensor, confs: torch.Tensor, 
                          vis_conf_threshold:float, save_path: str):
    """
    Visualization of predicted boxes and confidences. Used for MMDetection models.
    Only used for 'person' class.
    Params:
        image_file: path to image.
        boxes: predicted boxes.
        confs: predicted confidences.
        vis_conf_threshold: threshold confidence for visualization.
        save_path: path to save images.
    """
    keep_indices = confs > vis_conf_threshold
    filtered_scores = confs[keep_indices]
    filtered_boxes = boxes[keep_indices]
    object_number = len(filtered_boxes)
    filterd_labels = ['person'] * object_number
    # save visualization results
    img = Image.open(image_file)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for box, score, label in zip(filtered_boxes, filtered_scores, filterd_labels):
        x_min, y_min, x_max, y_max = box.tolist()
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)], 
            outline=box_color, 
            width=2
        )
        # label text
        label_text = f"{label}:{score:.2f}"
        # adding text
        text_bbox = draw.textbbox((x_min, y_min), label_text, font=font)
        draw.rectangle(
            [(text_bbox[0]-2, text_bbox[1]-2), (text_bbox[2]+2, text_bbox[3]+2)],
            fill=box_color
        )
        # draw text
        draw.text(
            (x_min, y_min),
            label_text,
            fill=text_color,
            font=font
        )
    # 保存图片
    img.save(save_path)
    
