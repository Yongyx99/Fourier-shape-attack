"""
loading datasets
"""
import numpy as np
import torch
from ipdb import set_trace as st
import os
import cv2
from torchvision import transforms

class LLVIPDataloader:
    """
    load single image
    """
    def __init__(self, img_dir, label_dir, batch_size=1):
        """
        img_dir: images directory
        label_dir: labels directory
        batch_size: batch size
        image_size: image size
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.num_samples = len(self.image_files)
        self.indices = np.arange(self.num_samples)

        self.current_index = 0
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def reset(self):
        self.current_index = 0
    
    def __iter__(self):
        self.reset()
        return self

    def _load_and_preprocess_image(self, img_path):
        # read image
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"cannot open image file: {img_path}")
        # RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # torch.Size([channels, height, width])
        tensor_img = self.trans(img)
        return tensor_img
    
    def _load_labels(self, img_name):
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.label_dir, f"{base_name}.txt")
        img_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id, obj_center_x, obj_center_y, obj_w, obj_h = \
                            int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    img_labels.append([class_id, obj_center_x, obj_center_y, obj_w, obj_h])
        return img_labels
    
    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration
        end_index = min(self.current_index + self.batch_size, self.num_samples)
        # initializing batch
        batch_images = []   # [tensor1, tensor2, tensor3,...]，torch.Size([channels, height, width])
        batch_labels = []  # [batch, image_objects, obj_info]
        batch_basenames = []

        for i in range(self.current_index, end_index):
            img_name = self.image_files[i]
            img_path = os.path.join(self.img_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            
            img = self._load_and_preprocess_image(img_path)
            batch_images.append(img)
            
            labels = self._load_labels(img_name)
            batch_labels.append(labels)

            batch_basenames.append(base_name)
        
        # list to tensor (batch, channels, height, width)
        image_tensor = torch.stack(batch_images, dim=0)
        self.current_index = end_index
        
        return image_tensor, batch_labels, batch_basenames       # image_tensor after normalization


class LLVIPDataloader_resize:
    def __init__(self, img_dir, label_dir, batch_size=16):
        """
        img_dir: images directory
        label_dir: labels directory
        batch_size: batch size
        image_size: image size
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.target_size = (334, 224)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.num_samples = len(self.image_files)
        self.indices = np.arange(self.num_samples)

        self.current_index = 0
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def reset(self):
        self.current_index = 0
    
    def __iter__(self):
        self.reset()
        return self

    def _load_and_preprocess_image(self, img_path):
        """
        load images
        Args:
            img_path: image path
        Returns:
            tensor_img: tensor image, RGB mode, range~[0, 1], torch.Size([C, H, W])
        """
        # read image
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"cannot open image file: {img_path}")
        # RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        # 将图像缩放到target_size上
        img_resized = cv2.resize(img, (self.target_size[1], self.target_size[0]))
        # torch.Size([channels, height, width])
        tensor_img = self.trans(img_resized)
        return tensor_img
    
    def _load_labels(self, img_name):
        """
        load labels, on resized [334, 224] images
        """
        base_name = os.path.splitext(img_name)[0]
        orig_img_path = os.path.join(self.img_dir, img_name)
        orig_img = cv2.imread(orig_img_path)
        orig_img_h, orig_img_w = orig_img.shape[:2]
        target_h, target_w = self.target_size
        scale_h, scale_w = target_h / orig_img_h, target_w / orig_img_w

        label_path = os.path.join(self.label_dir, f"{base_name}.txt")
        img_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id, obj_center_x, obj_center_y, obj_w, obj_h = \
                            int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    # resize
                    obj_center_x_resize, obj_center_y_resize, obj_w_resize, obj_h_resize = \
                            obj_center_x * scale_w, obj_center_y * scale_h, obj_w * scale_w, obj_h * scale_h
                    img_labels.append([class_id, obj_center_x_resize, obj_center_y_resize, obj_w_resize, obj_h_resize])
        return img_labels
    
    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration
        end_index = min(self.current_index + self.batch_size, self.num_samples)
        # initializing batch
        batch_images = []   # [tensor1, tensor2, tensor3,...]，torch.Size([channels, height, width])
        batch_labels = []  # [batch, image_objects, obj_info]
        batch_basenames = []

        for i in range(self.current_index, end_index):
            img_name = self.image_files[i]
            img_path = os.path.join(self.img_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            
            img = self._load_and_preprocess_image(img_path)
            batch_images.append(img)
            
            labels = self._load_labels(img_name)
            batch_labels.append(labels)

            batch_basenames.append(base_name)
        
        # list to tensor (batch, channels, height, width)
        image_tensor = torch.stack(batch_images, dim=0)
        self.current_index = end_index
        
        return image_tensor, batch_labels, batch_basenames       # image_tensor after normalization