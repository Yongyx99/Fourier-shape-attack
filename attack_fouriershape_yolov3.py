"""
Base class of Fourier shape attack and its variants
Based on YOLOv3
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
import os
import cv2
from torchvision import transforms
import stat
import json
import random
import cmath
from PIL import Image
from attack_utils.visualize import vis_image_boxes, vis_image_boxes_mmdet
from attack_utils.apply_utils import PatchTransformer, PatchApplier
from base import BaseFourierAttack
# YOLOv3, env = 'yolofuse'
from attack_utils.datasets import LLVIPDataloader
from yolov3.detect_custom import detect_batch, calculate_confs_v2, load_infrared_model, load_visible_model


cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(6)


class FourierAttack_InputSpecific_Infrared(BaseFourierAttack):
    """
    Fourier shape attack (input specific) on infrared images
    """
    def  __init__(self, model, workspace):
        super(FourierAttack_InputSpecific_Infrared, self).__init__(model, workspace)
        self.stop_threshold = 0.1
        self.stop_count_threshold = 10
        # saving folder
        self.log_files = os.path.join(self.workspace, 'logs')
        for path in [self.log_files]:
            try:
                # create directory (if already exists, ignore)
                os.makedirs(path, mode=0o777, exist_ok=True)
                # confirm permission (some system need to set separately)
                os.chmod(path, 0o777)
                print(f"Successfully create new file: {path}")
            except Exception as e:
                print(f"Warning: create {path} fail: {str(e)}")
    
    def train(self, batch_data):
        """
        Input specific Fourier shape adversarial patch optimization.
        Args:
            batch_data (torch.Tensor): 1 batch of image data, i.e. 1 image
        """
        # initialization 
        count_below_threshold = 0
        # batch_data unfold
        images, labels, basenames = batch_data      # batch_data: (image, label, basename)
        # initialize
        c = torch.tensor(self.c_init, dtype=torch.complex64).to(self.device)
        learnable_c = nn.Parameter(c, requires_grad=True)
        optimizer = torch.optim.Adam([learnable_c], lr=self.lr_init, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, 
        #     milestones=[1800],
        #     gamma=0.2
        # )
        img_size = (images.shape[-2], images.shape[-1])
        basename_elem = basenames[0].split('_')[:3]
        basename = '_'.join(basename_elem)
        # set logger
        loss_log_path = os.path.join(self.log_files, f'log_{basename}.txt')
        open_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        mode = 0o777
        fd = os.open(loss_log_path, open_flags, mode)
        log_file = open(loss_log_path, 'a')
        # empty cache
        torch.cuda.empty_cache()
        training_epochs = 0
        for epoch in range(self.num_epochs):
            training_epochs += 1
            # generate G_values_t
            G_values_t = self.generate_G(learnable_c)
            # batch processing
            image_t = images.squeeze(0).to(self.device)       # torch.Size([3, 416, 416])
            label = labels[0]
            # generating mask
            patch_bboxes = self.generate_patch_boxes_t(label)
            if patch_bboxes.size(0) == 0:
                st()
                continue
            num_patches = patch_bboxes.shape[0]
            # generating adv_batch
            adv_batch = G_values_t.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(self.device)     # torch.Size([patch_number,3, patch_size, patch_size])
            adv_transformed, msk_transformed = self.transformer.resize_rotate(
                adv_batch, patch_bboxes, img_size
            )       # adv_transformed: torch.Size[(num_patches, 3, img_h, img_w)], msk_transformed: torch.Size[(num_patches, 1, img_h, img_w)]
            adv_img_melt, _ = torch.max(adv_transformed, dim=0) # torch.Size([num_patches, 3, img_h, img_w]) melting
            # infrared，apply black fourier patch
            applied_image = self.applier.forward_infrared(image_t, adv_img_melt)
            adv_examples_t = applied_image.unsqueeze(0)
            # Calculate confidence loss
            conf_loss = calculate_confs_v2(self.model, adv_examples_t)   # 可见光
            # regularization loss
            reg_loss = self.fourier_constraints(learnable_c)
            # total loss calculation
            loss = self.conf_weight * conf_loss + self.reg_weight * reg_loss       # loss_area权重在boundary_aware_area_loss函数中调
            # log
            print(f"Epoch [{epoch+1}/{self.num_epochs}], conf-loss: {conf_loss.item():.4f}, reg-loss: {reg_loss.item():.4f}, total-loss: {loss.item():.4f}")
            log_file.write(f"Epoch [{epoch+1}/{self.num_epochs}] "
                           f"conf-loss: {conf_loss.item():.4f} "
                           f"reg-loss: {reg_loss.item():.4f} "
                           f"total-loss: {loss.item():.4f}\n"
                           # f"lr: {scheduler.get_last_lr()[0]:.4f}\n"
                           )
            log_file.flush()
            # convergence condition
            if conf_loss.item() < self.stop_threshold:
                count_below_threshold += 1
            else:
                count_below_threshold = 0
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update learning rate
            # scheduler.step() 
            # stop condition
            if count_below_threshold >= self.stop_count_threshold:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        return basename, learnable_c, adv_examples_t, training_epochs

    def save(self, basename, learnable_c, adv_examples_t):
        """
        save images, adversairla examples, fourier curve c
        only save final results.
        Args:
            basename: image name. basename format: 'XXXXXX_instance_id'
            learnable_c: learnable_c
            adv_examples_t: adversarial examples
        """
        # Fourier curve visualization
        c_curve_name = os.path.join(self.c_curve, f'curve_{basename}.png')
        self.visualize_winding_number_curve(learnable_c, filename=c_curve_name)
        print('curve saved')
        # adv example undetected visualization
        adv_example_t_vis = adv_examples_t.squeeze(0)
        adv_example_t_vis = adv_example_t_vis * 255.0
        applied_img_np = adv_example_t_vis.cpu().detach().numpy()
        applied_img_np = applied_img_np.transpose(1, 2, 0).astype(np.uint8)
        a_img_np = Image.fromarray(applied_img_np)
        adv_example_name = os.path.join(self.advexamples_nodetect, f'advexample_{basename}.png')
        a_img_np.save(adv_example_name)
        print('adv examples nodetect saved')
        # adv example detection visualization
        boxes, confs = detect_batch(self.model, adv_examples_t)
        box = [boxes[0]]
        conf = [confs[0]]
        adv_save_path = os.path.join(self.advexamples_detect, f'advexample_{basename}.png')
        vis_image_boxes(adv_examples_t, box, conf, self.vis_conf_threshold, adv_save_path) 
        print('adv_examples detect saved')
        # save learnable_c in json file
        json_file_path = os.path.join(self.c_path, f'c_{basename}.json')
        num_coeffs = len(learnable_c)
        k = (num_coeffs - 1) // 2
        c_list = learnable_c.clone().detach().cpu().tolist()     
        coeff_dict = {}
        for i, coeff in enumerate(c_list):
            harmonic = i - k
            coeff_dict[str(harmonic)] = [coeff.real, coeff.imag]
        with open(json_file_path, 'w') as f:
            json.dump(coeff_dict, f)   
        os.chmod(json_file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        print(f'Saved Fourier coefficients to {json_file_path}')


if __name__ == "__main__":
    # # Wei Proposed YOLOv3, input specific, infrared, fourier attack
    workspace = './FourierAttack_train/exp0'
    model= load_infrared_model()
    fourier_attack = FourierAttack_InputSpecific_Infrared(model=model, workspace=workspace)
    dataloader = LLVIPDataloader(
        img_dir='./LLVIP_person/instances_imgs',
        label_dir='./LLVIP_person/instances_labels',
        batch_size=1
    )
    start_time = time.time()
    single_image_training_epochs = []
    for batch_idx, batch_data in enumerate(dataloader):
        basename, learnable_c, adv_examples_t, training_epochs = fourier_attack.train(batch_data)
        fourier_attack.save(basename, learnable_c, adv_examples_t)
        single_image_training_epochs.append(training_epochs)
        print(f'current training epoch list on single images: {single_image_training_epochs}')
    end_time = time.time()
    total_time = end_time - start_time
    average_training_epochs = sum(single_image_training_epochs)/len(single_image_training_epochs)
    print(f'advshape training total time on yolov3: {total_time}s')
    print(f'average training epochs on single image is: {average_training_epochs}')