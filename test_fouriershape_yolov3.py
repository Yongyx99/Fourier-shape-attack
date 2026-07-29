"""
Test of Fourier shape attack and its variants
Based on YOLOv3
"""
import numpy as np
import torch
import torch.nn.functional as F
from ipdb import set_trace as st
import os
import re
import cv2
from PIL import Image
from attack_utils.test import compute_ap
from base import BaseTest

# YOLOv3, env = 'yolov3'
from attack_utils.datasets import LLVIPDataloader
from yolov3.detect_custom import load_infrared_model

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(6)

TARGET_MODEL = 'yolov3'
   

class FourierShape_White_InputSpecific_Test(BaseTest):
    """
    test of input specific Fourier shape attack, white inside Fourier shape
    """
    def __init__(self, model, workspace, learnable_c_folder):
        super(FourierShape_White_InputSpecific_Test, self).__init__(model, workspace)
        self.learnable_c_folder = learnable_c_folder
        self.confs = np.arange(0.05, 1.0, 0.05)
    
    def test_asr(self, dataloader):
        """
        calculating ASR under confidences [0.05, 0.95]
        """
        for conf_thresh in self.confs:
            total_targets = 0
            success_attacks = 0
            for images, labels, basenames in dataloader:
                basename_elem = basenames[0].split('_')[:3]
                basename = '_'.join(basename_elem)
                # load c json file and generating G_values_t
                learnable_c_path = os.path.join(self.learnable_c_file, f'c_final_{basename}.json')
                G_values_t = self.generate_mask_from_cjson(learnable_c_path)
                # load images
                image_t = images[0].to(self.device) # RGB
                label = labels[0]       # torch.Size([N, 6]) N:object number
                img_size = (images.shape[-2], images.shape[-1])
                # generated patch boxes
                patch_bboxes = self.generate_patch_boxes_t(label)
                if patch_bboxes.size(0) == 0:
                    continue
                num_patches = patch_bboxes.shape[0]
                # obtain adv_batch
                adv_batch = G_values_t.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(self.device)     # torch.Size([patch_number,3, patch_size, patch_size])
                adv_transformed, msk_transformed = self.transformer.resize_rotate(
                    adv_batch, patch_bboxes, img_size
                )       # adv_transformed: torch.Size[(num_patches, 3, img_h, img_w)], msk_transformed: torch.Size[(num_patches, 1, img_h, img_w)]
                # 对torch.Size([num_patches, 3, img_h, img_w])的adv_transformed进行融合
                adv_img_melt, _ = torch.max(adv_transformed, dim=0)
                # apply white patch
                applied_image = self.applier.forward(image_t, adv_img_melt)
                '''
                # # vis test
                # applied_image_vis = applied_image * 255.0
                # image_np = applied_image_vis.detach().cpu().numpy()
                # image_np = image_np.transpose(1, 2, 0)
                # image_np = image_np.astype(np.uint8)
                # image_save = Image.fromarray(image_np)      # 按照RGB格式保存
                # image_save.save('output_advshape_image.jpg')
                # assert False
                '''
                adv_examples_t = applied_image.unsqueeze(0)
                # calculating attack success number and total attack number
                success_num, total_num = self.calculate_success_attack(images, adv_examples_t, label, conf_thresh)
                success_attacks += success_num
                total_targets += total_num
            asr = success_attacks / total_targets if total_targets > 0 else 0
            print(f'Attack Success Rate of advshape at conf={conf_thresh:.2f} is:{asr:.4f}')

    def test_APdrop(self, dataloader):
        """
        calculating APdrop of general Fourier shape attack
        未验证！
        """
        # calculating AP drop
        orig_preds, orig_targets, adv_preds, adv_targets = [], [], [], []
        idx = 0
        for images, labels, basenames in dataloader:
            basename_elem = basenames[0].split('_')[:3]
            basename = '_'.join(basename_elem)
            # load c json file and generating G_values_t
            learnable_c_path = os.path.join(self.learnable_c_folder, f'c_{basename}.json')
            G_values_t = self.generate_mask_from_cjson(learnable_c_path)
            img_size = (images.shape[-2], images.shape[-1])
            img_height, img_width = img_size
            torch.cuda.empty_cache()
            image_t = images.squeeze(0).to(self.device)       # torch.Size([3, 416, 416])
            label = labels[0]       # torch.Size([N, 6])
            # generated patches
            patch_bboxes = self.generate_patch_boxes_t(label)
            if patch_bboxes.size(0) == 0:
                continue
            num_patches = patch_bboxes.shape[0]
            # generating adv_batch
            adv_batch = G_values_t.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(self.device)     # torch.Size([patch_number,3, patch_size, patch_size])
            # applying adversarial patches
            adv_transformed, msk_transformed = self.transformer.resize_rotate(
                adv_batch, patch_bboxes, img_size
            )       # adv_transformed: torch.Size[(num_patches, 3, img_h, img_w)], msk_transformed: torch.Size[(num_patches, 1, img_h, img_w)]
            # 对torch.Size([num_patches, 3, img_h, img_w])的adv_transformed进行融合
            adv_img_melt, _ = torch.max(adv_transformed, dim=0)
            # infrared, apply black patch
            applied_image = self.applier.forward_infrared(image_t, adv_img_melt)
            
            # # vis test
            applied_image_vis = applied_image * 255.0
            image_np = applied_image_vis.detach().cpu().numpy()
            image_np = image_np.transpose(1, 2, 0)
            image_np = image_np.astype(np.uint8)
            image_save = Image.fromarray(image_np)      # 按照RGB格式保存
            image_save.save(f'output_advshape_{basename}.jpg')
            assert False
            
            adv_examples_t = applied_image.unsqueeze(0)
            # step 1: obtain benign image detection result
            orig_preds_elem, orig_targets_elem = self.obtain_preds_gts(images, label, idx)
            orig_preds.append(orig_preds_elem)
            orig_targets.append(orig_targets_elem)
            # step 2: obtain adversarial image detection result
            adv_preds_elem, adsv_targets_elem = self.obtain_preds_gts(adv_examples_t, label, idx)
            adv_preds.append(adv_preds_elem)
            adv_targets.append(adsv_targets_elem)
            idx += 1
        # step 5: compute benign images AP and adversarial images AP
        orig_ap, orig_recalls, orig_precisions = compute_ap(orig_preds, orig_targets, self.iou_thresh)
        adv_ap, adv_recalls, adv_precisions = compute_ap(adv_preds, adv_targets, self.iou_thresh)
        '''
        # save orig_recalls, orig_precision, adv_recalls, adv_precisions
        save_orig_recalls_path = os.path.join(self.PRs_path, 'orig_recalls.npy')
        save_orig_precisions_path = os.path.join(self.PRs_path, 'orig_precisions.npy')
        save_adv_recalls_path = os.path.join(self.PRs_path, 'adv_recalls.npy')
        save_adv_precisions_path = os.path.join(self.PRs_path, 'adv_precisions.npy')
        save_orig_ap_path = os.path.join(self.PRs_path, 'orig_ap.npy')
        save_adv_ap_path = os.path.join(self.PRs_path, 'adv_ap.npy')
        np.save(save_orig_recalls_path, orig_recalls)
        np.save(save_orig_precisions_path, orig_precisions)
        np.save(save_adv_recalls_path, adv_recalls)
        np.save(save_adv_precisions_path, adv_precisions)
        np.save(save_orig_ap_path, orig_ap)
        np.save(save_adv_ap_path, adv_ap)
        print('finish saving precisions, recalls, and APs.')
        '''
        
        # Calculate AP drop
        ap_drop = orig_ap - adv_ap
        print(f'original AP: {orig_ap:.4f}')
        print(f'adversarial AP: {adv_ap:.4f}')
        print(f'AP drop: {ap_drop:.4f}')

class FourierShape_Infrared_InputSpecific_Test(BaseTest):
    """
    test of input specific Fourier shape attack, infrared, black inside Fourier shape
    """
    def __init__(self, model, workspace, learnable_c_folder):
        super(FourierShape_Infrared_InputSpecific_Test, self).__init__(model, workspace)
        self.learnable_c_folder = learnable_c_folder
        self.confs = np.arange(0.05, 1.0, 0.05)
        self.PRs_path = os.path.join(self.workspace, 'PR_curve')
        for path in [self.PRs_path]:
            try:
                # create directory (if already exists, ignore)
                os.makedirs(path, mode=0o777, exist_ok=True)
                # confirm permission (some system need to set separately)
                os.chmod(path, 0o777)
                print(f"Successfully create new file: {path}")
            except Exception as e:
                print(f"Warning: create {path} fail: {str(e)}")
    
    def get_exact_unique_file(self, target_list):
        directory_path = self.learnable_c_folder
        all_files = os.listdir(directory_path)
        matched_files = []
        for filename in all_files:
            is_match = True
            for item in target_list:
                pattern = rf"(^|[^0-9]){re.escape(item)}([^0-9]|$)"
                if not re.search(pattern, filename):
                    is_match = False
                    break
            if is_match:
                matched_files.append(filename)
        if len(matched_files) == 1:
            return matched_files[0]
        else:
            print(f"匹配到的列表: {matched_files}")
            return None
    
    def test_asr(self, dataloader):
        """
        calculating ASR under confidences [0.05, 0.95]
        """
        for conf_thresh in self.confs:
            total_targets = 0
            success_attacks = 0
            for images, labels, basenames in dataloader:
                basename_elem = basenames[0].split('_')[:3]
                if TARGET_MODEL == 'YOLOv8':
                    basename_tmp = '_'.join(basename_elem)
                    basename = f'c_{basename_tmp}.json'
                elif TARGET_MODEL == 'retinanet' or TARGET_MODEL == 'fasterrcnn' or TARGET_MODEL == 'yolov3':
                    basename = self.get_exact_unique_file(basename_elem)
                # load c json file and generating G_values_t
                learnable_c_path = os.path.join(self.learnable_c_folder, f'{basename}')
                G_values_t = self.generate_mask_from_cjson(learnable_c_path)
                # load images
                image_t = images[0].to(self.device) # RGB
                label = labels[0]       # torch.Size([N, 6]) N:object number
                img_size = (images.shape[-2], images.shape[-1])
                # generated patch boxes
                patch_bboxes = self.generate_patch_boxes_t(label)
                if patch_bboxes.size(0) == 0:
                    continue
                num_patches = patch_bboxes.shape[0]
                # obtain adv_batch
                adv_batch = G_values_t.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(self.device)     # torch.Size([patch_number,3, patch_size, patch_size])
                adv_transformed, msk_transformed = self.transformer.resize_rotate(
                    adv_batch, patch_bboxes, img_size
                )       # adv_transformed: torch.Size[(num_patches, 3, img_h, img_w)], msk_transformed: torch.Size[(num_patches, 1, img_h, img_w)]
                # 对torch.Size([num_patches, 3, img_h, img_w])的adv_transformed进行融合
                adv_img_melt, _ = torch.max(adv_transformed, dim=0)
                # apply white patch
                applied_image = self.applier.forward_infrared(image_t, adv_img_melt)
                '''
                # vis test
                applied_image_vis = applied_image * 255.0
                image_np = applied_image_vis.detach().cpu().numpy()
                image_np = image_np.transpose(1, 2, 0)
                image_np = image_np.astype(np.uint8)
                image_save = Image.fromarray(image_np)      # 按照RGB格式保存
                image_save.save('output_advshape_image.jpg')
                assert False
                '''
                adv_examples_t = applied_image.unsqueeze(0)
                # calculating attack success number and total attack number
                success_num, total_num = self.calculate_success_attack(images, adv_examples_t, label, conf_thresh)
                success_attacks += success_num
                total_targets += total_num
            asr = success_attacks / total_targets if total_targets > 0 else 0
            print(f'Attack Success Rate of advshape at conf={conf_thresh:.2f} is:{asr:.4f}')

    def test_APdrop(self, dataloader):
        """
        calculating APdrop of input specific Fourier shape attack
        """
        # calculating AP drop
        orig_preds, orig_targets, adv_preds, adv_targets = [], [], [], []
        idx = 0
        for images, labels, basenames in dataloader:
            basename_elem = basenames[0].split('_')[:3]
            if TARGET_MODEL == 'YOLOv8':
                basename_tmp = '_'.join(basename_elem)
                basename = f'c_{basename_tmp}.json'
            elif TARGET_MODEL == 'retinanet' or TARGET_MODEL == 'fasterrcnn' or TARGET_MODEL == 'yolov3':
                basename = self.get_exact_unique_file(basename_elem)
            # load c json file and generating G_values_t
            learnable_c_path = os.path.join(self.learnable_c_folder, f'{basename}')
            G_values_t = self.generate_mask_from_cjson(learnable_c_path)
            img_size = (images.shape[-2], images.shape[-1])
            img_height, img_width = img_size
            torch.cuda.empty_cache()
            image_t = images.squeeze(0).to(self.device)       # torch.Size([3, 416, 416])
            label = labels[0]       # torch.Size([N, 6])
            # generated patches
            patch_bboxes = self.generate_patch_boxes_t(label)
            if patch_bboxes.size(0) == 0:
                continue
            num_patches = patch_bboxes.shape[0]
            # generating adv_batch
            adv_batch = G_values_t.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(self.device)     # torch.Size([patch_number,3, patch_size, patch_size])
            # applying adversarial patches
            adv_transformed, msk_transformed = self.transformer.resize_rotate(
                adv_batch, patch_bboxes, img_size
            )       # adv_transformed: torch.Size[(num_patches, 3, img_h, img_w)], msk_transformed: torch.Size[(num_patches, 1, img_h, img_w)]
            # 对torch.Size([num_patches, 3, img_h, img_w])的adv_transformed进行融合
            adv_img_melt, _ = torch.max(adv_transformed, dim=0)
            # infrared, apply black patch
            applied_image = self.applier.forward_infrared(image_t, adv_img_melt)
            '''
            # # vis test
            applied_image_vis = applied_image * 255.0
            image_np = applied_image_vis.detach().cpu().numpy()
            image_np = image_np.transpose(1, 2, 0)
            image_np = image_np.astype(np.uint8)
            image_save = Image.fromarray(image_np)      # 按照RGB格式保存
            image_save.save(f'output_advshape_{basename}.jpg')
            assert False
            '''
            adv_examples_t = applied_image.unsqueeze(0)
            # step 1: obtain benign image detection result
            orig_preds_elem, orig_targets_elem = self.obtain_preds_gts(images, label, idx)
            orig_preds.append(orig_preds_elem)
            orig_targets.append(orig_targets_elem)
            # step 2: obtain adversarial image detection result
            adv_preds_elem, adsv_targets_elem = self.obtain_preds_gts(adv_examples_t, label, idx)
            adv_preds.append(adv_preds_elem)
            adv_targets.append(adsv_targets_elem)
            idx += 1
        # step 5: compute benign images AP and adversarial images AP
        orig_ap, orig_recalls, orig_precisions = compute_ap(orig_preds, orig_targets, self.iou_thresh)
        adv_ap, adv_recalls, adv_precisions = compute_ap(adv_preds, adv_targets, self.iou_thresh)
        
        # save orig_recalls, orig_precision, adv_recalls, adv_precisions
        # save_orig_recalls_path = os.path.join(self.PRs_path, 'orig_recalls.npy')
        # save_orig_precisions_path = os.path.join(self.PRs_path, 'orig_precisions.npy')
        # save_adv_recalls_path = os.path.join(self.PRs_path, 'adv_recalls.npy')
        # save_adv_precisions_path = os.path.join(self.PRs_path, 'adv_precisions.npy')
        # save_orig_ap_path = os.path.join(self.PRs_path, 'orig_ap.npy')
        # save_adv_ap_path = os.path.join(self.PRs_path, 'adv_ap.npy')
        # np.save(save_orig_recalls_path, orig_recalls)
        # np.save(save_orig_precisions_path, orig_precisions)
        # np.save(save_adv_recalls_path, adv_recalls)
        # np.save(save_adv_precisions_path, adv_precisions)
        # np.save(save_orig_ap_path, orig_ap)
        # np.save(save_adv_ap_path, adv_ap)
        # print('finish saving precisions, recalls, and APs.')
        
        # Calculate AP drop
        ap_drop = orig_ap - adv_ap
        print(f'original AP: {orig_ap:.4f}')
        print(f'adversarial AP: {adv_ap:.4f}')
        print(f'AP drop: {ap_drop:.4f}')


class FourierShape_Color_InputSpecific_Test(BaseTest):
    """
    test of input specific Fourier shape attack, white inside Fourier shape
    """    
    def __init__(self, model, workspace, learnable_c_folder, learnable_texture_folder):
        super(FourierShape_Color_InputSpecific_Test, self).__init__(model, workspace)    # learnable_c_file shoule be a json folder
        self.learnable_c_folder = learnable_c_folder
        self.learnable_texture_folder = learnable_texture_folder
        self.confs = np.arange(0.05, 1.0, 0.05)
    
    def test_asr(self, dataloader):
        """
        calculating ASR under confidences [0.05, 0.95]
        """
        for conf_thresh in self.confs:
            total_targets = 0
            success_attacks = 0
            for images, labels, basenames in dataloader:
                basename = basenames[0]
                basename_elem = basenames[0].split('_')[:3]
                basename = '_'.join(basename_elem)
                # load c json file and generating G_values_t
                learnable_c_path = os.path.join(self.learnable_c_folder, f'learnable_c_{basename}.json')
                learnable_texture_path = os.path.join(self.learnable_texture_folder, f'learnable_texture_{basename}.pth')
                G_values_t = self.generate_mask_from_cjson(learnable_c_path)
                # load images
                image_t = images[0].to(self.device) # RGB
                label = labels[0]       # torch.Size([N, 6]) N:object number
                img_size = (images.shape[-2], images.shape[-1])
                # generated patch boxes
                patch_bboxes = self.generate_patch_boxes_t(label)
                if patch_bboxes.size(0) == 0:
                    continue
                num_patches = patch_bboxes.shape[0]
                # generate learnable_texture
                learnable_texture = torch.load(learnable_texture_path)
                learnable_texture_fix = learnable_texture.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(self.device)
                # obtain adv_batch
                G_values_t_fix = G_values_t.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(self.device)     # torch.Size([patch_number,3, patch_size, patch_size])
                adv_batch = G_values_t_fix * learnable_texture_fix
                mask_transformed, _ = self.transformer.resize_rotate(
                    G_values_t_fix, patch_bboxes, img_size
                )
                mask_melt, _ = torch.max(mask_transformed, dim=0)
                adv_transformed, _ = self.transformer.resize_rotate(
                    adv_batch, patch_bboxes, img_size
                )       # adv_transformed: torch.Size[(num_patches, 3, img_h, img_w)], msk_transformed: torch.Size[(num_patches, 1, img_h, img_w)]
                adv_img_melt, _ = torch.max(adv_transformed, dim=0) # torch.Size([num_patches, 3, img_h, img_w]) melting
                # apply patch 
                applied_image = self.applier.forward_color(image_t, adv_img_melt, mask_melt)
                '''
                # # vis test
                # applied_image_vis = applied_image * 255.0
                # image_np = applied_image_vis.detach().cpu().numpy()
                # image_np = image_np.transpose(1, 2, 0)
                # image_np = image_np.astype(np.uint8)
                # image_save = Image.fromarray(image_np)      # 按照RGB格式保存
                # image_save.save('output_advshape_image.jpg')
                # assert False
                '''
                adv_examples_t = applied_image.unsqueeze(0)
                # calculating attack success number and total attack number
                success_num, total_num = self.calculate_success_attack(images, adv_examples_t, label, conf_thresh)
                success_attacks += success_num
                total_targets += total_num
            asr = success_attacks / total_targets if total_targets > 0 else 0
            print(f'Attack Success Rate of advshape at conf={conf_thresh:.2f} is:{asr:.4f}')

    def test_APdrop(self, dataloader):
        """
        calculating AP drop of color input-specific Fourier shape attack
        """
        pass

class FourierShape_White_General_Test(BaseTest):
    def __init__(self, model, workspace, learnable_c_file):
        super(FourierShape_White_General_Test, self).__init__(model, workspace)
        self.learnable_c_file = learnable_c_file    # learnable_c_file shoule be a json file, not folder
        self.confs = np.arange(0.05, 1.0, 0.05)
    
    def test_asr(self, dataloader):
        """
        calculating ASR under confidences [0.05, 0.95]
        """
        for conf_thresh in self.confs:
            total_targets = 0
            success_attacks = 0
            for images, labels, basenames in dataloader:
                basename_elem = basenames[0].split('_')[:3]
                basename = '_'.join(basename_elem)
                # load c json file and generating G_values_t
                G_values_t = self.generate_mask_from_cjson(self.learnable_c_file)
                image_t = images[0].to(self.device)
                label = labels[0]       # torch.Size([N, 6]) N:object number
                img_size = (images.shape[-2], images.shape[-1])
                # generated patches
                patch_bboxes = self.generate_patch_boxes_t(label)
                if patch_bboxes.size(0) == 0:
                    continue
                num_patches = patch_bboxes.shape[0]
                # obtain adv_batch
                adv_batch = G_values_t.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(self.device)     # torch.Size([patch_number,3, patch_size, patch_size])
                adv_transformed, msk_transformed = self.transformer.resize_rotate(
                    adv_batch, patch_bboxes, img_size
                )       # adv_transformed: torch.Size[(num_patches, 3, img_h, img_w)], msk_transformed: torch.Size[(num_patches, 1, img_h, img_w)]
                # 对torch.Size([num_patches, 3, img_h, img_w])的adv_transformed进行融合
                adv_img_melt, _ = torch.max(adv_transformed, dim=0)
                # apply black patch
                applied_image = self.applier.forward_infrared(image_t, adv_img_melt)
                adv_examples_t = applied_image.unsqueeze(0)
                # calculating attack success number and total attack number
                success_num, total_num = self.calculate_success_attack(images, adv_examples_t, label, conf_thresh)
                success_attacks += success_num
                total_targets += total_num
            asr = success_attacks / total_targets if total_targets > 0 else 0
            print(f'Attack Success Rate of advshape at conf={conf_thresh:.2f} is:{asr:.4f}')
    
    def test_APdrop(self, dataloader):
        """
        calculating APdrop of general Fourier shape attack
        """
        # Generating G from c json file
        G_values_t = self.generate_mask_from_cjson(self.learnable_c_file)
        # calculating AP drop
        orig_preds, orig_targets, adv_preds, adv_targets = [], [], [], []
        idx = 0
        for images, labels, basenames in dataloader:
            basename = basenames[0]
            img_size = (images.shape[-2], images.shape[-1])
            img_height, img_width = img_size
            torch.cuda.empty_cache()
            image_t = images.squeeze(0).to(self.device)       # torch.Size([3, 416, 416])
            label = labels[0]       # torch.Size([N, 6])
            # generated patches
            patch_bboxes = self.generate_patch_boxes_t(label)
            if patch_bboxes.size(0) == 0:
                continue
            num_patches = patch_bboxes.shape[0]
            # generating adv_batch
            adv_batch = G_values_t.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(self.device)     # torch.Size([patch_number,3, patch_size, patch_size])
            # applying adversarial patches
            adv_transformed, msk_transformed = self.transformer.resize_rotate(
                adv_batch, patch_bboxes, img_size
            )       # adv_transformed: torch.Size[(num_patches, 3, img_h, img_w)], msk_transformed: torch.Size[(num_patches, 1, img_h, img_w)]
            # 对torch.Size([num_patches, 3, img_h, img_w])的adv_transformed进行融合
            adv_img_melt, _ = torch.max(adv_transformed, dim=0)
            # infrared, apply black patch
            applied_image = self.applier.forward_infrared(image_t, adv_img_melt)
            adv_examples_t = applied_image.unsqueeze(0)
            # step 1: obtain benign image detection result
            orig_preds_elem, orig_targets_elem = self.obtain_preds_gts(images, label, idx)
            orig_preds.append(orig_preds_elem)
            orig_targets.append(orig_targets_elem)
            # step 2: obtain adversarial image detection result
            adv_preds_elem, adsv_targets_elem = self.obtain_preds_gts(adv_examples_t, label, idx)
            adv_preds.append(adv_preds_elem)
            adv_targets.append(adsv_targets_elem)
            idx += 1
        # step 5: compute benign images AP and adversarial images AP
        orig_ap, orig_recalls, orig_precisions = compute_ap(orig_preds, orig_targets, self.iou_thresh)
        adv_ap, adv_recalls, adv_precisions = compute_ap(adv_preds, adv_targets, self.iou_thresh)
        '''
        # save orig_recalls, orig_precision, adv_recalls, adv_precisions
        save_orig_recalls_path = os.path.join(self.PRs_path, 'orig_recalls.npy')
        save_orig_precisions_path = os.path.join(self.PRs_path, 'orig_precisions.npy')
        save_adv_recalls_path = os.path.join(self.PRs_path, 'adv_recalls.npy')
        save_adv_precisions_path = os.path.join(self.PRs_path, 'adv_precisions.npy')
        save_orig_ap_path = os.path.join(self.PRs_path, 'orig_ap.npy')
        save_adv_ap_path = os.path.join(self.PRs_path, 'adv_ap.npy')
        np.save(save_orig_recalls_path, orig_recalls)
        np.save(save_orig_precisions_path, orig_precisions)
        np.save(save_adv_recalls_path, adv_recalls)
        np.save(save_adv_precisions_path, adv_precisions)
        np.save(save_orig_ap_path, orig_ap)
        np.save(save_adv_ap_path, adv_ap)
        print('finish saving precisions, recalls, and APs.')
        '''
        # Calculate AP drop
        ap_drop = orig_ap - adv_ap
        print(f'original AP: {orig_ap:.4f}')
        print(f'adversarial AP: {adv_ap:.4f}')
        print(f'AP drop: {ap_drop:.4f}')
    
    def visualization(self, dataloader):
        """
        visualize general fourier shape on images in dataloader.
        Args:
            dataloader: dataloader for images
        """
        # Generating G from c json file
        G_values_t = self.generate_mask_from_cjson(self.learnable_c_file)
        for batch_id, (images, labels, basenames) in enumerate(dataloader):
            basename = basenames[0]
            basename_idx = basename.split('.')[0]
            # load image_t
            image_t = images[0].to(self.device)
            label = labels[0]       # torch.Size([N, 6]) 其中N为物体的数量
            # 设置img_size
            img_size = (images.shape[-2], images.shape[-1])
            # generate patches from label
            patch_bboxes = self.generate_patch_boxes_t(label)
            if patch_bboxes.size(0) == 0:
                continue
            # use image_t and patch_bboxes from one image to generate adversarial example
            num_patches = patch_bboxes.shape[0]
            # obtain adv_batch
            adv_batch = G_values_t.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(self.device)     # torch.Size([patch_number,3, patch_size, patch_size])
            # apply adversarial patch
            adv_transformed, msk_transformed = self.transformer.resize_rotate(
                adv_batch, patch_bboxes, img_size
            )       # adv_transformed: torch.Size[(num_patches, 3, img_h, img_w)], msk_transformed: torch.Size[(num_patches, 1, img_h, img_w)]
            adv_img_melt, _ = torch.max(adv_transformed, dim=0)
            # visible，apply white patch
            applied_image = self.applier(image_t, adv_img_melt)
            # visualize applied image, range from 0~255，w/o mean，std normalization，bgr
            applied_image_vis = applied_image * 255.0
            image_np = applied_image_vis.detach().cpu().numpy()
            image_np = image_np.transpose(1, 2, 0).astype(np.uint8)
            image_save = Image.fromarray(image_np)      # 按照RGB格式保存'
            img_path = os.path.join(self.clean_advexample_path, f'{basename_idx}.jpg')
            image_save.save(img_path)
        print('General Fourier Patch visualization finished!')

class FourierShape_Color_General_Test(BaseTest):
    """
    test of general Fourier shape attack with color
    """
    def __init__(self, model, workspace, learnable_c_file, learnable_texture_file):
        super(FourierShape_Color_General_Test, self).__init__(model, workspace)    # learnable_c_file shoule be a json file, not folder
        self.learnable_c_file = learnable_c_file
        self.learnable_texture_file = learnable_texture_file
        self.confs = np.arange(0.05, 1.0, 0.05)
    
    def test_asr(self, dataloader):
        """
        calculating ASR under confidences [0.05, 0.95]
        """
        for conf_thresh in self.confs:
            total_targets = 0
            success_attacks = 0
            for images, labels, basenames in dataloader:
                basename = basenames[0]
                # load c json file and generating G_values_t
                G_values_t = self.generate_mask_from_cjson(self.learnable_c_file)
                # load images
                image_t = images[0].to(self.device) # RGB
                label = labels[0]       # torch.Size([N, 6]) N:object number
                img_size = (images.shape[-2], images.shape[-1])
                # generated patch boxes
                patch_bboxes = self.generate_patch_boxes_t(label)
                if patch_bboxes.size(0) == 0:
                    continue
                num_patches = patch_bboxes.shape[0]
                # load learnable_texture
                learnable_texture = torch.load(self.learnable_texture_file)
                learnable_texture_fix = learnable_texture.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(self.device)
                # obtain adv_batch
                G_values_t_fix = G_values_t.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(self.device)     # torch.Size([patch_number,3, patch_size, patch_size])
                adv_batch = G_values_t_fix * learnable_texture_fix
                mask_transformed, _ = self.transformer.resize_rotate(
                    G_values_t_fix, patch_bboxes, img_size
                )
                mask_melt, _ = torch.max(mask_transformed, dim=0)
                adv_transformed, _ = self.transformer.resize_rotate(
                    adv_batch, patch_bboxes, img_size
                )       # adv_transformed: torch.Size[(num_patches, 3, img_h, img_w)], msk_transformed: torch.Size[(num_patches, 1, img_h, img_w)]
                adv_img_melt, _ = torch.max(adv_transformed, dim=0) # torch.Size([num_patches, 3, img_h, img_w]) melting
                # apply patch 
                applied_image = self.applier.forward_color(image_t, adv_img_melt, mask_melt)
                '''
                # # vis test
                # applied_image_vis = applied_image * 255.0
                # image_np = applied_image_vis.detach().cpu().numpy()
                # image_np = image_np.transpose(1, 2, 0)
                # image_np = image_np.astype(np.uint8)
                # image_save = Image.fromarray(image_np)      # 按照RGB格式保存
                # image_save.save('output_advshape_image.jpg')
                # assert False
                '''
                adv_examples_t = applied_image.unsqueeze(0)
                # calculating attack success number and total attack number
                success_num, total_num = self.calculate_success_attack(images, adv_examples_t, label, conf_thresh)
                success_attacks += success_num
                total_targets += total_num
            asr = success_attacks / total_targets if total_targets > 0 else 0
            print(f'Attack Success Rate of advshape at conf={conf_thresh:.2f} is:{asr:.4f}')
    
    def visualization(self, dataloader):
        """
        visualize general fourier shape on images in dataloader, with color inside.
        Args:
            dataloader: dataloader for images
        """
        # Generating G from c json file
        G_values_t = self.generate_mask_from_cjson(self.learnable_c_file)
        for batch_id, (images, labels, basenames) in enumerate(dataloader):
            basename = basenames[0]
            basename_idx = basename.split('.')[0]
            # load image_t
            image_t = images[0].to(self.device)
            label = labels[0]       # torch.Size([N, 6]) 其中N为物体的数量
            # 设置img_size
            img_size = (images.shape[-2], images.shape[-1])
            # 通过label生成patches
            patch_bboxes = self.generate_patch_boxes_t(label)
            if patch_bboxes.size(0) == 0:
                continue
            # use image_t and patch_bboxes from one image to generate adversarial example
            num_patches = patch_bboxes.shape[0]
            # load learnable_texture
            learnable_texture = torch.load(self.learnable_texture_file)
            learnable_texture_fix = learnable_texture.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(self.device)
            # obtain adv_batch
            G_values_t_fix = G_values_t.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(self.device)     # torch.Size([patch_number,3, patch_size, patch_size])
            adv_batch = G_values_t_fix * learnable_texture_fix
            mask_transformed, _ = self.transformer.resize_rotate(
                G_values_t_fix, patch_bboxes, img_size
            )
            mask_melt, _ = torch.max(mask_transformed, dim=0)
            adv_transformed, _ = self.transformer.resize_rotate(
                adv_batch, patch_bboxes, img_size
            )       # adv_transformed: torch.Size[(num_patches, 3, img_h, img_w)], msk_transformed: torch.Size[(num_patches, 1, img_h, img_w)]
            adv_img_melt, _ = torch.max(adv_transformed, dim=0) # torch.Size([num_patches, 3, img_h, img_w]) melting
            # apply patch 
            applied_image = self.applier.forward_color(image_t, adv_img_melt, mask_melt)
            # visualize applied image, 取值范围为0~255，没有经过mean，std归一化，bgr格式
            applied_image_vis = applied_image * 255.0
            image_np = applied_image_vis.detach().cpu().numpy()
            image_np = image_np.transpose(1, 2, 0).astype(np.uint8)
            image_save = Image.fromarray(image_np)      # 按照RGB格式保存'
            img_path = os.path.join(self.clean_advexample_path, f'{basename_idx}.jpg')
            image_save.save(img_path)
        print('General Fourier Patch visualization finished!')
    

class BaseTestArea(BaseTest):
    """
    base class for testing infrared adversarial patch area.
    """
    def __init__(self, model, workspace):
        super(BaseTestArea, self).__init__(model, workspace)

    def test_fouriershape_area(self, dataloader, learnable_c_folder, binarization_thresh=0.5):
        """
        test area of fourier shape.
        Args:
            dataloader: dataloader for images
            learnable_c_folder: folder for learnable_c
        Returns:
            mean_area: mean area (percentage, patch_area / bounding_box_area) of fourier shape
        """
        # calculating mean area of fourier shape
        total_image_number = len(dataloader)
        total_area_ratio = 0
        for images, labels, basenames in dataloader:
            # origin version
            basename = basenames[0]
            # fix version
            # basename_elem = basenames[0].split('_')[:3]
            # basename = '_'.join(basename_elem)
            # load c json file and generating G_values_t
            learnable_c_path = os.path.join(learnable_c_folder, f'c_final_{basename}.json')
            G_values_t = self.generate_mask_from_cjson(learnable_c_path)
            img_size = (images.shape[-2], images.shape[-1])
            img_height, img_width = img_size
            torch.cuda.empty_cache()
            image_t = images.squeeze(0).to(self.device)       # torch.Size([3, 416, 416])
            label = labels[0]       # torch.Size([N, 6])
            # obtain object box size
            box_x1, box_y1, box_x2, box_y2 = label[0][1], label[0][2], label[0][3], label[0][4]
            box_width = int(box_x2 - box_x1)
            box_height = int(box_y2 - box_y1)
            box_area = box_width * box_height
            # generated patches
            patch_bboxes = self.generate_patch_boxes_t(label)
            if patch_bboxes.size(0) == 0:
                continue
            num_patches = patch_bboxes.shape[0]
            # generating adv_batch
            adv_batch = G_values_t.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(self.device)     # torch.Size([patch_number,3, patch_size, patch_size])
            # applying adversarial patches
            adv_transformed, msk_transformed = self.transformer.resize_rotate(
                adv_batch, patch_bboxes, img_size
            )       # adv_transformed: torch.Size[(num_patches, 3, img_h, img_w)], msk_transformed: torch.Size[(num_patches, 1, img_h, img_w)]
            # 对torch.Size([num_patches, 3, img_h, img_w])的adv_transformed进行融合
            adv_img_melt, _ = torch.max(adv_transformed, dim=0)     # torch.Size([3, img_h, img_w])
            # mask process （binarization）
            adv_mask_melt = torch.mean(adv_img_melt, dim=0, keepdim=True)
            binary_adv_mask = (adv_mask_melt > binarization_thresh).int()   # torch.Size([1, img_h, img_w])
            '''
            # visualization
            binary_adv_mask = binary_adv_mask.repeat(3, 1, 1)
            adv_mask_vis = binary_adv_mask * 255.0
            mask_np = adv_mask_vis.detach().cpu().numpy()
            mask_np = mask_np.transpose(1, 2, 0).astype(np.uint8)
            mask_save = Image.fromarray(mask_np)      # 按照RGB格式保存
            mask_save.save(f'output_advshape_mask_{basename}.jpg')
            assert False
            '''
            # area calculation
            adv_mask_area = binary_adv_mask.sum().item()       # 1 channels
            area_ratio = adv_mask_area / box_area
            total_area_ratio += area_ratio
        
        # calculate final result
        mean_area = total_area_ratio / total_image_number
        print(f'mean_area: {mean_area}')
    
    def test_uap_area(self, dataloader, mask_folder, binarization_thresh=0.5):
        """
        test area of uap attack.
        mask are (.png) files under same size of benign instance images, so directly calculate area on mask files to stand for real mask area of each instance.
        Args:
            dataloader: dataloader for images
            mask_folder: folder for masks
        Returns:
            mean_area: mean area (percentage, patch_area / bounding_box_area) of uap patch
        """
        def find_matching_files(target_string, folder_path):
            """
            given target string with format 'XXX_XXX_XXX', find Amatching file in a folder with name 'XXX_XXX_XXX_aaa'
            Args:
                target_string: target string
                folder_path: folder path
            Returns:
                matching_files: list of matching files
            """
            search_prefix = target_string + '_'
            try:
                all_files = os.listdir(folder_path)
            except FileNotFoundError:
                print(f"Error: Failed to list files in {folder_path}")
                return []
            # search files
            matching_files = []
            for filename in all_files:
                if filename.startswith(search_prefix):
                    suffix_part = filename[len(search_prefix):]
                    if suffix_part and not suffix_part.startswith('_'):
                        matching_files.append(filename)
            return matching_files
                
        # calculating mean area of fourier shape
        total_image_number = len(dataloader)
        total_area_ratio = 0
        for images, labels, basenames in dataloader:
            # origin version
            basename_elem = basenames[0].split('_')[:3]
            basename = '_'.join(basename_elem)
            # match name of basename and mask name
            mask_file_list = find_matching_files(basename, mask_folder)
            if len(mask_file_list) != 1:
                print(f'Error: mask file number is not equal to 1, {mask_file_list}')
                assert False, f'mask file number is not equal to 1, {mask_file_list}'
            mask_file_path = os.path.join(mask_folder, mask_file_list[0])
            mask_img = cv2.imread(mask_file_path)
            mask_img_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
            mask_img_normalized = mask_img_rgb.astype(np.float32) / 255.0
            mask_img_chw = np.transpose(mask_img_normalized, (2, 0, 1))
            mask_t = torch.from_numpy(mask_img_chw)
            mask_t = 1 - mask_t     # transform to binary mask that 1 inside patch and 0 outside patch, torch.Size([3, img_h, img_w])

            img_size = (images.shape[-2], images.shape[-1])
            img_height, img_width = img_size
            torch.cuda.empty_cache()
            image_t = images.squeeze(0).to(self.device)       # torch.Size([3, 416, 416])
            label = labels[0]       # torch.Size([N, 6])
            # obtain object box size
            box_x1, box_y1, box_x2, box_y2 = label[0][1], label[0][2], label[0][3], label[0][4]
            box_width = int(box_x2 - box_x1)
            box_height = int(box_y2 - box_y1)
            box_area = box_width * box_height
            # mask process （binarization）
            adv_mask_melt = torch.mean(mask_t, dim=0, keepdim=True)
            binary_adv_mask = adv_mask_melt.int()   # torch.Size([1, img_h, img_w])
            '''
            # visualization
            binary_adv_mask = binary_adv_mask.repeat(3, 1, 1)
            adv_mask_vis = binary_adv_mask * 255.0
            mask_np = adv_mask_vis.detach().cpu().numpy()
            mask_np = mask_np.transpose(1, 2, 0).astype(np.uint8)
            mask_save = Image.fromarray(mask_np)      # 按照RGB格式保存
            mask_save.save(f'output_advshape_mask_{basename}.jpg')
            assert False
            '''
            # area calculation
            adv_mask_area = binary_adv_mask.sum().item()       # 1 channels
            area_ratio = adv_mask_area / box_area
            total_area_ratio += area_ratio
        # calculate final result
        mean_area = total_area_ratio / total_image_number
        print(f'mean_area: {mean_area}')
    
    def test_infp_area(self, dataloader, mask_folder, binarization_thresh=0.5):
        """
        test area of Infrared.P attack.
        mask are (.jpg) files under same size of benign instance images, so directly calculate area on mask files to stand for real mask area of each instance.
        Args:
            dataloader: dataloader for images
            mask_folder: folder for masks
        Returns:
            mean_area: mean area (percentage, patch_area / bounding_box_area) of infp mask
        """
        def find_matching_files(target_string, folder_path):
            """
            given target string with format 'XXX_XXX_XXX', find Amatching file in a folder with name 'XXX_XXX_XXX_aaa'
            Args:
                target_string: target string
                folder_path: folder path
            Returns:
                matching_files: list of matching files
            """
            search_prefix = target_string + '_'
            try:
                all_files = os.listdir(folder_path)
            except FileNotFoundError:
                print(f"Error: Failed to list files in {folder_path}")
                return []
            # search files
            matching_files = []
            for filename in all_files:
                if filename.startswith(search_prefix):
                    suffix_part = filename[len(search_prefix):]
                    if suffix_part and not suffix_part.startswith('_'):
                        matching_files.append(filename)
            return matching_files
                
        # calculating mean area of fourier shape
        total_image_number = len(dataloader)
        total_area_ratio = 0
        for images, labels, basenames in dataloader:
            # origin version
            basename_elem = basenames[0].split('_')[:3]
            basename = '_'.join(basename_elem)
            # match name of basename and mask name
            mask_file_list = find_matching_files(basename, mask_folder)
            if len(mask_file_list) != 1:
                print(f'Error: mask file number is not equal to 1, {mask_file_list}')
                assert False, f'mask file number is not equal to 1, {mask_file_list}'
            mask_file_path = os.path.join(mask_folder, mask_file_list[0])
            mask_img_rgb = Image.open(mask_file_path).convert('RGB')
            mask_img_normalized = np.array(mask_img_rgb) / 255.0
            mask_img_chw = np.transpose(mask_img_normalized, (2, 0, 1))
            mask_t = torch.from_numpy(mask_img_chw)  # torch.Size([3, img_h, img_w])

            img_size = (images.shape[-2], images.shape[-1])
            img_height, img_width = img_size
            torch.cuda.empty_cache()
            image_t = images.squeeze(0).to(self.device)       # torch.Size([3, 416, 416])
            label = labels[0]       # torch.Size([N, 6])
            # obtain object box size
            box_x1, box_y1, box_x2, box_y2 = label[0][1], label[0][2], label[0][3], label[0][4]
            box_width = int(box_x2 - box_x1)
            box_height = int(box_y2 - box_y1)
            box_area = box_width * box_height
            # mask process （binarization）
            adv_mask_melt = torch.mean(mask_t, dim=0, keepdim=True)
            binary_adv_mask = (adv_mask_melt > binarization_thresh).int()
            binary_adv_mask = binary_adv_mask.int()   # torch.Size([1, img_h, img_w])
            '''
            # visualization
            binary_adv_mask = binary_adv_mask.repeat(3, 1, 1)
            adv_mask_vis = binary_adv_mask * 255.0
            mask_np = adv_mask_vis.detach().cpu().numpy()
            mask_np = mask_np.transpose(1, 2, 0).astype(np.uint8)
            mask_save = Image.fromarray(mask_np)      # 按照RGB格式保存
            mask_save.save(f'output_advshape_mask_{basename}.jpg')
            assert False
            '''
            # area calculation
            adv_mask_area = binary_adv_mask.sum().item()       # 1 channels
            area_ratio = adv_mask_area / box_area
            total_area_ratio += area_ratio
        # calculate final result
        mean_area = total_area_ratio / total_image_number
        print(f'mean_area: {mean_area}')
    
    def test_defp_area(self, dataloader, mask_folder, binarization_thresh=0.5):
        """
        test area of Infrared.P attack.
        mask are (.pth) files under same size of benign instance images, so directly calculate area on mask files to stand for real mask area of each instance.
        Args:
            dataloader: dataloader for images
            mask_folder: folder for masks, masks are in the form of tensor (.pth)
        Returns:
            mean_area: mean area (percentage, patch_area / bounding_box_area) of infp mask
        """
        patch_scale = 0.6
        def find_matching_files(target_string, folder_path):
            """
            given target string with format 'XXX_XXX_XXX', find Amatching file in a folder with name 'mask_t_final_XXX_XXX_XXX_aaa'
            Args:
                target_string: target string
                folder_path: folder path
            Returns:
                matching_files: list of matching files
            """
            search_prefix = f'mask_t_final_{target_string}_'
            try:
                all_files = os.listdir(folder_path)
            except FileNotFoundError:
                print(f"Error: Failed to list files in {folder_path}")
                return []
            # search files
            matching_files = []
            for filename in all_files:
                if filename.startswith(search_prefix):
                    suffix_part = filename[len(search_prefix):]
                    if suffix_part and not suffix_part.startswith('_'):
                        matching_files.append(filename)
            return matching_files
                
        # calculating mean area of fourier shape
        total_image_number = len(dataloader)
        total_area_ratio = 0
        for images, labels, basenames in dataloader:
            # origin version
            basename_elem = basenames[0].split('_')[:3]
            basename = '_'.join(basename_elem)
            # match name of basename and mask name
            mask_file_list = find_matching_files(basename, mask_folder)
            if len(mask_file_list) != 1:
                print(f'Error: mask file number is not equal to 1, {mask_file_list}')
                assert False, f'mask file number is not equal to 1, {mask_file_list}'
            mask_file_path = os.path.join(mask_folder, mask_file_list[0])
            mask_t_orig = torch.load(mask_file_path)  # torch.Size([3, 200, 200])

            img_size = (images.shape[-2], images.shape[-1])
            img_height, img_width = img_size
            image_t = images.squeeze(0).to(self.device)       # torch.Size([3, 416, 416])
            label = labels[0]       # torch.Size([N, 6])
            # obtain object box size
            box_x1, box_y1, box_x2, box_y2 = label[0][1], label[0][2], label[0][3], label[0][4]
            box_width = int(box_x2 - box_x1)
            box_height = int(box_y2 - box_y1)
            box_area = box_width * box_height
            # mask process （need resize and binarization）
            patch_width, patch_height = int(box_width * patch_scale), int(box_height * patch_scale)
            mask_t_resize = F.interpolate(mask_t_orig.unsqueeze(0), size=(patch_height, patch_width), mode='bilinear', align_corners=False).squeeze(0)
            adv_mask_melt = torch.mean(mask_t_resize, dim=0, keepdim=True)
            binary_adv_mask = (adv_mask_melt > binarization_thresh).int()
            binary_adv_mask = binary_adv_mask.int()   # torch.Size([1, patch_box_h, patch_box_w])
            '''
            # visualization
            binary_adv_mask = binary_adv_mask.repeat(3, 1, 1)
            adv_mask_vis = binary_adv_mask * 255.0
            mask_np = adv_mask_vis.detach().cpu().numpy()
            mask_np = mask_np.transpose(1, 2, 0).astype(np.uint8)
            mask_save = Image.fromarray(mask_np)      # 按照RGB格式保存
            mask_save.save(f'output_advshape_mask_{basename}.jpg')
            assert False
            '''
            # area calculation
            adv_mask_area = binary_adv_mask.sum().item()       # 1 channels
            area_ratio = adv_mask_area / box_area
            total_area_ratio += area_ratio
        # calculate final result
        mean_area = total_area_ratio / total_image_number
        print(f'mean_area: {mean_area}')





if __name__ == "__main__":
    # Wei proposed infrared yolov3, input specific, infrared
    workspace = f'./FourierAttack_test/exp0'
    model= load_infrared_model()
    learnable_c_folder = './FourierAttack_train/exp0/learnable_c'
    fourierattack_test = FourierShape_Infrared_InputSpecific_Test(model=model, workspace=workspace, learnable_c_folder=learnable_c_folder)
    dataloader = LLVIPDataloader(
        img_dir='./LLVIP_person/instances_imgs',
        label_dir='./LLVIP_person/instances_labels',
        batch_size=1
    )
    fourierattack_test.test_asr(dataloader)
    fourierattack_test.test_APdrop(dataloader)

    
