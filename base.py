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
from attack_utils.test import calculate_iou

# YOLOv3, env = 'yolov3'
from yolov3.detect_custom import detect_batch, detect_AP


cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(6)

class BaseFourierAttack:
    """
    base class of Fourier attack
    """
    def __init__(self, model, workspace):
        # initialize general parameters
        self.model = model
        self.workspace = workspace
        # weights
        self.conf_weight = 3.0
        self.reg_weight = 1.0
        # ablation parameters
        self.N = 6
        self.patch_scale = 0.6
        # other parameters
        self.epsilon = 1e-4
        self.vis_conf_threshold = 0.1
        self.num_epochs = 500
        self.lr_init = 0.002
        self.c_init = self.initialize_c(N=self.N)
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # patch applier
        self.transformer = PatchTransformer().to(self.device)
        self.applier = PatchApplier().to(self.device)
        # saving folder
        self.c_path = os.path.join(self.workspace, 'learnable_c')       # final learnable c (.json) file
        self.c_curve = os.path.join(self.workspace, 'c_curve')      # final learnabel fourier curve 
        self.advexamples_nodetect = os.path.join(self.workspace, 'advexamples_nodetect') # final undetected adversarial examples 
        self.advexamples_detect = os.path.join(self.workspace, 'advexamples_detect')    # final detected adversarial examples
        for path in [self.c_path, self.c_curve, self.advexamples_nodetect, self.advexamples_detect]:
            try:
                # create directory (if already exists, ignore)
                os.makedirs(path, mode=0o777, exist_ok=True)
                # confirm permission (some system need to set separately)
                os.chmod(path, 0o777)
                print(f"Successfully create new file: {path}")
            except Exception as e:
                print(f"Warning: create {path} fail: {str(e)}")
    
    def make_fourier_curve(self, c):
        """
        Generating Fourier curve.
        """
        index = torch.arange(-int(len(c)/2), int(len(c)/2)+1)
        # Calculating real and imaginary parts
        def X_func(t):
            exponents = torch.tensor(index, dtype=t.dtype).to(t.device)
            t_expanded = t.view(-1, 1)
            exp_terms = torch.exp(1j * t_expanded * exponents).to(t.device)
            f_t = torch.sum(c * exp_terms, dim=1)
            return torch.real(f_t)
        def Y_func(t):
            exponents = torch.tensor(index, dtype=t.dtype).to(t.device)
            t_expanded = t.view(-1, 1)
            exp_terms = torch.exp(1j * t_expanded * exponents).to(t.device)
            f_t = torch.sum(c * exp_terms, dim=1)
            return torch.imag(f_t)
        return X_func, Y_func

    def differentiable_winding_number(self, x, y, t, X_func, Y_func, epsilon=1e-3):
        t_tensor = t.clone().detach().to(torch.float32).requires_grad_(True).to(t.device)
        X = X_func(t_tensor)
        Y = Y_func(t_tensor)
        dX_dt = torch.autograd.grad(X, t_tensor, 
                                grad_outputs=torch.ones_like(X),
                                create_graph=True,
                                retain_graph=True)[0]
        dY_dt = torch.autograd.grad(Y, t_tensor, 
                                grad_outputs=torch.ones_like(Y),
                                create_graph=True,
                                retain_graph=True)[0]
        X = X.unsqueeze(-1).to(t.device)
        Y = Y.unsqueeze(-1).to(t.device)
        dX_dt = dX_dt.unsqueeze(-1).to(t.device)
        dY_dt = dY_dt.unsqueeze(-1).to(t.device)
        # Calculating winding number W(q)
        numerator = (X - x) * dY_dt - (Y - y) * dX_dt
        denominator = (X - x)**2 + (Y - y)**2 + epsilon
        integrand = numerator / denominator
        G = torch.trapz(integrand, t_tensor, dim=0) / (2 * np.pi)
        return G

    def initialize_c(self, N=6, max_iter=1000, complexity=0.7):       # original：0.7
        """
        Initilizing C
        Params:
            N: max harmonic order (default: 6)
            max_iter: max iteration number (default: 1000)
            complexity: complexity of curve (0-1, default: 0.7)
        Returns:
            c: list of complex, len=2N+1, index from k=-N to k=N
        """
        # angle parameters
        t_points = np.linspace(0, 2*np.pi, 1000)
        # complexity control parameters
        decay_base = 0.8 * (1 - 0.4 * complexity)  # 衰减因子
        max_high_freq = 0.1 * (0.5 + complexity)  # 高频分量最大振幅
        for _ in range(max_iter): 
            c = np.zeros(2*N+1, dtype=complex)
            # x direction main components
            c_x_main = []
            for k in range(1, 4):  # 优先使用低阶谐波
                mag = np.random.uniform(0.2, 0.3) * np.exp(-decay_base * k)
                phase = np.random.uniform(0, 2*np.pi)
                c_x_main.append(mag * np.exp(1j * (phase + np.pi/2)))  # 相位偏移使x方向扩展
            # y direction main components
            c_y_main = []
            for k in range(1, 4):
                mag = np.random.uniform(0.2, 0.3) * np.exp(-decay_base * k)
                phase = np.random.uniform(0, 2*np.pi)
                c_y_main.append(mag * np.exp(1j * phase))  # 无相位偏移使y方向扩展
            for k in range(1, min(N, 4)+1):
                idx = N + np.random.choice([k, -k])
                if c_x_main and np.random.rand() > 0.3:
                    c[idx] += c_x_main.pop(0)
                elif c_y_main:
                    c[idx] += c_y_main.pop(0)
            for k in range(-N, N+1):
                idx = k + N
                if c[idx] == 0:
                    if k == 0:
                        c[idx] = np.random.uniform(-0.02, 0.02) + 1j*np.random.uniform(-0.02, 0.02)
                    else:
                        decay_factor = np.exp(-decay_base * abs(k))
                        max_mag = max_high_freq * decay_factor
                        if complexity > 0.5 and abs(k) > 2:
                            mag = np.random.uniform(0.01, max_mag)
                        else:
                            mag = np.random.uniform(0, max_mag)
                        phase = np.random.uniform(0, 2*np.pi)
                        c[idx] = mag * np.exp(1j * phase)
            # evaluate the curve
            z = np.zeros_like(t_points, dtype=complex)
            for i, k in enumerate(range(-N, N+1)):
                z += c[i] * np.exp(1j * k * t_points)
            x, y = np.real(z), np.imag(z)
            max_x = np.max(np.abs(x))
            max_y = np.max(np.abs(y))
            area_fill_x = max_x / 0.5
            area_fill_y = max_y / 0.5
            area_fill = min(area_fill_x, area_fill_y)
            if area_fill < 0.85:
                # 计算当前最大扩展方向
                max_val = max(max_x, max_y, 0.01)
                scale_factor = 0.45 / max_val
                c *= scale_factor
                x, y = np.real(z * scale_factor), np.imag(z * scale_factor)
                max_x = np.max(np.abs(x))
                max_y = np.max(np.abs(y))
                area_fill_x = max_x / 0.5
                area_fill_y = max_y / 0.5
                area_fill = min(area_fill_x, area_fill_y)
            x_range = np.max(x) - np.min(x)
            y_range = np.max(y) - np.min(y)
            complexity_score = (np.std(np.diff(x)) + np.std(np.diff(y))) * 100
            target_low_complexity = 0.3 + 0.4 * complexity
            target_high_complexity = 0.5 + 0.6 * complexity
            if (area_fill >= 0.85 and
                max_x > 0.4 and max_y > 0.4 and
                target_low_complexity < complexity_score < target_high_complexity and
                x_range > 0.7 and y_range > 0.7):
                return c.tolist()
        print('Warrning: using default c_init_parameters')
        c_init = [
            0.0 - 0.0j, 0.0 + 0.0j, 0.0 - 0.0j, 0.0 + 0.0j, 0.0 + 0.125j, -0.25 + 0.0j,
            0.0 - 0.0j,
            0.25 - 0.0j, 0.0 - 0.125j, 0.0 + 0.0j, 0.0 - 0.0j, 0.0 - 0.0j, 0.0 + 0.0j
        ]
        return c_init

    def fourier_constraints(self, c: torch.Tensor, epsilon: float = 0.25, base_dominance: float = 2.0):
        """
        Fourier constraints for not self-intersection
        Args:
            c: complex tensors [c_{-k}, c_{-k+1}, ..., c_0, ..., c_{k-1}, c_k]
            epsilon: Harmonic amplitude constraint coefficient (default: 0.25)
            base_dominance: Fundamental frequency dominant constraint coefficient (default: 4.0)
        """
        assert len(c) % 2 == 1, "length of c must be odd"
        k = (len(c) - 1) // 2
        center_idx = k
        if k >= 1:
            c_neg1 = c[center_idx - 1]  # c_{-1}
            c_pos1 = c[center_idx + 1]  # c_{1}
        else:
            return torch.tensor(0.0, device=c.device)
        base_mag = torch.abs(c_neg1) + torch.abs(c_pos1)
        min_base = torch.min(torch.abs(c_neg1), torch.abs(c_pos1))
        # regression loss calculation
        reg_loss = torch.tensor(0.0, device=c.device)
        high_freq_mag = torch.tensor(0.0, device=c.device)
        for i, coeff in enumerate(c):
            freq = i - k
            if abs(freq) >= 2:
                high_freq_mag += torch.abs(coeff)
        reg_loss += torch.relu(high_freq_mag * base_dominance - base_mag)
        for i, coeff in enumerate(c):
            freq = i - k
            if abs(freq) >= 2:
                reg_loss += torch.relu(torch.abs(coeff) - epsilon * min_base)
        return reg_loss
    
    def generate_G(self, learnable_c):
        """
        generate G_values_t from complex learnable tensor learnable_c.
        Args:
            learnable_c (torch.Tensor): complex tensor [c_{-k}, c_{-k+1}, ..., c_0, ..., c_{k-1}, c_k]
        Returns:
            G_values_t (torch.Tensor): G_values_t, a [0, 1] shape mask, torch.Size([3, 200, 200]), value=1 in Fourier shape while value=0 out of Fourier shape
        """
        X_func, Y_func = self.make_fourier_curve(learnable_c)
        t = torch.linspace(0, 2 * np.pi, 1000).to(self.device)
        bound_size = 0.5                
        x_grid = torch.linspace(-bound_size, bound_size, 200).to(self.device)
        y_grid = torch.linspace(-bound_size, bound_size, 200).to(self.device)
        X_mesh, Y_mesh = torch.meshgrid(x_grid, y_grid, indexing='xy')
        x_flat = X_mesh.reshape(-1)
        y_flat = Y_mesh.reshape(-1)
        G_values = self.differentiable_winding_number(x_flat, y_flat, t, X_func, Y_func, self.epsilon)
        G_values_abs = torch.abs(G_values)
        G_values_abs_clamp = torch.clamp(G_values_abs, min=0.0, max=1.0)
        G_values_reshape = G_values_abs_clamp.reshape(X_mesh.shape)   # torch.Size([100, 100])
        G_values_t = G_values_reshape.unsqueeze(0).repeat(3, 1, 1)  # torch.Size([3, patch_w, patch_h])
        return G_values_t
    
    def visualize_winding_number_curve(self, c, filename='winding_numer_curve.png'):
        """
        visualize winding number field
        positive x-axis direction: ->, keep the same with image
        """
        X_func, Y_func = self.make_fourier_curve(c)
        t = torch.linspace(0, 2 * np.pi, 1000).to(self.device)
        bound_size=0.5
        x_grid = torch.linspace(-bound_size, bound_size, 200).to(self.device)
        y_grid = torch.linspace(-bound_size, bound_size, 200).to(self.device)
        X_mesh, Y_mesh = torch.meshgrid(x_grid, y_grid, indexing='xy')
        x_flat = X_mesh.reshape(-1)
        y_flat = Y_mesh.reshape(-1)
        # Calculating G
        G_values = self.differentiable_winding_number(x_flat, y_flat, t, X_func, Y_func, self.epsilon)
        G_values = torch.abs(G_values)
        G_values = G_values.reshape(X_mesh.shape).detach().cpu().numpy()
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(
            X_mesh.cpu().numpy(), 
            -Y_mesh.cpu().numpy(),      # Y_mesh.cpu().numpy()
            G_values, 
            levels=50, 
            cmap='coolwarm')
        plt.colorbar(contour, label='Winding Number G(x,y)')
        # plot curve
        t_curve_tensor = t.clone().detach().to(torch.float32)
        x_curve = X_func(t_curve_tensor).detach().cpu().numpy()
        y_curve = -Y_func(t_curve_tensor).detach().cpu().numpy()     # y_curve = Y_func(t_curve_tensor).detach().cpu().numpy()
        plt.plot(x_curve, y_curve, 'k-', linewidth=1, label='curve')
        # plot image
        plt.title("fourier curve winding number filed")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')
        # save image
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    def generate_patch_boxes_t(self, label):
        """
        calculate boxes coordinates for applying adversarial patches. patch size should be 0.6 * GTbox 
        add scale to input, should used in MMDetection
        Args:
            label: ground truth label, extract from dataloader. torch.Size([N, 5]) N:object number
        Returns:
            patch_bboxes: patch boxes tensor, torch.Size([N, 4])
        """
        patches = []
        for obj_i in range(len(label)):
            parts = label[obj_i]        # ith object
            class_id, obj_left, obj_up, obj_right, obj_buttom = \
                        int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])       # [x1, y1, x2, y2] annotations
            obj_w = obj_right - obj_left
            obj_h = obj_buttom - obj_up
            obj_center_x = obj_left + obj_w / 2.0
            obj_center_y = obj_up + obj_h / 2.0
            patch_size_w, patch_size_h = obj_w * self.patch_scale, obj_h * self.patch_scale
            patch_center_x, patch_center_y = obj_center_x , obj_center_y
            # adding to patches list
            patches.append([patch_center_x, patch_center_y, patch_size_w, patch_size_h])
        patch_bboxes = torch.tensor(patches, dtype=torch.float32, device=self.device)
        return patch_bboxes
    
    def train(self):
        """
        training function
        """
        pass


class BaseTest(BaseFourierAttack):
    """
    base test class of fourier shape attack.
    """
    def __init__(self, model, workspace):
        super(BaseTest, self).__init__(model, workspace)
        self.iou_thresh = 0.45
    
    def generate_mask_from_cjson(self, json_file):
        """
        generating mask G_values_t from fourier coefficients c.
        Args:
          json_file: json file of fourier coefficients (.json)
        Returns:
          G_values_t: mask tensor G_values_t, torch.Size([3, patch_w, patch_h])
        """
        # initialize generating G_values
        t = torch.linspace(0, 2 * np.pi, 1000).to(self.device)
        bound_size = 0.5
        x_grid = torch.linspace(-bound_size, bound_size, 200).to(self.device)
        y_grid = torch.linspace(-bound_size, bound_size, 200).to(self.device)
        X_mesh, Y_mesh = torch.meshgrid(x_grid, y_grid, indexing='xy')
        x_flat = X_mesh.reshape(-1)
        y_flat = Y_mesh.reshape(-1)
        # load json file
        with open(json_file, 'r') as f:
            coeff_dict = json.load(f)
        # harmonic range
        harmonics = sorted([int(k) for k in coeff_dict.keys()])
        min_harmonic, max_harmonic = harmonics[0], harmonics[-1]
        k = max(abs(min_harmonic), abs(max_harmonic))
        # data integrity
        expected_keys = sorted([str(i) for i in range(-k, k+1)])
        actual_keys = sorted(coeff_dict.keys())
        if expected_keys != actual_keys:
            raise ValueError(f"JSON keys mismatch. Expected harmonics: {-k} to {k}, found: {min_harmonic} to {max_harmonic}")
        # Create a complex list (sorted by harmonic number from -k to+k)
        complex_list = []
        for harmonic in range(-k, k+1):
            real_part = coeff_dict[str(harmonic)][0]
            imag_part = coeff_dict[str(harmonic)][1]
            complex_list.append(complex(real_part, imag_part))
        c = torch.tensor(complex_list, dtype=torch.complex64).to(self.device)
        learnable_c = nn.Parameter(c, requires_grad=True)
        # Generating G_values_t
        X_func, Y_func = self.make_fourier_curve(learnable_c)
        G_values = self.differentiable_winding_number(x_flat, y_flat, t, X_func, Y_func, self.epsilon)
        G_values_abs = torch.abs(G_values)
        G_values_abs_clamp = torch.clamp(G_values_abs, min=0.0, max=1.0)
        G_values_reshape = G_values_abs_clamp.reshape(X_mesh.shape)   # torch.Size([100, 100])
        G_values_t = G_values_reshape.unsqueeze(0).repeat(3, 1, 1)  # torch.Size([3, patch_w, patch_h])

        return G_values_t
    
    def generate_patch_boxes_t(self, label):
        """
        calculate boxes coordinates for applying adversarial patches. patch size should be 0.6 * GTbox 
        Args:
            label: ground truth label, extract from dataloader. torch.Size([N, 5]) N:object number
        Returns:
            patch_bboxes: patch boxes tensor, torch.Size([N, 4])
        """
        patches = []
        for obj_i in range(len(label)):
            parts = label[obj_i]        # ith object
            class_id, obj_left, obj_up, obj_right, obj_buttom = \
                        int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])       # [x1, y1, x2, y2] annotations
            obj_w = obj_right - obj_left
            obj_h = obj_buttom - obj_up
            obj_center_x = obj_left + obj_w / 2.0
            obj_center_y = obj_up + obj_h / 2.0
            patch_size_w, patch_size_h = obj_w * self.patch_scale, obj_h * self.patch_scale
            patch_center_x, patch_center_y = obj_center_x , obj_center_y
            # adding to patches list
            patches.append([patch_center_x, patch_center_y, patch_size_w, patch_size_h])
        patch_bboxes = torch.tensor(patches, dtype=torch.float32, device=self.device)
        return patch_bboxes
    
    def calculate_success_attack(self, orig_images, adv_examples, label, conf_thresh):
        """
        calculate ASR from detection results
        Args:
            orig_images: original images tensor, torch.Size([1, 3, img_h, img_w])
            adv_examples: adversarial examples tensor, torch.Size([1, 3, img_h, img_w])
            label: ground truth labels on 1 image (N = 1)
            conf_thresh: confidence threshold. if confidence < conf_thresh, consider it as a successful attack
        Returns:
            success_num: number of successful attacks
            total_num: total number of attacking attempts
        """
        # step 1: obtain benign image detection result
        orig_boxes, orig_confs = detect_batch(self.model, orig_images)
        orig_boxes = orig_boxes[0]
        # step 2: obtain adversarial image detection result
        adv_boxes, adv_confs = detect_batch(self.model, adv_examples)
        adv_boxes = adv_boxes[0]
        adv_confs = adv_confs[0]
        # step 3: obtain ground truth
        gt_boxes = []
        for obj in label:
            class_id, left, up, right, buttom = obj
            gt_boxes.append([left, up, right, buttom])
        # step 4: matches attack success
        success_num = 0
        total_num = 0
        for target in gt_boxes:
            detected = False
            for box, conf in zip(adv_boxes, adv_confs):
                if conf < conf_thresh:
                    continue
                if calculate_iou(box, target) > self.iou_thresh:
                    detected = True
                    break
            if not detected:
                success_num += 1
            total_num += 1
        return success_num, total_num
    
    def obtain_preds_gts(self, images, label, image_id):
        """
        obtain prediction results and ground truth of input images (torch.Size([1, C, H, W]))
        Args:
            images: input images tensor, torch.Size([1, C, H, W])
            label: ground truth labels on 1 image (N = 1)
            image_id: image id (int)
        Returns:
            preds_elem: prediction results on 1 image, {'boxes':..., 'scores':..., 'image_id':...}
            targets_elem: ground truth, {'boxes':..., 'image_id':...}
        """
        yolo_img_size = 416
        img_height, img_width = images.shape[-2], images.shape[-1]
        # step 1: obtain benign image detection result under origin size image
        preds = detect_AP(self.model, images, conf_thr=0.001)       # results are detected on yolo_img_size * yolo_img_size image, conf_default=0.001
        pred = preds[0]
        boxes = pred[:,  :4].tolist()
        clipped_boxes = []
        for box in boxes:
            left = max(0, min(box[0], yolo_img_size))
            right = max(0, min(box[2], yolo_img_size))
            top = max(0, min(box[1], yolo_img_size))
            bottom = max(0, min(box[3], yolo_img_size))
            clipped_boxes.append([left, top, right, bottom])
        confs = pred[:, 4].tolist()
        scale_x = img_width / yolo_img_size
        scale_y = img_height / yolo_img_size
        boxes_rescale = [[box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y] for box in clipped_boxes]
        # step 2: obtain ground truth, under origin size image
        gt_boxes = []
        for obj in label:
            class_id, left, up, right, buttom = obj
            gt_boxes.append([left, up, right, buttom])
        # step 3: adding results to list
        preds_elem = {'boxes': boxes_rescale, 'scores': confs, 'image_id': image_id}
        targets_elem = {'boxes': gt_boxes, 'image_id': image_id}

        return preds_elem, targets_elem


    def test_asr(self, dataloader):
        """
        calculating ASR
        """
        pass
    
    def test_AP  drop(self, dataloader):
        """
        calculating APdrop
        """
        pass


