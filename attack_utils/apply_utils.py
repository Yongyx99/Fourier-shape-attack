"""
patch apply utils
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace as st
import numpy as np
from PIL import Image
import cv2

class PatchTransformer(nn.Module):
    """
    PatchTransformer: transforms batch of patches
    Module providing the functionality necessary to transform a batch of patches by: 
        - randomly adjusting brightness and contrast, 
        - adding random amount of noise,
        - rotating randomly, and 
        - resizing patches according to as size based on the batch of labels, 
          and pads them to the dimension of an image.
    """
    def __init__(self,
                 min_contrast=0.80,
                 max_contrast=1.20,
                 min_brightness=-0.10,
                 max_brightness=0.10,
                 noise_factor=0.10):
        super(PatchTransformer, self).__init__()
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.noise_factor = noise_factor
    def resize_rotate(self, adv_batch, fake_bboxes, img_size):
        """
        patch resize and rotate according to fake_bboxes_batch
        Parameters:
            adv_batch (tensor): adv_batch after expand of adv_patch, torch.Size([patch_number, 3, patch_size, patch_size]), for 1 image
            fake_bboxes (Tensor): fake bounding boxes. tensor.Size([patch_number, 4]), [center_x, center_y, w, h]
            img_size (tuple): original image size. (H, W)
        Returns:
            adv_batch_masked (tensor): patch on the image. 0 around patch location, and patch_value on patch location. torch.Size([patch_number, 3, img_size[0], img_size[1]])
            msk_batch (Tensor): mask of patch. 0 around patch location, and 1 on patch location. torch.Size([patch_number, 3, img_size[0], img_size[1]])
        """
        H, W = img_size
        patch_number = adv_batch.size(0)
        msk_batch = torch.ones_like(adv_batch[:, :1])  # [patch_number, 1, patch_size_h, patch_size_w]

        center_x = fake_bboxes[:, 0]  # [patch_number]
        center_y = fake_bboxes[:, 1]
        w = fake_bboxes[:, 2].clamp(min=1.0)
        h = fake_bboxes[:, 3].clamp(min=1.0)

        center_x_norm = (center_x / (W-1)) * 2 - 1
        center_y_norm = (center_y / (H-1)) * 2 - 1

        scale_w = W / w
        scale_h = H / h

        affine_matrix = torch.zeros((patch_number, 2, 3), device=adv_batch.device)
        affine_matrix[:, 0, 0] = scale_w
        affine_matrix[:, 1, 1] = scale_h
        affine_matrix[:, 0, 2] = -center_x_norm * scale_w
        affine_matrix[:, 1, 2] = -center_y_norm * scale_h

        output_size = torch.Size([patch_number, 3, H, W])

        grid = F.affine_grid(affine_matrix, output_size, align_corners=False)
        adv_transformed = F.grid_sample(adv_batch, grid, align_corners=False)       # torch.Size([patch_number, 3, H, W])
        msk_transformed = F.grid_sample(msk_batch, grid, align_corners=False)       # torch.Size([patch_number, 1, H, W])

        # # vis
        # adv_transformed = adv_transformed.squeeze(0)
        # adv_transformed_np = adv_transformed.detach().cpu().numpy()
        # adv_transformed_np = adv_transformed_np * 255.0
        # adv_transformed_img = adv_transformed_np.transpose(1, 2, 0).astype(np.uint8)
        # save_path = 'test_adv_transformed.png'
        # cv2.imwrite(save_path, adv_transformed_img)

        # mask_transformed = msk_transformed.repeat(1, 3, 1, 1)
        # mask_transformed_vis = mask_transformed.squeeze(0)
        # mask_transformed_np = mask_transformed_vis.detach().cpu().numpy()
        # mask_transformed_np = mask_transformed_np * 255.0
        # mask_transformed_img = mask_transformed_np.transpose(1, 2, 0).astype(np.uint8)
        # save_path = 'test_mask_transformed.png'
        # cv2.imwrite(save_path, mask_transformed_img)
        # assert False

        return adv_transformed, msk_transformed

class PatchApplier(nn.Module):
    """
    PatchApplier: applies adversarial patches to images.
    Params:
        img:
        adv_masked:
    Returns:
        applied_img:
    """
    def __init__(self):
        super(PatchApplier, self).__init__()
    def forward(self, img, adv_masked):        # img：torch.Size([3, H, W]), adv_masked: torch.Size([3, H, W]), adv_masked is all patches on 1 image。
        # # img visualize test
        # img_v = img * 255.0
        # image_np = img_v.cpu().detach().numpy().transpose(1, 2, 0)
        # image_np = image_np.astype(np.uint8)
        # img_np = Image.fromarray(image_np)
        # img_np.save('test_img_in_patchapplier.jpg')

        # # mask visualize test
        # adv_masked_vis = adv_masked * 255
        # mask_np = adv_masked_vis.cpu().detach().numpy().transpose(1, 2, 0)
        # mask_np_rgb = mask_np[:, :, ::-1]
        # mask_np_rgb = mask_np_rgb.astype(np.uint8)
        # mask_np = Image.fromarray(mask_np_rgb)
        # mask_np.save('test_mask_in_patchapplier.jpg')
        # st()

        # img (0-255)，so mask need to multiply 255
        applied_image = (1-adv_masked) * img + adv_masked       # This applies to pure white textures without normalization

        # # save_applied_image test
        # applied_img_np = applied_image.cpu().detach().numpy().transpose(1, 2, 0)
        # applied_img_np_rgb = applied_img_np[:, :, ::-1]
        # applied_img_np_rgb = applied_img_np_rgb.astype(np.uint8)
        # a_img_np = Image.fromarray(applied_img_np_rgb)
        # a_img_np.save('test_appliedimg_in_patchapplier.jpg')
        # assert False

        return applied_image


    def forward_infrared(self, img, adv_masked):
        applied_image = (1-adv_masked) * img
        return applied_image
    
    def forward_color(self, img, adv, mask):
        # # 可视化检查
        # img_v = img * 255.0
        # image_np = img_v.cpu().detach().numpy().transpose(1, 2, 0)
        # image_np = image_np.astype(np.uint8)
        # img_np = Image.fromarray(image_np)
        # img_np.save('test_img_in_patchapplier.jpg')

        # # mask visualize test
        # adv_masked_vis = adv * 255
        # mask_np = adv_masked_vis.cpu().detach().numpy().transpose(1, 2, 0)
        # mask_np_rgb = mask_np[:, :, ::-1]
        # mask_np_rgb = mask_np_rgb.astype(np.uint8)
        # mask_np = Image.fromarray(mask_np_rgb)
        # mask_np.save('test_advmask_in_patchapplier.jpg')
        
        # mask
        # mask_vis = mask * 255
        # mask_np = mask_vis.cpu().detach().numpy().transpose(1, 2, 0)
        # mask_np_rgb = mask_np[:, :, ::-1]
        # mask_np_rgb = mask_np_rgb.astype(np.uint8)
        # mask_np = Image.fromarray(mask_np_rgb)
        # mask_np.save('test_mask_in_patchapplier.jpg')

        applied_image = (1-mask) * img + adv

        # # save_applied_image test
        # applied_img_vis = applied_image * 255.0
        # applied_img_np = applied_img_vis.cpu().detach().numpy().transpose(1, 2, 0)
        # applied_img_np_rgb = applied_img_np[:, :, ::-1]
        # applied_img_np_rgb = applied_img_np_rgb.astype(np.uint8)
        # a_img_np = Image.fromarray(applied_img_np_rgb)
        # a_img_np.save('test_appliedimg_in_patchapplier.jpg')
        # assert False

        return applied_image
