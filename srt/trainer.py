import torch
import numpy as np
from tqdm import tqdm

import srt.utils.visualize as vis
from srt.utils.common import mse2psnr, reduce_dict, gather_all
from srt.utils import nerf
from srt.utils.common import get_rank, get_world_size

import os
import math
from collections import defaultdict
import torchvision.utils as vutils
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure as ssim
import lpips

class SRTTrainer:
    def __init__(self, model, optimizer, cfg, device, out_dir, render_kwargs):
        self.model = model
        self.optimizer = optimizer
        self.config = cfg
        self.device = device
        self.out_dir = out_dir
        self.render_kwargs = render_kwargs
        # Initialize LPIPS model
        self.lpips_model = lpips.LPIPS(net='vgg').cuda()
        if 'num_coarse_samples' in cfg['training']:
            self.render_kwargs['num_coarse_samples'] = cfg['training']['num_coarse_samples']
        if 'num_fine_samples' in cfg['training']:
            self.render_kwargs['num_fine_samples'] = cfg['training']['num_fine_samples']

    def evaluate(self, val_loader, **kwargs):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        self.model.eval()
        eval_lists = defaultdict(list)

        # loader = val_loader if get_rank() > 0 else tqdm(val_loader)
        loader = val_loader
        sceneids = []

        for itr, data in tqdm(enumerate(loader)):
            sceneids.append(data['sceneid'])
            eval_step_dict = self.eval_step(data, itr, **kwargs)

            for k, v in eval_step_dict.items():
                eval_lists[k].append(v)

        sceneids = torch.cat(sceneids, 0).cuda()
        sceneids = torch.cat(gather_all(sceneids), 0)

        print(f'Evaluated {len(torch.unique(sceneids))} unique scenes.')

        eval_dict = {k: torch.cat(v, 0) for k, v in eval_lists.items()}
        eval_dict = reduce_dict(eval_dict, average=True)  # Average across processes
        eval_dict = {k: v.mean().item() for k, v in eval_dict.items()}  # Average across batch_size
        print('Evaluation results:')
        print(eval_dict)
        return eval_dict

    def train_step(self, data, it):
        self.model.train()
        self.optimizer.zero_grad()
        loss, loss_terms = self.compute_loss(data, it)
        loss = loss.mean(0)
        loss_terms = {k: v.mean(0).item() for k, v in loss_terms.items()}
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_terms

    def save_image(self, pixels, path="/scratch/as3ek/github/yinzhu/", name="test.png"):
        images = pixels.view(pixels.size(0), 64, 64, 3)
        # Assuming images is your tensor of size torch.Size([32, 64, 64, 3])
        # Convert the tensor to the format expected by make_grid (C x H x W)
        images = images.permute(0, 3, 1, 2)  # Convert to (32, 3, 64, 64)
        # Create a grid of images
        grid = vutils.make_grid(images, nrow=8)  # Arrange images into an 8x4 grid
        # Convert the grid to a numpy array
        ndarr = grid.mul(255).byte().permute(1, 2, 0).cpu().numpy()
        # Create a PIL image
        im = Image.fromarray(ndarr)
        # Save the image as a PNG file
        im.save(os.path.join(path, name))

    def calculate_ssim_lpips(self, batch1, batch2):
        """
        Calculate the SSIM between two batches of images.

        Args:
        batch1 (torch.Tensor): First batch of images with shape [B, H, W, C].
        batch2 (torch.Tensor): Second batch of images with shape [B, H, W, C].

        Returns:
        torch.Tensor: Tensor containing SSIM values for each pair of images in the batches.
        """
        # Ensure the input batches are in the shape [B, H, W, C]
        if batch1.shape != batch2.shape:
            raise ValueError("Input batches must have the same shape")
        
        batch1 = batch1.view(-1, 64, 64, 3)
        batch2 = batch2.view(-1, 64, 64, 3)

        # Permute the batches to match the expected format (B, C, H, W)
        batch1 = batch1.permute(0, 3, 1, 2)
        batch2 = batch2.permute(0, 3, 1, 2)

        # Calculate SSIM for each pair of images in the batches
        ssim_values = [ssim(img1.unsqueeze(0), img2.unsqueeze(0)) for img1, img2 in zip(batch1, batch2)]

        # Convert the list of SSIM values to a tensor
        ssim_values = torch.tensor(ssim_values)

        # Calculate LPIPS for each pair of images in the batches
        lpips_values = [self.lpips_model(img1.unsqueeze(0), img2.unsqueeze(0)).item() for img1, img2 in zip(batch1, batch2)]
        lpips_values = torch.tensor(lpips_values)

        return ssim_values, lpips_values

    def compute_loss(self, data, it):
        # import ipdb; ipdb.set_trace()
        device = self.device

        input_images = data.get('input_images').to(device)
        input_camera_pos = data.get('input_camera_pos').to(device)
        input_rays = data.get('input_rays').to(device)
        target_pixels = data.get('target_pixels').to(device)

        z = self.model.encoder(input_images, input_camera_pos, input_rays)

        target_camera_pos = data.get('target_camera_pos').to(device)
        target_rays = data.get('target_rays').to(device)

        loss = 0.
        loss_terms = dict()

        pred_pixels, extras = self.model.decoder(z, target_camera_pos, target_rays, **self.render_kwargs)

        loss = loss + ((pred_pixels - target_pixels)**2).mean((1, 2))

        loss_terms['mse'] = loss

        # rand = np.random.randint(1, 10001)
        # self.save_image(pixels=pred_pixels, name=f"pred{rand}.png")
        # self.save_image(pixels=target_pixels, name=f"target{rand}.png")

        input_images_save = input_images.squeeze(1)

        pred_images = pred_pixels.view(pred_pixels.size(0), 64, 64, 3)
        pred_images = pred_images.permute(0, 3, 1, 2)  # Convert to (32, 3, 64, 64)

        target_images = target_pixels.view(target_pixels.size(0), 64, 64, 3)
        target_images = target_images.permute(0, 3, 1, 2)  # Convert to (32, 3, 64, 64)

        # Save the tensor to a .pt file
        torch.save(pred_images, f"/scratch/as3ek/github/yinzhu/srt_plane_samples/pred{it}.pt")
        torch.save(target_images, f"/scratch/as3ek/github/yinzhu/srt_plane_samples/target{it}.pt")
        torch.save(input_images_save, f"/scratch/as3ek/github/yinzhu/srt_plane_samples/input{it}.pt")

        loss_terms['ssim'], loss_terms['lpips'] = self.calculate_ssim_lpips(pred_pixels, target_pixels)

        if 'coarse_img' in extras:
            coarse_loss = ((extras['coarse_img'] - target_pixels)**2).mean((1, 2))
            loss_terms['coarse_mse'] = coarse_loss
            loss = loss + coarse_loss

        return loss, loss_terms

    def eval_step(self, data, itr, full_scale=False):
        with torch.no_grad():
            loss, loss_terms = self.compute_loss(data, itr)

        mse = loss_terms['mse']
        psnr = mse2psnr(mse)
        return {'psnr': psnr, 'mse': mse, **loss_terms}

    def render_image(self, z, camera_pos, rays, **render_kwargs):
        """
        Args:
            z [n, k, c]: set structured latent variables
            camera_pos [n, 3]: camera position
            rays [n, h, w, 3]: ray directions
            render_kwargs: kwargs passed on to decoder
        """
        batch_size, height, width = rays.shape[:3]
        rays = rays.flatten(1, 2)
        camera_pos = camera_pos.unsqueeze(1).repeat(1, rays.shape[1], 1)

        max_num_rays = self.config['data']['num_points'] * \
                self.config['training']['batch_size'] // (rays.shape[0] * get_world_size())
        num_rays = rays.shape[1]
        img = torch.zeros_like(rays)
        all_extras = []
        for i in range(0, num_rays, max_num_rays):
            img[:, i:i+max_num_rays], extras = self.model.decoder(
                z=z, x=camera_pos[:, i:i+max_num_rays], rays=rays[:, i:i+max_num_rays],
                **render_kwargs)
            all_extras.append(extras)

        agg_extras = {}
        for key in all_extras[0]:
            agg_extras[key] = torch.cat([extras[key] for extras in all_extras], 1)
            agg_extras[key] = agg_extras[key].view(batch_size, height, width, -1)

        img = img.view(img.shape[0], height, width, 3)
        return img, agg_extras


    def visualize(self, data, mode='val'):
        self.model.eval()

        with torch.no_grad():
            device = self.device
            input_images = data.get('input_images').to(device)
            input_camera_pos = data.get('input_camera_pos').to(device)
            input_rays = data.get('input_rays').to(device)

            camera_pos_base = input_camera_pos[:, 0]
            input_rays_base = input_rays[:, 0]

            if 'transform' in data:
                # If the data is transformed in some different coordinate system, where
                # rotating around the z axis doesn't make sense, we first undo this transform,
                # then rotate, and then reapply it.
                
                transform = data['transform'].to(device)
                inv_transform = torch.inverse(transform)
                camera_pos_base = nerf.transform_points_torch(camera_pos_base, inv_transform)
                input_rays_base = nerf.transform_points_torch(
                    input_rays_base, inv_transform.unsqueeze(1).unsqueeze(2), translate=False)
            else:
                transform = None

            input_images_np = np.transpose(input_images.cpu().numpy(), (0, 1, 3, 4, 2))

            z = self.model.encoder(input_images, input_camera_pos, input_rays)

            batch_size, num_input_images, height, width, _ = input_rays.shape

            num_angles = 6

            columns = []
            for i in range(num_input_images):
                header = 'input' if num_input_images == 1 else f'input {i+1}'
                columns.append((header, input_images_np[:, i], 'image'))

            all_extras = []
            for i in range(num_angles):
                angle = i * (2 * math.pi / num_angles)
                angle_deg = (i * 360) // num_angles

                camera_pos_rot = nerf.rotate_around_z_axis_torch(camera_pos_base, angle)
                rays_rot = nerf.rotate_around_z_axis_torch(input_rays_base, angle)

                if transform is not None:
                    camera_pos_rot = nerf.transform_points_torch(camera_pos_rot, transform)
                    rays_rot = nerf.transform_points_torch(
                        rays_rot, transform.unsqueeze(1).unsqueeze(2), translate=False)

                img, extras = self.render_image(z, camera_pos_rot, rays_rot, **self.render_kwargs)
                all_extras.append(extras)
                columns.append((f'render {angle_deg}°', img.cpu().numpy(), 'image'))

            for i, extras in enumerate(all_extras):
                if 'depth' in extras:
                    depth_img = extras['depth'].unsqueeze(-1) / self.render_kwargs['max_dist']
                    depth_img = depth_img.view(batch_size, height, width, 1)
                    columns.append((f'depths {angle_deg}°', depth_img.cpu().numpy(), 'image'))

            output_img_path = os.path.join(self.out_dir, f'renders-{mode}')
            vis.draw_visualization_grid(columns, output_img_path)

