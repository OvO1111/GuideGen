from ldm.models.diffusion.ddpm import (
    LatentDiffusion,
    default,
    instantiate_from_config,
    DDIMSampler,
    noise_like,
    rearrange,
    repeat,
    make_grid,
    LambdaLR
)

import torch, math
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import pytorch_lightning as pl

from omegaconf import OmegaConf
from functools import reduce
from projectors.leap import LeaptorchProjectorWrapper
from projectors.r2gaussian import angle2pose
from r2_gaussian.r2_gaussian.dataset.cameras import Camera
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)
from ldm.data.make_dataset.settings import geometry0825_onerow, geometry0825


def debug_log(x):
    sitk.WriteImage(sitk.GetImageFromArray(((x - x.min()) / (x.max() - x.min())).data.cpu().squeeze().numpy()), '/mnt/data_1/dlr/data/cache/temp.nii.gz')
    return 1


class ProjectionDiffusion2DTest(LatentDiffusion):
    def __init__(self,
                 batch_size=32,
                 label_key='',
                 projector_config=None,
                 normalizer_config=None,
                 use_projected_noise=True,
                 hlcc_n=180,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hlcc_n = hlcc_n
        self.label_key = label_key
        self.batch_size = batch_size
        self.use_projected_noise = use_projected_noise
        assert self.batch_size % self.channels == 0
        self.normalizer = instantiate_from_config(normalizer_config)
        self.projector_config = OmegaConf.to_container(projector_config)
    
    def on_fit_start(self, **kwargs):
        self.projector_config['params']['device'] = self.device
        self.projector_config['params']['batch_size'] = 1 if self.channels == 1 else self.batch_size // self.channels
        self.projector = instantiate_from_config(self.projector_config)

    def get_noise(self):
        noise_3d = noise_like((1 if self.channels == 1 else self.batch_size // self.channels,) +\
                               tuple(self.projector.geometry.n_voxel), self.device, repeat=False)
        noise2ct = lambda x: (x - x.min()) / (x.max() - x.min()) * 2 - 1
        noise_3d = noise2ct(noise_3d)
        noise_2d = self.projector.forward(noise_3d)
        noise_2d = noise2ct(noise_2d) * self.normalizer.oM
        return noise_2d

    def get_hlcc_coef(self, full_proj):
        pi_gt = torch.arange(0, 360, device=self.device) * torch.pi / 180
        t = torch.linspace(-1, 1, full_proj.shape[2], device=self.device).view(1, 1, -1) ** self.hlcc_n
        y_gt = (full_proj * t).sum([-1], keepdim=True).permute(1, 0, 2)  # hxmx1
        A = torch.stack([torch.ones_like(pi_gt)] + \
                        [torch.cos(_ * pi_gt) for _ in range(1, self.hlcc_n + 1)] + \
                        [torch.sin(_ * pi_gt) for _ in range(1, self.hlcc_n + 1)], dim=-1)[None].repeat_interleave(y_gt.shape[0], 0)  # hxmx2n
        AtA, Aty = A.transpose(1, 2) @ A, A.transpose(1, 2) @ y_gt
        coef = torch.linalg.solve(AtA, Aty)
        return coef

    def get_input(self, 
                  batch, 
                  k, 
                  return_first_stage_outputs=False, 
                  force_c_encode=False, 
                  cond_key=None, 
                  return_original_cond=False, 
                  bs=None,
                  return_angles=False,
                  return_hlcc_coefs=False):
        inputs = super().get_input(batch, k, return_first_stage_outputs, force_c_encode, cond_key, return_original_cond, bs)
        hlcc_coef = self.get_hlcc_coef(inputs[0].squeeze())
        # chn = torch.sin(torch.linspace(0, 2*torch.pi, 360)) + 1
        chn = torch.ones((360,))
        chn = torch.multinomial(chn, num_samples=self.batch_size, replacement=False)

        inputs = [i[:, chn].view(-1, 1, *i.shape[2:]) for i in inputs]
        ret = inputs
        if return_angles: 
            angles = super(LatentDiffusion, self).get_input(batch, self.label_key)[:, chn]
            ret += [angles[0].long()]
        if return_hlcc_coefs:
            ret += [hlcc_coef]
        return ret
    
    def shared_step(self, batch, **kwargs):
        x, c, a = self.get_input(batch, self.first_stage_key, return_angles=True)
        loss = self(x, c, angles=a)
        return loss
    
    def p_losses(self, x_start, cond, t, noise=None, angles=None, hlcc_coef=None):
        if not self.use_projected_noise:
            noise = default(noise, lambda: torch.randn_like(x_start))
        else:
            noise_full = self.get_noise()
            noise = noise_full[:, angles.long()].transpose(0, 1)
        
        # x, c = self.dequeue(8)
        # indices = torch.randperm(x_start.shape[0], device=self.device)[:1]
        # self.enqueue(x_start[indices], cond[indices])
        # x_start, cond = torch.cat([x_start, x], dim=0), torch.cat([cond, c], dim=0)

        x_start = x_start.view(-1, self.channels, *self.infer_image_size)
        if cond is not None: cond = cond.view(-1, self.channels, *cond.shape[2:])
        angles = angles.view(-1, self.channels)
        if self.channels != 1:
            noise_index = [[_ for _ in range(self.batch_size)], reduce(lambda x, y: x + y, [[_] * self.channels for _ in range(self.batch_size // self.channels)])]
            noise = noise[*noise_index].view(-1, self.channels, *self.infer_image_size)
            t = t[:self.batch_size // self.channels]

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output, latent = self.apply_model(x_noisy, t, cond, return_latents=True)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3] + [4,] if self.dims == 3 else [])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar.to(self.device)[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean([1, 2, 3] + [4,] if self.dims == 3 else [])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        # temperature = .1
        # batch_size = latent.shape[0]
        # pos_thresh = max(1, 60 * (1 - 10 * self.trainer.current_epoch / self.trainer.max_epochs))
        # feat = nn.functional.normalize(latent.reshape(batch_size, -1), dim=1)
        # sim_matrix = torch.matmul(feat, feat.T) / temperature - torch.eye(batch_size, device=feat.device) * 1e5
        # diff = (angles.unsqueeze(1) - angles.unsqueeze(0)).abs()
        # sim_gt = torch.minimum(diff, 360 - diff)
        # pos_mask = (sim_gt < pos_thresh) & (~torch.eye(batch_size, dtype=torch.bool, device=feat.device))

        # pos_exp = sim_matrix.exp() * pos_mask.float()
        # pos_sum = pos_exp.sum(dim=1)
        # all_sum = sim_matrix.exp().sum(dim=1)

        # loss_cont = (-torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))).mean()
        # loss += loss_cont
        # loss_dict.update({f'{prefix}/loss_cont': loss_cont})

        x_start_hat = self.predict_start_from_noise(x_noisy, t, model_output)
        o = torch.zeros((360,) + tuple(self.projector.geometry.n_detector), device=self.device)
        cond_angles, test_angles = torch.split(angles.flatten(), [(angles).numel() // 2, (angles).numel() // 2])
        o[cond_angles] = x_start_hat[:len(cond_angles), 0]
        bp = self.projector.backward(self.normalizer.backward(o), use_fbp=True)
        fp = self.normalizer.forward(self.projector.forward(bp))

        loss_con = self.get_loss(fp[test_angles], x_start_hat[len(cond_angles):, 0])
        loss_dict.update({f'{prefix}/loss_con': loss_con})
        loss += loss_con * 0.1
        # pi = angles * torch.pi / 180
        # t = torch.linspace(-1, 1, x_start_hat.shape[2], device=self.device).view(1, 1, -1) ** self.hlcc_n
        # y = (x_start_hat * t).sum([-1], keepdim=True).permute(1, 0, 2)  # hxmx1
        # X = torch.stack([torch.ones_like(pi)] + \
        #                 [torch.cos(_ * pi) for _ in range(1, self.hlcc_n + 1)] + \
        #                 [torch.sin(_ * pi) for _ in range(1, self.hlcc_n + 1)], dim=-1)[None].repeat_interleave(y.shape[0], 0)  # hxmx2n
        # loss_hlcc = ((y - X @ hlcc_coef) ** 2).sqrt().mean()
        # loss_dict.update({f'{prefix}/loss_hlcc': loss_hlcc})
        # loss += loss_hlcc * 0.1

        return loss, loss_dict
    
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, angles, **kwargs):
        if cond is not None: cond = cond.view(-1, self.channels, *cond.shape[2:])
        if ddim:
            ddim_sampler = DDIMSampler(self, **self.ddim_config)
            shape = (self.channels, ) + tuple(self.infer_image_size)
            if self.use_projected_noise: 
                x_noisy = self.get_noise()[0:1, angles].transpose(0, 1).view(-1, self.channels, *self.infer_image_size)
                samples, intermediates = ddim_sampler.sample(ddim_steps, self.batch_size // self.channels,
                                                            shape, cond, verbose=False, x_T=x_noisy, **kwargs)
            else:
                samples, intermediates = ddim_sampler.sample(ddim_steps, self.batch_size // self.channels,
                                                            shape, cond, verbose=False, **kwargs)
        else:
            samples, intermediates = self.sample(cond=cond, batch_size=self.batch_size // self.channels,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates
    
    @torch.no_grad()
    def log_images(self, batch, N=32, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc, a = self.get_input(batch, self.first_stage_key,
                                             return_first_stage_outputs=True,
                                             force_c_encode=True,
                                             return_original_cond=True,
                                             bs=N, return_angles=True)
        cf = super().get_input(batch, self.first_stage_model.cond_key).to(self.device) if self.is_conditional_first_stage else None
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = self._decode_logging(x).view(-1, self.channels, *self.infer_image_size)

        if sample:
            # get denoise row
            with self.ema_scope():
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta, batch=batch, angles=a.long())
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples, cf)
            log["samples"] = self._decode_logging(x_samples)
            if self.use_projected_noise:
                fbp = torch.zeros((360,) + tuple(self.projector.geometry.n_detector), device=self.device)
                fbp[a.long()] = log["samples"].squeeze().view(-1, *self.projector.geometry.n_detector)
                log["fbp"] = self.projector.backward(fbp, use_fbp=True)[None, None]

        return log


class ProjectionDiffusion2D(LatentDiffusion):
    def __init__(self,
                 batch_size=32,
                 label_key='',
                 projector_config=None,
                 normalizer_config=None,
                 use_projected_noise=True,
                 hlcc_n=180,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hlcc_n = hlcc_n
        self.label_key = label_key
        self.batch_size = batch_size
        self.use_projected_noise = use_projected_noise
        assert self.batch_size % self.channels == 0
        self.normalizer = instantiate_from_config(normalizer_config)
        if use_projected_noise:
            self.projector_config = OmegaConf.to_container(projector_config)
    
    def on_fit_start(self, **kwargs):
        if self.use_projected_noise:
            self.projector_config['params']['device'] = self.device
            self.projector_config['params']['batch_size'] = 1 if self.channels == 1 else self.batch_size // self.channels
            self.projector = instantiate_from_config(self.projector_config)
        self.buffer_idx = 0
        self.buffered_images = torch.zeros((100, 1, 256, 256), device=self.device)
        self.buffered_angles = torch.zeros((100, 1, 64), device=self.device)

    def enqueue(self, x, c):
        new_buffer_idx = x.shape[0] + self.buffer_idx
        if new_buffer_idx > self.buffered_images.shape[0]:
            new_buffer_idx = new_buffer_idx % self.buffered_images.shape[0]
            offset = self.buffered_images.shape[0] - self.buffer_idx
            x_split = torch.split(x, [offset, x.shape[0] - offset], dim=0)
            c_split = torch.split(c, [offset, x.shape[0] - offset], dim=0)
            self.buffered_images[self.buffer_idx:] = x_split[0]
            self.buffered_images[:new_buffer_idx] = x_split[1]
            self.buffered_angles[self.buffer_idx:] = c_split[0]
            self.buffered_angles[:new_buffer_idx] = c_split[1]
        else:
            self.buffered_images[self.buffer_idx:new_buffer_idx] = x
            self.buffered_angles[self.buffer_idx:new_buffer_idx] = c
        self.buffer_idx = new_buffer_idx

    def dequeue(self, size):
        assert size <= self.buffered_images.shape[0]
        return self.buffered_images[-size:], self.buffered_angles[-size:]

    def get_noise(self):
        noise_3d = noise_like((1 if self.channels == 1 else self.batch_size // self.channels,) +\
                               tuple(self.projector.geometry.n_voxel), self.device, repeat=False)
        noise2ct = lambda x: (x - x.min()) / (x.max() - x.min()) * 2 - 1
        noise_3d = noise2ct(noise_3d)
        noise_2d = self.projector.forward(noise_3d)
        noise_2d = noise2ct(noise_2d) * self.normalizer.oM
        return noise_2d

    def get_hlcc_coef(self, full_proj):
        pi_gt = torch.arange(0, 360, device=self.device) * torch.pi / 180
        t = torch.linspace(-1, 1, full_proj.shape[2], device=self.device).view(1, 1, -1) ** self.hlcc_n
        y_gt = (full_proj * t).sum([-1], keepdim=True).permute(1, 0, 2)  # hxmx1
        A = torch.stack([torch.ones_like(pi_gt)] + \
                        [torch.cos(_ * pi_gt) for _ in range(1, self.hlcc_n + 1)] + \
                        [torch.sin(_ * pi_gt) for _ in range(1, self.hlcc_n + 1)], dim=-1)[None].repeat_interleave(y_gt.shape[0], 0)  # hxmx2n
        AtA, Aty = A.transpose(1, 2) @ A, A.transpose(1, 2) @ y_gt
        coef = torch.linalg.solve(AtA, Aty)
        return coef

    def get_input(self, 
                  batch, 
                  k, 
                  return_first_stage_outputs=False, 
                  force_c_encode=False, 
                  cond_key=None, 
                  return_original_cond=False, 
                  bs=None,
                  return_angles=False,
                  return_hlcc_coefs=False):
        inputs = super().get_input(batch, k, return_first_stage_outputs, force_c_encode, cond_key, return_original_cond, bs)
        hlcc_coef = self.get_hlcc_coef(inputs[0].squeeze())
        # chn = torch.sin(torch.linspace(0, 2*torch.pi, 360)) + 1
        chn = torch.ones((360,))
        chn = torch.multinomial(chn, num_samples=self.batch_size, replacement=False)

        inputs = [i[:, chn].view(-1, 1, *i.shape[2:]) if i is not None else None for i in inputs]
        ret = inputs
        if return_angles: 
            angles = super(LatentDiffusion, self).get_input(batch, self.label_key)[:, chn]
            ret += [angles[0].long()]
        if return_hlcc_coefs:
            ret += [hlcc_coef]
        return ret
    
    def shared_step(self, batch, **kwargs):
        x, c, a = self.get_input(batch, self.first_stage_key, return_angles=True)
        loss = self(x, c, angles=a)
        return loss

    def p_losses(self, x_start, cond, t, noise=None, angles=None, hlcc_coef=None):
        if not self.use_projected_noise:
            noise = default(noise, lambda: torch.randn_like(x_start))
        else:
            noise_full = self.get_noise()
            noise = noise_full[:, angles.long()].transpose(0, 1)
        
        # x, c = self.dequeue(8)
        # indices = torch.randperm(x_start.shape[0], device=self.device)[:1]
        # self.enqueue(x_start[indices], cond[indices])
        # x_start, cond = torch.cat([x_start, x], dim=0), torch.cat([cond, c], dim=0)

        x_start = self.normalizer.forward(x_start.view(-1, self.channels, *self.infer_image_size))
        if cond is not None: cond = cond.view(-1, self.channels, *cond.shape[2:])
        angles = angles.view(-1, self.channels)
        if self.channels != 1:
            if self.use_projected_noise:
                noise_index = [[_ for _ in range(self.batch_size)], reduce(lambda x, y: x + y, [[_] * self.channels for _ in range(self.batch_size // self.channels)])]
                noise = noise[*noise_index].view(-1, self.channels, *self.infer_image_size)
            else:
                noise = noise.view(-1, self.channels, *self.infer_image_size)
            t = t[:self.batch_size // self.channels]
        else:
            t = t[0:1]

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output, latent = self.apply_model(x_noisy, t, cond, return_latents=True)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3] + [4,] if self.dims == 3 else [])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar.to(self.device)[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean([1, 2, 3] + [4,] if self.dims == 3 else [])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        # temperature = .1
        # batch_size = latent.shape[0]
        # pos_thresh = max(1, 60 * (1 - 10 * self.trainer.current_epoch / self.trainer.max_epochs))
        # feat = nn.functional.normalize(latent.reshape(batch_size, -1), dim=1)
        # sim_matrix = torch.matmul(feat, feat.T) / temperature - torch.eye(batch_size, device=feat.device) * 1e5
        # diff = (angles.unsqueeze(1) - angles.unsqueeze(0)).abs()
        # sim_gt = torch.minimum(diff, 360 - diff)
        # pos_mask = (sim_gt < pos_thresh) & (~torch.eye(batch_size, dtype=torch.bool, device=feat.device))

        # pos_exp = sim_matrix.exp() * pos_mask.float()
        # pos_sum = pos_exp.sum(dim=1)
        # all_sum = sim_matrix.exp().sum(dim=1)

        # loss_cont = (-torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))).mean()
        # loss += loss_cont
        # loss_dict.update({f'{prefix}/loss_cont': loss_cont})

        # x_start_hat = self.predict_start_from_noise(x_noisy, t, model_output)
        # pi = angles * torch.pi / 180
        # t = torch.linspace(-1, 1, x_start_hat.shape[2], device=self.device).view(1, 1, -1) ** self.hlcc_n
        # y = (x_start_hat * t).sum([-1], keepdim=True).permute(1, 0, 2)  # hxmx1
        # X = torch.stack([torch.ones_like(pi)] + \
        #                 [torch.cos(_ * pi) for _ in range(1, self.hlcc_n + 1)] + \
        #                 [torch.sin(_ * pi) for _ in range(1, self.hlcc_n + 1)], dim=-1)[None].repeat_interleave(y.shape[0], 0)  # hxmx2n
        # loss_hlcc = ((y - X @ hlcc_coef) ** 2).sqrt().mean()
        # loss_dict.update({f'{prefix}/loss_hlcc': loss_hlcc})
        # loss += loss_hlcc * 0.1

        return loss, loss_dict
    
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, angles, **kwargs):
        if cond is not None: cond = cond.view(-1, self.channels, *cond.shape[2:])
        if ddim:
            ddim_sampler = DDIMSampler(self, **self.ddim_config)
            shape = (self.channels, ) + tuple(self.infer_image_size)
            if self.use_projected_noise: 
                x_noisy = self.get_noise()[0:1, angles].transpose(0, 1).view(-1, self.channels, *self.infer_image_size)
                samples, intermediates = ddim_sampler.sample(ddim_steps, self.batch_size // self.channels,
                                                            shape, cond, verbose=False, x_T=x_noisy, **kwargs)
            else:
                # _, cond = self.parse_inputs(cond.clone(), cond)
                samples, intermediates = ddim_sampler.sample(ddim_steps, self.batch_size // self.channels,
                                                            shape, cond, verbose=False, **kwargs)
        else:
            samples, intermediates = self.sample(cond=cond, batch_size=self.batch_size // self.channels,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates
    
    @torch.no_grad()
    def log_images(self, batch, N=32, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc, a = self.get_input(batch, self.first_stage_key,
                                             return_first_stage_outputs=True,
                                             force_c_encode=True,
                                             return_original_cond=True,
                                             bs=N, return_angles=True)
        cf = super().get_input(batch, self.first_stage_model.cond_key).to(self.device) if self.is_conditional_first_stage else None
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = self._decode_logging(x).view(-1, self.channels, *self.infer_image_size)

        if sample:
            # get denoise row
            with self.ema_scope():
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta, batch=batch, angles=a.long())
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples, cf)
            log["samples"] = self._decode_logging(x_samples)
            if self.use_projected_noise:
                fbp = torch.zeros((360,) + tuple(self.projector.geometry.n_detector), device=self.device)
                fbp[a.long()] = log["samples"].squeeze().view(-1, *self.projector.geometry.n_detector)
                log["fbp"] = self.projector.backward(fbp, use_fbp=True)[None, None]

        return log


class ProjectionDiffusion3D(LatentDiffusion):
    def __init__(self, 
                 projector,
                 batch_size=12,
                 pm=0.5e5 / 3072, pM=3.5e5 / 3072,
                 nm=-1, nM=1, vm=0, vM=3072,
                 use_different_timesteps_per_batch=True,
                 **kw):
        super().__init__(**kw)
        self.batch_size = batch_size
        self.projector = OmegaConf.to_container(projector)

        self.pm, self.pM, self.nm, self.nM = pm, pM, nm, nM
        self.vm, self.vM = vm, vM
        self.use_different_timesteps_per_batch = use_different_timesteps_per_batch

    def normalize_volume(self, vol, backward=False):
        if not backward:
            return vol * 1 / self.vM
        else:
            return vol * self.vM

    def normalize_projection(self, proj, backward=False):
        if not backward:
            return (proj - self.pm) / (self.pM - self.pm) * (self.nM - self.nm) + self.nm
        else:
            return (proj - self.nm) / (self.nM - self.nm) * (self.pM - self.pm) + self.pm

    def normalize_noise(self, noise):
        return (noise - noise.min()) / (noise.max() - noise.min()) * (self.nM - self.nm) + self.nm

    def on_fit_start(self):
        self.projector['params']['device'] = self.device
        self.projector0 = instantiate_from_config(self.projector)
        self.projector1 = instantiate_from_config(self.projector)
        if self.use_different_timesteps_per_batch:
            self.projector['params']['batch_size'] = self.batch_size
        self.projector2 = instantiate_from_config(self.projector)
    
    def get_hlcc_loss(self, x, pi_trgt, model_outputs, n=180):
        pi_gt = torch.arange(0, 360, device=self.device) * torch.pi / 180
        pi = pi_trgt * torch.pi / 180
        t = torch.linspace(-1, 1, x.shape[2], device=self.device).view(1, 1, -1) ** n
        y_gt = (x * t).sum([-1], keepdim=True).permute(1, 0, 2)  # hxmx1
        y = (model_outputs * t).sum([-1], keepdim=True).permute(1, 0, 2)  # hxmx1
        A = torch.stack([torch.ones_like(pi_gt)] + \
                        [torch.cos(_ * pi_gt) for _ in range(1, n + 1)] + \
                        [torch.sin(_ * pi_gt) for _ in range(1, n + 1)], dim=-1)[None].repeat_interleave(y_gt.shape[0], 0)  # hxmx2n
        X = torch.stack([torch.ones_like(pi)] + \
                        [torch.cos(_ * pi) for _ in range(1, n + 1)] + \
                        [torch.sin(_ * pi) for _ in range(1, n + 1)], dim=-1)[None].repeat_interleave(y.shape[0], 0)  # hxmx2n
        AtA, Aty = A.transpose(1, 2) @ A, A.transpose(1, 2) @ y_gt
        coef = torch.linalg.solve(AtA, Aty)
        return ((y - X @ coef) ** 2).sqrt().mean()
    
    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, 
                          (self.batch_size,) if self.use_different_timesteps_per_batch else (x.shape[0],),
                          device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.to(self.dtype)))
        return self.p_losses(x, c, t, *args, **kwargs)
    
    def p_losses(self, v_start, cond, t, noise=None):
        v_start += v_start.min().abs()
        v_start = self.normalize_volume(v_start)
        noise = torch.randn_like(v_start)
        noise = self.normalize_noise(noise)
        # 2d
        noise = self.projector0.forward(noise.squeeze())
        x_start = self.projector1.forward(v_start.squeeze())

        noise = self.normalize_projection(noise)
        x_start = self.normalize_projection(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        chn = torch.ones((360,), device=self.device)
        chn = torch.multinomial(chn, num_samples=self.batch_size, replacement=False)

        cond = torch.arange(0, 360, device=self.device)
        cond = self.get_learned_conditioning(cond[chn]).transpose(0, 1)

        x_noisy = x_noisy[chn].unsqueeze(1)
        if not self.use_different_timesteps_per_batch:
            t = t.repeat_interleave(self.batch_size, dim=0)
        noise = noise[chn].unsqueeze(1)
        model_output, latent = self.apply_model(x_noisy, t, cond, return_latents=True)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3] + [4,] if self.dims == 3 else [])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar.to(self.device)[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean([1, 2, 3] + [4,] if self.dims == 3 else [])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})
        # contrastive loss
        # temperature = .1
        # batch_size = latent.shape[0]
        # pos_thresh = max(1, 120 * (1 - 10 * self.trainer.current_epoch / self.trainer.max_epochs))
        # feat = nn.functional.normalize(latent.reshape(batch_size, -1), dim=1)
        # sim_matrix = torch.matmul(feat, feat.T) / temperature - torch.eye(batch_size, device=feat.device) * 1e5
        # diff = (chn.unsqueeze(1) - chn.unsqueeze(0)).abs()
        # sim_gt = torch.minimum(diff, 360 - diff)
        # pos_mask = (sim_gt < pos_thresh) & (~torch.eye(batch_size, dtype=torch.bool, device=feat.device))

        # pos_exp = sim_matrix.exp() * pos_mask.float()
        # pos_sum = pos_exp.sum(dim=1)
        # all_sum = sim_matrix.exp().sum(dim=1)

        # loss_cont = (-torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))).mean()
        # loss += loss_cont
        # loss_dict.update({f'{prefix}/loss_cont': loss_cont})
        # hlcc loss
        # x_start_hat = self.predict_start_from_noise(x_noisy, t, model_output)
        # loss_hlcc = self.get_hlcc_loss(x_start, chn, x_start_hat.squeeze(), n=180)
        # loss_dict.update({f'{prefix}/loss_hlcc': loss_hlcc})
        # loss += loss_hlcc * 0.1

        return loss, loss_dict
    
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        infer_batch_size = 24
        shape = (self.channels, ) + tuple(self.infer_image_size)
        chn = torch.randperm(shape[1])[:infer_batch_size]
        cond = torch.arange(0, 360, device=self.device)
        cond = self.get_learned_conditioning(cond[chn]).transpose(0, 1)

        if ddim:
            ddim_sampler = DDIMSampler(self, **self.ddim_config)
            noise = torch.randn(shape)
            noise = self.normalize_noise(noise)

            x_noisy = self.projector1.forward(noise)
            x_noisy = self.normalize_projection(x_noisy)
            x_noisy = x_noisy[:, chn].transpose(0, 1)

            samples, intermediates = ddim_sampler.sample(ddim_steps, infer_batch_size,
                                                         shape, cond, verbose=False, x_T=x_noisy, **kwargs)
            samples = torch.flip(samples, [1])

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates
    
    # def log_images(self, batch, **kw):
    #     logs = super(ProjectionDiffusion3D, self).log_images(batch, **kw)
    #     logs['fbp'] = self.projector.backward(logs['samples'])
    #     return logs
    
    
class ConsistencyDiffusion(LatentDiffusion):
    def __init__(self, cons_type='point_decode', *args, **kwargs):
        self.cons_type = cons_type
        self.freeze_load = False
        super().__init__(*args, **kwargs)

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

        if self.freeze_load:
            target_param_iter = self.named_parameters() if not only_model else self.model.named_parameters()
            target_params = dict(target_param_iter)
            sd_keys = set(sd.keys())
            for name, param in target_params.items():
                # Only consider parameter names that appear in the checkpoint and weren't missing.
                if name in sd_keys and name not in missing + unexpected:
                    param.requires_grad = False

    def shared_step(self, batch, **kwargs):
        x = super(LatentDiffusion, self).get_input(batch, self.first_stage_key).transpose(0, 1)
        c = super(LatentDiffusion, self).get_input(batch, self.cond_stage_key).transpose(0, 1)
        p = super(LatentDiffusion, self).get_input(batch, 'points_2d')[0]
        t = torch.randint(self.num_timesteps // 2, self.num_timesteps, (x.shape[0],), device=self.device).long()
        c = self.get_learned_conditioning(c)
        # debug
        # volume_at_d = batch['volume'][:, batch['selected_coord'].long()[0]]
        # projection_at_d = self.minmax(self.projector.forward(volume_at_d)).squeeze()[batch['selected_angle_in_degrees'].long()[0]]
        # raw_at_d = self.minmax(torch.flip(batch['projection'][..., 255-batch['selected_coord'].long()[0], :], dims=[1])).squeeze()[batch['selected_angle_in_degrees'].long()[0]]
        # point_coords = torch.clip((p + 1) / 2 * 255, 0, 255).long()
        # projection_from_points_at_d = self.minmax(batch['projection'][..., 255-batch['selected_coord'].long()[0], :])

        loss, loss_dict = self.p_losses(x, c, t, 
                                        points_2d=p, 
                                        selected_coord=batch['selected_coord'].long()[0],
                                        selected_angles=batch['selected_angle_in_degrees'].long()[0])
        return loss, loss_dict
    
    def on_fit_start(self,):
        geometry = geometry0825_onerow
        self.projector = LeaptorchProjectorWrapper(geometry, self.device)
    
    def apply_model(self, x_noisy, t, cond, **kw):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        x_recon = self.model(x_noisy, t, **cond, **kw)
        return x_recon
    
    @staticmethod
    def minmax(p):
        return (p - p.min()) / (p.max() - p.min() + 1e-8)

    def p_losses(self, x_start, cond, t, noise=None, points_2d=None, selected_angles=None, selected_coord=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = torch.cat([x_noisy[:x_noisy.shape[0]//2], x_noisy[:x_noisy.shape[0]//2]], dim=0)

        # b1 = torch.randint(1, x_noisy.shape[0], (1,))
        # model_output_dict = self.apply_model(torch.cat([x_noisy[:b1], x_start[b1:]], dim=0),
        #                                     torch.cat([t[:b1], torch.zeros_like(t[b1:])], dim=0),
        #                                     cond,
        #                                     point_indices=points_2d,
        #                                     model_2d=True,)
        image_out = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(image_out, target, mean=False).mean([1, 2, 3] + [4,] if self.dims == 3 else [])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar.to(self.device)[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(image_out, target, mean=False).mean([1, 2, 3] + [4,] if self.dims == 3 else [])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        
        # point consistentcy loss
        # loss_point = self.get_loss(point_out, point_out_gt, mean=False).mean(1)
        # loss += loss_point.max() * 5
        # loss_dict.update({f'{prefix}/loss_point': loss_point.max()})
        
        # point recon loss
        # loss_recon = self.get_loss(self.minmax(plane_projected_points[selected_angles]),
        #                            self.minmax(torch.flip(x_start[..., selected_coord], dims=[-1])), mean=False).mean()
        # loss += loss_recon
        # loss_dict.update({f'{prefix}/loss_recon': loss_recon})
        loss_dict.update({f'{prefix}/loss': loss})
        
        return loss, loss_dict

    @torch.no_grad()
    def log_images(self, batch, N=32, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=False, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
                   plot_diffusion_rows=False, verbose=False, plot_conditioned_samples=False, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        x = super(LatentDiffusion, self).get_input(batch, self.first_stage_key)
        c = super(LatentDiffusion, self).get_input(batch, self.cond_stage_key)
        x = x.transpose(0, 1)
        c = c.transpose(0, 1)
        z = xrec = x
        xc = None
        c = self.get_learned_conditioning(c)
        
        cf = super().get_input(batch, self.first_stage_model.cond_key).to(self.device) if self.is_conditional_first_stage else None
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = self._decode_logging(x)
        log["reconstruction"] = self._decode_logging(xrec)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy, cf))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c ... -> b n c ...')
            if self.dims == 3:
                diffusion_grid = rearrange(diffusion_grid, 'b n c h w d -> (b n h) c w d')
            elif self.dims == 2: diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope():
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta, batch=batch)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples, cf)
            log["samples"] = self._decode_logging(x_samples)
            # if self.cons_type == 'point_decode':
            #     geometry = self.trainer.datamodule.datasets['train'].geometry
            #     # points_3d = np.mgrid[:geometry['n_voxel'][0], :geometry['n_voxel'][1], :geometry['n_voxel'][2]].reshape(3, -1).T
            #     # points_3d = points_3d.astype(np.float32) / (np.array([geometry['n_voxel']]) - 1)
            #     # _, points_2d = self.trainer.datamodule.datasets['train'].get_points(points_3d, batch['selected_angle_in_radians'][0, :samples.shape[0]].cpu().numpy())
            #     # points_2d = torch.tensor(points_2d, device=self.device, dtype=torch.float32)
            #     points_2d_hat = []
            #     D, H, W = geometry['n_voxel']
            #     d = np.random.randint(0, D, (6,)).tolist()
            #     for _d in d:
            #         y, z = np.meshgrid(np.linspace(0, 1, H, endpoint=False), np.linspace(0, 1, W, endpoint=False))
            #         points_3d = np.stack([
            #             np.full(H * W, _d / D),
            #             y.ravel(),
            #             z.ravel()
            #         ], axis=1)
            #         _, points_2d = self.trainer.datamodule.datasets['train'].get_points(points_3d, batch['selected_angle_in_radians'][0, :samples.shape[0]].cpu().numpy())
            #         model_output = self.apply_model(samples, torch.ones(samples.shape[0], device=self.device),
            #                                         cond=c[:samples.shape[0]], 
            #                                         point_indices=torch.tensor(points_2d, dtype=torch.float32, device=self.device),
            #                                         model_2d=True,)
            #         points_2d_hat.append(model_output['plane_out'])
            #     points_3d_hat = torch.cat(points_2d_hat, dim=2).view(-1, len(d), H, W)
            #     log['backward'] = points_3d_hat[:, None]

            #     point_2d_proj = []
            #     angles = batch['selected_angle_in_degrees'][0, :samples.shape[0]].long().cpu().numpy()
            #     for i in range(len(d)):
            #         point_2d_proj.append(self.projector.forward(points_3d_hat[0, i].reshape(-1, *self.projector.geometry.n_voxel) + points_3d_hat.min().abs()).clone())
            #     point_2d_proj = torch.stack([self.minmax(torch.cat(point_2d_proj, dim=0)[:, angles].permute(0, 2, 1, 3)),
            #                                  self.minmax(torch.flip(x[..., d], dims=[-1]).permute(3, 1, 0, 2))], dim=0)
            #     log['forward'] = point_2d_proj
            # if plot_denoise_rows:
            #     denoise_grid = self._get_denoise_row_from_list(z_denoise_row, cf)
            #     log["denoise_row"] = denoise_grid
            
            # if plot_conditioned_samples and self.model.conditioning_key == "concat":
            #     log["conditioned_samples"] = torch.cat([log["samples"], log["conditioning"]], dim=1)

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels,) + tuple(self.infer_image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, cf, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(p for p in self.model.parameters() if p.requires_grad)
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(p for p in self.cond_stage_model.parameters() if p.requires_grad)
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt


class R2GaussianDiffusion(LatentDiffusion):
    def __init__(self, geometry=None, **kw):
        super().__init__(**kw)
        self.geometry = geometry0825

    def get_input(self, batch):
        x = super(LatentDiffusion, self).get_input(batch, self.first_stage_key)[None].transpose(0, 1)
        c = super(LatentDiffusion, self).get_input(batch, self.cond_stage_key)[None].transpose(0, 1)
        c = self.get_learned_conditioning(c)
        cameras = self.make_camera_from_view(batch['selected_angle_in_degrees'].long())
        return x, c, cameras
    
    def shared_step(self, batch, **kwargs):
        x = super(LatentDiffusion, self).get_input(batch, self.first_stage_key)
        c = super(LatentDiffusion, self).get_input(batch, self.cond_stage_key)
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        c = self.get_learned_conditioning(c)
        cameras = self.make_camera_from_view(batch['selected_angle_in_degrees'].long())

        loss, loss_dict = self.p_losses(x, c, t, cameras=cameras)
        return loss, loss_dict

    def make_camera_from_view(self, pis):
        cams = []
        # angle mapping : 0,1 (leapct) -> -90,-91 (r2gaussian)
        _pis = -pis.to(torch.float32) * torch.pi / 180 - torch.pi / 2
        for i, angle in enumerate(_pis):
            c2w = angle2pose(self.geometry["distance_source_origin"], angle.item())
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            FovX = np.arctan2(self.geometry["s_detector"][1] / 2, self.geometry["distance_source_detector"]) * 2
            FovY = np.arctan2(self.geometry["s_detector"][0] / 2, self.geometry["distance_source_detector"]) * 2
            mode = 0 if 'parallel' in self.geometry["mode"] else 1

            cams.append(
                Camera(
                    colmap_id=i,
                    uid=i,
                    R=R,
                    T=T,
                    angle=angle,
                    FoVx=FovX,
                    FoVy=FovY,   # use this attr to store angle information
                    image=torch.zeros(self.geometry["n_detector"], device=self.device)[None],
                    image_name=f'any_{i}',
                    mode=mode,
                    scanner_cfg=self.geometry,
                    data_device=self.device
                )
            )
        return cams

    def query(
        self,
        xyz,
        density,
        scaling,
        rotation,
        center=None,
        nVoxel=None,
        sVoxel=None,
        scaling_modifier=1.0,
    ):
        nVoxel = nVoxel or self.geometry["n_voxel"]
        sVoxel = sVoxel or self.geometry["s_voxel"]
        center = center or self.geometry["offset_origin"]
        voxel_settings = GaussianVoxelizationSettings(
            scale_modifier=scaling_modifier,
            nVoxel_x=int(nVoxel[0]),
            nVoxel_y=int(nVoxel[1]),
            nVoxel_z=int(nVoxel[2]),
            sVoxel_x=float(sVoxel[0]),
            sVoxel_y=float(sVoxel[1]),
            sVoxel_z=float(sVoxel[2]),
            center_x=float(center[0]),
            center_y=float(center[1]),
            center_z=float(center[2]),
            prefiltered=False,
            debug=False,
        )
        voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

        means3D = xyz
        density = density
        scales = scaling
        rotations = rotation

        vol_pred, radii = voxelizer(
            means3D=means3D,
            opacities=density,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )

        return {
            "vol": vol_pred,
            "radii": radii,
        }

    def render(
        self,
        viewpoint_camera,
        xyz,
        density,
        scaling,
        rotation,
        scaling_modifier=1.0,
    ):
        screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device=xyz.device)
        screenspace_points.retain_grad()

        means3D = xyz
        means2D = screenspace_points
        density = density
        scales = scaling
        rotations = rotation

        renders = []
        radiis = []
        for camera in viewpoint_camera:
            mode = camera.mode
            if mode == 0:
                tanfovx = 1.0
                tanfovy = 1.0
            elif mode == 1:
                tanfovx = math.tan(camera.FovX * 0.5)
                tanfovy = math.tan(camera.FovY * 0.5)
            else:
                raise ValueError("Unsupported mode!")
            
            raster_settings = GaussianRasterizationSettings(
                image_height=int(camera.image_height),
                image_width=int(camera.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                scale_modifier=scaling_modifier,
                viewmatrix=camera.world_view_transform,
                projmatrix=camera.full_proj_transform,
                campos=camera.camera_center,
                prefiltered=False,
                mode=camera.mode,
                debug=False,
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            rendered_image, radii = rasterizer(
                means3D=means3D,
                means2D=means2D,
                opacities=density,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None,
            )
            renders.append(rendered_image)
            radiis.append(radii)

        renders = torch.stack(renders, dim=0)
        radiis = torch.stack(radiis, dim=0)
        return {
            "render": renders,
            "viewspace_points": screenspace_points,
            "visibility_filter": radiis > 0,
            "radii": radiis,
        }

    def apply_model(self, x_noisy, t, cond, cameras=None, return_gaussian=False, **kw):
        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        output_dict = self.model(x_noisy, t, **cond, **kw)
        image_out = output_dict['image_out']
        xyz = torch.clip(output_dict['xyz'], -1, 1).view(-1, 3)
        density = torch.nn.functional.softplus(output_dict['density']).view(-1, 1)
        scaling = torch.exp(output_dict['scaling']).view(-1, 3)
        rotation = torch.nn.functional.normalize(output_dict['rot'].view(-1, 4))
        
        render_pkgs = self.render(cameras, xyz=xyz, density=density, scaling=scaling, rotation=rotation)
        image_out_gaussian = render_pkgs['render'].view(x_noisy.shape)
        # image_out_gaussian = (image_out_gaussian - image_out_gaussian.min()) / ((image_out_gaussian.max() - image_out_gaussian.min()) / 3 + 1e-8) - 1.5
        
        if torch.isnan(image_out).any():
            print('nan loss')
            raise ValueError('nan loss!')
        if not return_gaussian: return image_out
        else: return {'image_out': image_out, 'image_out_gaussian': image_out_gaussian,
                       'xyz': xyz, 'density': density, 'scaling': scaling, 'rotation': rotation}

    def p_losses(self, x_start, cond, t, noise=None, cameras=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # gaussian = self.apply_model(x_noisy, t, cond)
        # gaussian = gaussian.mean(0)
        # image_out = self.render(cameras,
        #                         xyz=torch.clip(gaussian[:3], -1, 1).view(-1, 3),
        #                         density=torch.nn.functional.softplus(gaussian[3:4]).view(-1, 1), 
        #                         scaling=torch.exp(gaussian[4:7]).view(-1, 3),
        #                         rotation=torch.nn.functional.normalize(gaussian[7:11].view(-1, 4)), )

        # image_out = image_out['render'].view(x_start.shape)
        ret = self.apply_model(x_noisy, t, cond, cameras=cameras, return_gaussian=True)
        # self.query(xyz=torch.clip(gaussian[:3], -1, 1).view(-1, 3),
        #                          density=torch.nn.functional.softplus(gaussian[3:4]).view(-1, 1), 
        #                          scaling=torch.exp(gaussian[4:7]).view(-1, 3),
        #                          rotation=torch.nn.functional.normalize(gaussian[7:11].view(-1, 4)), )['vol']
        image_out = ret['image_out']
        image_out_gaussian = ret['image_out_gaussian']

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        # image_out = (image_out - image_out.min()) / (image_out.max() - image_out.min() + 1e-8)
        # target = (target - target.min()) / (target.max() - target.min() + 1e-8)

        loss_simple = self.get_loss(image_out, target, mean=False).mean([1, 2, 3] + [4,] if self.dims == 3 else [])
        loss_simple += self.get_loss(image_out_gaussian, target, mean=False).mean([1, 2, 3] + [4,] if self.dims == 3 else [])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar.to(self.device)[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(image_out, target, mean=False).mean([1, 2, 3] + [4,] if self.dims == 3 else [])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)

        loss_dict.update({f'{prefix}/loss': loss})
        
        return loss, loss_dict

    @torch.no_grad()
    def log_images(self, batch, N=32, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=False, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
                   plot_diffusion_rows=False, verbose=False, plot_conditioned_samples=False, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        x = super(LatentDiffusion, self).get_input(batch, self.first_stage_key)
        c = super(LatentDiffusion, self).get_input(batch, self.cond_stage_key)
        c = self.get_learned_conditioning(c)
        
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = self._decode_logging(x)

        if sample:
            noise = torch.randn_like(x)[:N]
            b = noise.shape[0]
            t = torch.randint(self.num_timesteps-1, self.num_timesteps, (b,), device=self.device).long()
            x_noisy = self.q_sample(x_start=x, t=t, noise=noise)

            cameras = self.make_camera_from_view(batch['selected_angle_in_degrees'].long())
            ret = self.apply_model(x_noisy, t, c, cameras=cameras, return_gaussian=True)
            image_out = ret['image_out']
            image_out_gaussian = ret['image_out_gaussian']
            vol_out = self.query(xyz=ret['xyz'],
                                 density=ret['density'], 
                                 scaling=ret['scaling'],
                                 rotation=ret['rotation'], )['vol']

            log["samples"] = image_out
            log["samples_gaussian"] = image_out_gaussian
            log["volumes"] = vol_out[None, None]

        return log