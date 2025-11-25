import math
import torch
from functools import partial
from collections.abc import Iterable
from ldm.modules.diffusionmodules.util import noise_like


class DDIMStepSolver:
    def __init__(self, 
                 ddim_sampler,
                 ddim_use_original_steps=False, 
                 quantize_denoised=False, 
                 temperature=1., 
                 noise_dropout=0., 
                 score_corrector=None, 
                 unconditional_guidance_scale=1., 
                 unconditional_conditioning=None,
                 repeat_noise=False,
                 timesteps=None,
                 max_batch=32,
                 iterator=None,
                 **kw
                ):
        self.ddim_sampler = ddim_sampler
        self.use_original_steps = ddim_use_original_steps
        self.quantize_denoised = quantize_denoised
        self.temperature = temperature
        self.noise_dropout = noise_dropout
        self.score_corrector = score_corrector
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.unconditional_conditioning = unconditional_conditioning
        self.repeat_noise = repeat_noise
        
        # alias
        self.model = ddim_sampler.model
        self.ddim_alphas = ddim_sampler.ddim_alphas
        self.ddim_alphas_prev = ddim_sampler.ddim_alphas_prev
        self.ddim_sqrt_one_minus_alphas = ddim_sampler.ddim_sqrt_one_minus_alphas
        self.ddim_sigmas = ddim_sampler.ddim_sigmas
        self.ddim_sigmas_for_original_num_steps = ddim_sampler.ddim_sigmas_for_original_num_steps
        
        self.timesteps = timesteps
        self.step_counter = 0
        self.max_batch = max_batch
        self.iterator = iterator
        
    def parse_step(self, b, device):
        t = torch.full((b,), self.timesteps[self.step_counter], device=device, dtype=torch.long)
        index = len(self.timesteps) - self.step_counter - 1
        return t, index
    
    def apply_model(self, x, t, c, **kw):
        if x.shape[0] <= self.max_batch:
            return self.model.apply_model(x, t, c, **kw)
        else:
            res = []
            for idx in range(0, x.shape[0], self.max_batch):
                res.append(self.model.apply_model(*map(lambda d: d[idx: idx+self.max_batch] if d is not None else None, [x, t, c]), **kw)) 
            res = torch.cat(res, dim=0)
            # res2 = []
            # for idx in range(0, x.shape[0], self.max_batch):
            #     res2.append(self.model.apply_model(*map(lambda d: d[idx: idx+self.max_batch] if d is not None else None, [x, t, c]), return_latents=True, **kw)[-1]) 
            # res2 = torch.cat(res2, dim=0)
            return res
    
    def _retrieve_score(self, x, c, **kw):
        b, *_, device = *x.shape, x.device
        t, index = self.parse_step(b, device)
        if self.unconditional_conditioning is None or self.unconditional_guidance_scale == 1.:
            e_t = self.apply_model(x, t, c, **kw)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([self.unconditional_conditioning, c])
            e_t_uncond, e_t = self.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + self.unconditional_guidance_scale * (e_t - e_t_uncond)
        
        # if self.score_corrector is not None:
        #     assert self.model.parameterization == "eps"
        #     e_t = self.score_corrector.modify_score(ddim_sampler=self.ddim_sampler,
        #                                             score=e_t, 
        #                                             x=x, 
        #                                             t=t, 
        #                                             c=c, 
        #                                             index=index, **kw)
        return e_t
    
    def _retrieve_xprev(self, x, e_t, **kw):
        b, *_, device = *x.shape, x.device
        t, index = self.parse_step(b, device)
        alphas = self.model.alphas_cumprod if self.use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if self.use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if self.use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps * self.ddim_sampler.eta if self.use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1,) + (1,) * self.model.dims, alphas[index], device=device)
        a_prev = torch.full((b, 1,) + (1,) * self.model.dims, alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1,) + (1,) * self.model.dims, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1,) + (1,) * self.model.dims, sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization == 'eps':
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            eps = e_t
        elif self.model.parameterization == 'x0':
            pred_x0 = e_t
            eps = (x - a_t.sqrt() * pred_x0) / sqrt_one_minus_at
        if self.quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * eps
        noise = sigma_t * noise_like(x.shape, device, self.repeat_noise) * self.temperature
        if self.noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=self.noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        # score corrector
        # if self.score_corrector is not None:
        #     x_prev = self.score_corrector.modify_score(ddim_sampler=self.ddim_sampler,
        #                                                 ddim_solver=self,
        #                                                 x=x,
        #                                                 score=e_t,
        #                                                 index=index,
        #                                                 x_prev=x_prev, 
        #                                                 dir_xt=dir_xt,
        #                                                 noise=noise)
        return x_prev, pred_x0, dir_xt, noise

    def step(self, x, c, repeats=1, **kw):
        e_t = self._retrieve_score(x, c, **kw)
        x_prevs, pred_x0s, dir_xts, noises = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
        if self.score_corrector is not None:
            for i in range(repeats):
                x_prev, pred_x0, dir_xt, noise = self._retrieve_xprev(x, e_t, **kw)
                x_prevs += x_prev
                pred_x0s += pred_x0
                dir_xts += dir_xt
                noises += noise

            x_prev = x_prevs / repeats
            pred_x0 = pred_x0s / repeats
            dir_xt = dir_xts / repeats
            noise = noises / repeats
            t, index = self.parse_step(x.shape[0], x.device)
            with torch.enable_grad():
                x.requires_grad = True 
                x_prev = self.score_corrector.step(ddim_sampler=self.ddim_sampler,
                                                    ddim_solver=self,
                                                    x=x,
                                                    score=e_t,
                                                    index=index,
                                                    x_prev=x_prev,
                                                    dir_xt=dir_xt,
                                                    noise=noise)
        else:
            for i in range(repeats):
                x_prev, pred_x0, *_ = self._retrieve_xprev(x, e_t, **kw)
                x_prevs += x_prev
                pred_x0s += pred_x0
            x_prev = x_prevs / math.sqrt(repeats)
            pred_x0 = pred_x0s / math.sqrt(repeats)

        # with torch.enable_grad() if self.score_corrector is not None else torch.no_grad():
        #     if self.score_corrector is not None: x.requires_grad = True
        #     x_prev, pred_x0 = self._retrieve_xprev(x, e_t, **kw)
        x_prev = x_prev.detach()
        self.step_counter += 1
        return x_prev.type(x.dtype), pred_x0.type(x.dtype) 
    

class DDIMStepInverter:
    def __init__(self, 
                 ddim_sampler,
                 ddim_use_original_steps=False, 
                 temperature=1., 
                 noise_dropout=0., 
                 unconditional_guidance_scale=1., 
                 unconditional_conditioning=None,
                 repeat_noise=False,
                 timesteps=None,
                 max_batch=4,
                 iterator=None,
                 **kw
                ):
        self.ddim_sampler = ddim_sampler
        self.use_original_steps = ddim_use_original_steps
        self.temperature = temperature
        self.noise_dropout = noise_dropout
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.unconditional_conditioning = unconditional_conditioning
        self.repeat_noise = repeat_noise
        
        # alias
        self.model = ddim_sampler.model
        self.ddim_alphas = ddim_sampler.ddim_alphas
        self.ddim_alphas_prev = ddim_sampler.ddim_alphas_prev
        self.ddim_sqrt_one_minus_alphas = ddim_sampler.ddim_sqrt_one_minus_alphas
        self.ddim_sigmas = ddim_sampler.ddim_sigmas
        self.ddim_sigmas_for_original_num_steps = ddim_sampler.ddim_sigmas_for_original_num_steps
        
        self.timesteps = timesteps
        self.step_counter = 0
        self.max_batch = max_batch
        self.iterator = iterator
        
    def parse_step(self, b, device):
        t = torch.full((b,), self.timesteps[len(self.timesteps) - self.step_counter - 1] - 1, device=device, dtype=torch.long)
        index = self.step_counter
        return t, index
    
    def apply_model(self, x, t, c):
        if x.shape[0] <= self.max_batch:
            return self.model.apply_model(x, t, c)
        else:
            res = []
            for idx in range(0, x.shape[0], self.max_batch):
                res.append(self.model.apply_model(*map(lambda d: d[idx: idx+self.max_batch] if d is not None else None, [x, t, c]))) 
            return torch.cat(res, dim=0)
    
    def _retrieve_score(self, x, c, **kw):
        b, *_, device = *x.shape, x.device
        t, index = self.parse_step(b, device)
        if self.unconditional_conditioning is None or self.unconditional_guidance_scale == 1.:
            e_t = self.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([self.unconditional_conditioning, c])
            e_t_uncond, e_t = self.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + self.unconditional_guidance_scale * (e_t - e_t_uncond)
        
        return e_t
    
    def _retrieve_xpost(self, x, e_t, **kw):
        b, *_, device = *x.shape, x.device
        t, index = self.parse_step(b, device)
        alphas = self.model.alphas_cumprod if self.use_original_steps else self.ddim_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if self.use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_tp1 = torch.full((b, 1,) + (1,) * self.model.dims, alphas[index], device=device)
        a_t = torch.full((b, 1,) + (1,) * self.model.dims, alphas[index - 1] if index > 0 else 1., device=device)
        sigma_t = torch.full((b, 1,) + (1,) * self.model.dims, sigmas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - (1 - a_t).sqrt() * e_t) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - a_tp1 - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, self.repeat_noise) * self.temperature
        if self.noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=self.noise_dropout)
        x_post = a_tp1.sqrt() * pred_x0 + dir_xt + noise
        return x_post, pred_x0

    def step(self, x, c, **kw):
        e_t = self._retrieve_score(x, c, **kw)
        x_post, pred_x0 = self._retrieve_xpost(x, e_t, **kw)
        self.step_counter += 1
        return x_post.type(x.dtype), pred_x0.type(x.dtype) 