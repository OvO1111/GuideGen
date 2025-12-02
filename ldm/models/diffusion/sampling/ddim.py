"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    instantiate_from_config
)
from ldm.models.diffusion.sampling.helpers.solver import DDIMStepSolver



class DDIMSampler(object):
    def __init__(self, model, 
                 solver=DDIMStepSolver,
                 schedule="uniform",
                 score_corrector=None,
                 score_corrector_config=None, **kwargs):
        super().__init__()
        self.model = model
        self.schedule = schedule
        self.ddpm_num_timesteps = model.num_timesteps
        
        if score_corrector_config is not None: 
            self.corrector = instantiate_from_config(score_corrector_config).to(model.device)
            # self.corrector.anchor = kwargs['batch']['anchor']
        elif score_corrector is not None:
            self.corrector = score_corrector
            # self.corrector.anchor = kwargs['batch']['anchor']
        else: self.corrector = None
        self.step_solver = partial(solver, **kwargs)

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.model.device:
                attr = attr.to(self.model.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True, preset=None):
        self.eta = ddim_eta
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=self.schedule, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose, ddim_timesteps_preset=preset)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        ddim_use_original_steps=False,
        img_callback=None,
        quantize_x0=False,
        eta=0.,
        mask=None,
        x0=None,
        temperature=1.,
        noise_dropout=0.,
        verbose=True,
        x_T=None,
        preset=None,
        log_every_t=100,
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
                conditioning = conditioning[:batch_size]

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose, preset=preset)
        # sampling
        # C, H, W = shape
        size = (batch_size,) + shape
        if verbose: print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=ddim_use_original_steps,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    verbose=verbose, **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(
        self, cond, shape,
        x_T=None, 
        ddim_use_original_steps=False,
        callback=None, 
        timesteps=None, 
        quantize_denoised=False,
        mask=None, 
        x0=None, 
        img_callback=None, 
        log_every_t=100,
        temperature=1., 
        noise_dropout=0., 
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None, 
        verbose=False,
        repeats=1,
        **kwargs
    ):
        dtype = getattr(cond, "dtype", torch.float32) if cond is not None else torch.float32
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device, dtype=dtype)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if verbose: print(f"Running DDIM Sampling with {total_steps} timesteps and {repeats} repeats per step")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        self.ddim_solver = self.step_solver(
            self,
            ddim_use_original_steps=ddim_use_original_steps, 
            quantize_denoised=quantize_denoised, 
            temperature=temperature, 
            noise_dropout=noise_dropout, 
            score_corrector=self.corrector, 
            unconditional_guidance_scale=unconditional_guidance_scale, 
            unconditional_conditioning=unconditional_conditioning,
            timesteps=list(reversed(range(0, timesteps))) if ddim_use_original_steps else np.flip(timesteps),
            iterator=iterator
        )
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            img, pred_x0 = self.ddim_solver.step(img, cond, repeats=repeats, **kwargs)

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates
    
    
DefaultDDIMSampler = DDIMSampler