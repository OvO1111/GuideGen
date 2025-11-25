import sys, os
import random
sys.path.append('/home/dlr/code/diffusion/r2_gaussian/')

import torch as th
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from r2_gaussian.r2_gaussian.gaussian import (
    render, 
    query, 
    initialize_gaussian
)
from r2_gaussian.r2_gaussian.dataset.dataset_readers import (
    CameraInfo,
    angle2pose,
)
from r2_gaussian.r2_gaussian.dataset import SimpleScene, cameraList_from_camInfos
from r2_gaussian.r2_gaussian.gaussian.gaussian_model import GaussianModel
from ldm.data.make_dataset.settings import Geometry
from ldm.data.make_dataset.from_latents import ProjectionNorm
from ldm.modules.losses.functional import hlcc, ssim, tv_3d_loss

from omegaconf import OmegaConf
from inference.utils import image_logger
from ldm.util import instantiate_from_config
from ldm.modules.losses.lpips import LPIPS
from ldm.models.diffusion.ddpm import LatentDiffusion


class R2GaussianProjectorWrapper:
    def __init__(self, 
                 geometry, 
                 device, 
                 s_init=None,
                 o=None,
                 steps=100,
                 pi_ctxt=None,
                 pi_trgt=None,
                 normalization=None,
                 init_lrs=[2e-4, 1e-2, 5e-3, 1e-3]
                 ):
        self.device = device
        self.geometry = geometry
        self.scene = SimpleScene(geometry=self.geometry, vol_gt=s_init)
        self.gaussians = GaussianModel(device=self.device,)
        self.initialize_gaussian(s_init)
        self.scene.gaussians = self.gaussians
        self.normalization = normalization or ProjectionNorm()
        if len(init_lrs) != 4: init_lrs = [init_lrs[0]] * 4
        
        self.steps = steps
        self.progress = 0
        self.pi_ctxt = pi_ctxt
        self.pi_trgt = pi_trgt
        self.o = o + 2
        self.ctxt_cams = cameraList_from_camInfos(
            self._make_caminfo_from_pi(np.array(pi_ctxt), self.o),
            args=SimpleNamespace(data_device=self.device)
        )
        self.trgt_cams = cameraList_from_camInfos(
            self._make_caminfo_from_pi(np.array(pi_trgt), self.o),
            args=SimpleNamespace(data_device=self.device)
        )
        # init gaussian lr
        self.gaussians.training_setup(SimpleNamespace(
            position_lr_init=init_lrs[0], position_lr_final=0.1*init_lrs[0], position_lr_max_steps=steps,
            density_lr_init=init_lrs[1], density_lr_final=0.1*init_lrs[0], density_lr_max_steps=steps,
            scaling_lr_init=init_lrs[2], scaling_lr_final=0.1*init_lrs[0], scaling_lr_max_steps=steps,
            rotation_lr_init=init_lrs[3], rotation_lr_final=0.1*init_lrs[0], rotation_lr_max_steps=steps,
        ))
        # following r2gaussian conventions
        volume_to_world = max(self.scene.scanner_cfg["sVoxel"])
        self.pipeline_params = SimpleNamespace(
            compute_cov3D_python=False,
            debug=False
        )
        self.optimization_params = SimpleNamespace(
            densify_until_iter=round(0. * self.steps),
            densify_from_iter=round(0.2 * self.steps),
            densification_interval=round(0.05 * self.steps),
            densify_grad_threshold=5e-6, 
            density_min_threshold=1e-3,
            max_screen_size=None,
            max_num_gaussians=6e5,
            max_scale=None,
            densify_scale_threshold=0.01 * volume_to_world,
            bbox=self.scene.bbox,                                   # [[-1, -1, -1], [1, 1, 1]]
            tv_vol_size=32, lambda_mse=1, lambda_tv=.1, lambda_dssim=0.25, lambda_prep=1       # loss_related
        )

    @th.enable_grad()      
    def update(self, cams=None, obs=None, do_backward=False):
        cams = cams if cams is not None else self.ctxt_cams
        obs = obs if obs is not None else th.cat([cam.original_image for cam in cams], dim=0)
        render_pkgs = self.forward(cams, return_pkgs=True)
        obs_hat = th.cat([_['render'] for _ in render_pkgs], dim=0)
        # mse loss
        loss_mse = th.nn.functional.mse_loss(obs_hat, obs)
        # ssim loss
        loss_ssim = 1 - ssim(obs_hat[:, None], obs[:, None])
        # tv loss
        tv_vol_size = self.optimization_params.tv_vol_size
        tv_vol_nVoxel = th.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = th.tensor(self.scene.scanner_cfg["dVoxel"]) * tv_vol_nVoxel
        tv_vol_center = (self.optimization_params.bbox[0] + tv_vol_sVoxel / 2) + (
            self.optimization_params.bbox[1] - tv_vol_sVoxel - self.optimization_params.bbox[0]
        ) * th.rand(3)
        vol_pred = self.backward(
            tv_vol_center,
            tv_vol_nVoxel,
            tv_vol_sVoxel,
        )
        loss_tv = tv_3d_loss(vol_pred, reduction="mean")
        loss_dict = {"mse": loss_mse, "ssim": loss_ssim, "tv": loss_tv}
        weight_dict = {"mse": self.optimization_params.lambda_mse,
                       "ssim": self.optimization_params.lambda_dssim,
                       "tv": self.optimization_params.lambda_tv,}

        if not do_backward:
            return loss_dict, weight_dict, render_pkgs
        
        self.progress += 1
        loss = sum([v * weight_dict[k] for k, v in loss_dict.items()])
        loss.backward()

        self.update_gaussians(render_pkgs)
        self.gaussians.update_learning_rate(self.progress)
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)

        return loss_dict, weight_dict   
            
    def _make_caminfo_from_pi(self, pis=None, o=None):
        cams = []
        # angle mapping : 0,1 (leapct) -> -90,-91 (r2gaussian)
        _pis = -pis.astype(np.float32) * np.pi / 180 - np.pi / 2
        for i, angle in enumerate(_pis):
            c2w = angle2pose(self.geometry["distance_source_origin"], angle)
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            FovX = np.arctan2(self.geometry["s_detector"][1] / 2, self.geometry["distance_source_detector"]) * 2
            FovY = np.arctan2(self.geometry["s_detector"][0] / 2, self.geometry["distance_source_detector"]) * 2
            mode = 0 if 'parallel' in self.geometry["mode"] else 1

            cams.append(
                CameraInfo(
                    uid=i,
                    R=R,
                    T=T,
                    angle=angle,
                    FovX=FovX,
                    FovY=FovY,
                    image=o[i] * self.scene.scene_scale,
                    image_path=None,
                    image_name=int(pis[i]),      # use this attr to store angle information
                    width=self.geometry["n_detector"][1],
                    height=self.geometry["n_detector"][0],
                    mode=mode,
                    scanner_cfg=self.geometry
                )
            )
        return cams

    def initialize_gaussian(self, 
                            s_init=None, 
                            n_points=int(1e5), 
                            density_thresh=0.1,
                            density_rescale=0.1):
        if s_init is None:
            print(f"Initialize random point clouds.")
            _xyz = np.array(self.scene.scanner_cfg["offOrigin"])[None, ...] + np.array(
                self.scene.scanner_cfg["sVoxel"]
            )[None, ...] * (np.random.rand(n_points, 3) - 0.5)
            _density = np.random.rand(n_points,) * 1.0
        else:
            density_mask = s_init > density_thresh
            valid_indices = np.argwhere(density_mask)
            offOrigin = np.array(self.scene.scanner_cfg["offOrigin"])
            dVoxel = np.array(self.scene.scanner_cfg["dVoxel"])
            sVoxel = np.array(self.scene.scanner_cfg["sVoxel"])

            assert (
                valid_indices.shape[0] >= n_points
            ), "Valid voxels less than target number of sampling. Check threshold"

            sampled_indices = valid_indices[
                np.random.choice(len(valid_indices), n_points, replace=False)
            ]
            _xyz = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
            _density = s_init[
                sampled_indices[:, 0],
                sampled_indices[:, 1],
                sampled_indices[:, 2],
            ]
            _density = _density * density_rescale  # scale intensity to stable training
        self.gaussians.create_from_pcd(_xyz, _density[:, None], 1)

    def forward(self, cams=None, return_pkgs=False):
        # gauusian -> proj
        render_pkgs = []
        cams = cams or self.ctxt_cams
        for cam in cams:
            render_pkgs.append(render(cam, self.gaussians, self.pipeline_params))
            
        if not return_pkgs: y = th.cat([_['render'] for _ in render_pkgs], dim=0)
        else: y = render_pkgs
        return y
    
    @th.no_grad()
    def update_gaussians(self, render_pkgs):
        gaussians = self.gaussians
        # adaptive control
        with th.no_grad():
            for render_pkg in render_pkgs:
                _y, viewspace_point_tensor, visibility_filter, radii = list(render_pkg.values())
                gaussians.max_radii2D[visibility_filter] = th.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if self.progress < self.optimization_params.densify_until_iter:
                if (
                    self.progress > self.optimization_params.densify_from_iter
                    and self.progress % self.optimization_params.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        self.optimization_params.densify_grad_threshold,
                        self.optimization_params.density_min_threshold,
                        self.optimization_params.max_screen_size,
                        self.optimization_params.max_scale,
                        self.optimization_params.max_num_gaussians,
                        self.optimization_params.densify_scale_threshold,
                        self.optimization_params.bbox,
                    )
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )
    
    def backward(self, center=None, n_voxel=None, s_voxel=None):
        # gaussian -> ct
        if center is None: center = self.scene.scanner_cfg["offOrigin"]
        if n_voxel is None: n_voxel = self.scene.scanner_cfg["nVoxel"]
        if s_voxel is None: s_voxel = self.scene.scanner_cfg["sVoxel"]
        y = query(
            self.gaussians,
            center,
            n_voxel,
            s_voxel,
            self.pipeline_params
        )['vol'].permute(2, 1, 0).contiguous()

        return y



class R2GaussianV2(R2GaussianProjectorWrapper):
    def __init__(self, *args, ckpt_dir=None, suppl_device=th.device('cuda:1'), pixel_conf=None,view_conf=None, **kw):
        super().__init__(*args, **kw)

        self.optimization_params.density_min_threshold = 2e-2
        self.optimization_params.lambda_diff = .5
        self.optimization_params.lambda_tv = .0
        self.suppl_device = suppl_device
        # self.lpips = LPIPS().eval().to(self.suppl_device)
        self.raw_ctxt_cam_ptr = len(self.ctxt_cams)
        self.model: LatentDiffusion = self.initialize_diffusion(ckpt_dir).eval().to(self.suppl_device)
        # self.model.requires_grad_(False)
        
        from ldm.models.diffusion.sampling.helpers.corrector import AdamOptimizer
        self.opt1 = AdamOptimizer()
        self.opt2 = AdamOptimizer()
        # self.pixel_conf = pixel_conf.to(self.device) if pixel_conf is not None else th.ones_like(self.o)
        # self.view_conf = view_conf.to(self.device) if view_conf is not None else th.ones(len(self.pi_ctxt), device=self.device)
        # self.pixel_conf.requires_grad_(True)
        # self.view_conf.requires_grad_(True)

        self.consistent_mask = th.zeros((360, 1, 256, 256), device=self.suppl_device)  # 1 for consistent ones, 0 for not

    def initialize_diffusion(self, ckpt_dir):
        config = OmegaConf.merge(*[
            OmegaConf.load(os.path.join(ckpt_dir, 'configs', _)) 
            for _ in sorted(os.listdir((os.path.join(ckpt_dir, 'configs'))))
        ])
        model_conf = config['model']
        if 'test_target' in model_conf:
            test_model = model_conf['test_target']
        elif 'train_target' in model_conf:
            test_model = model_conf['train_target']
        else:
            test_model = model_conf['target']
        model_conf["params"]["ckpt_path"] = os.path.join(ckpt_dir, 'checkpoints', 'last.ckpt')
        return instantiate_from_config({"target": test_model, "params": model_conf["params"]})
    
    @th.no_grad()
    def p_sample(self, x, t, c, max_batch=64, noise=None):
        device = x.device
        x = x.to(self.suppl_device)
        if isinstance(t, int): t = th.tensor(t, device=x.device)
        elif isinstance(t, float): t = th.tensor(round(t * self.model.num_timesteps), device=x.device)
        t = max(th.ones_like(t), t)
        # v1
        # t = t.repeat(x.shape[0])
        # c = self.model.get_learned_conditioning(th.tensor(c, device=x.device)[:, None])
        # xt = self.model.q_sample(x, t)
        # eps = []
        # for _x in range(0, x.shape[0], max_batch):
        #     eps.append(self.model.apply_model(xt[_x:_x+max_batch], t[_x:_x+max_batch], c[_x:_x+max_batch]))
        # # eps = self.model.apply_model(xt, t, c)
        # eps = th.cat(eps, dim=0)
        # x0 = self.model.predict_start_from_noise(xt, t, eps)
        # return x0.to(device)
        # v2
        c = self.model.get_learned_conditioning(th.tensor(c, device=x.device)[:, None])
        xt = self.model.q_sample(x, t.repeat(x.shape[0]), noise=noise)
        x0 = []
        for _x in range(0, xt.shape[0], max_batch):
            x0.append(self.model.sample_log(c[_x:_x+max_batch], c[_x:_x+max_batch].shape[0], False, -1, x_T=xt[_x:_x+max_batch], timesteps=t.item(), verbose=False)[0])
        x0 = th.cat(x0, dim=0)
        return x0.to(device)

    @th.no_grad()
    def get_perceptual_loss(self, x, y, max_batch=32):
        x = self._norm(x)
        y = self._norm(y)
        loss = []
        for _x, _y in zip(x.split(max_batch, dim=0), y.split(max_batch, dim=0)):
            loss.append(self.lpips(_x, _y).view(_x.shape[0]))
        loss = th.cat(loss, dim=0).view(x.shape[0], -1).mean(-1)
        return loss

    def _norm(self, x):
        return (x - x.min()) / (x.max() - x.min())

    @th.enable_grad()      
    def update(self, cams=None, obs=None, do_backward=False, k=16):
        cams = cams if cams is not None else self.ctxt_cams
        obs = obs if obs is not None else th.cat([cam.original_image for cam in cams], dim=0)
        render_pkgs = self.forward(cams, return_pkgs=True)
        obs_hat = th.cat([_['render'] for _ in render_pkgs], dim=0)
        # mse loss
        loss_mse = (th.nn.functional.mse_loss(obs_hat, obs, reduction='none')).mean() # * self.pixel_conf[[cam.image_name for cam in cams]]).sum()
        # ssim loss
        loss_ssim = 1 - (ssim(obs_hat, obs, size_average=False)).mean() # * self.view_conf[[cam.image_name for cam in cams]]).sum()
        # tv loss
        tv_vol_size = self.optimization_params.tv_vol_size
        tv_vol_nVoxel = th.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = th.tensor(self.scene.scanner_cfg["dVoxel"]) * tv_vol_nVoxel
        tv_vol_center = (self.optimization_params.bbox[0] + tv_vol_sVoxel / 2) + (
            self.optimization_params.bbox[1] - tv_vol_sVoxel - self.optimization_params.bbox[0]
        ) * th.rand(3)
        vol_pred = self.backward(
            tv_vol_center,
            tv_vol_nVoxel,
            tv_vol_sVoxel,
        )
        loss_tv = tv_3d_loss(vol_pred, reduction="mean")
        # diffusion consistency loss
        # p = max(0.001, .9 - self.progress / self.steps)
        # p = .2
        # angle = random.sample(self.pi_trgt, k=k)
        # cams = [_ for _ in self.trgt_cams if _.image_name in angle]
        # x0_proj = self.forward(cams)[:, None] - 2
        # x0_t_proj = self.p_sample(x0_proj, p, th.tensor(angle, device=x0_proj.device))
        # loss_diff = self.get_perceptual_loss(x0_proj, x0_t_proj)
        # debug
        # angle2 = random.sample(self.pi_ctxt, k=min(len(self.pi_ctxt), k))
        # cams2 = [_ for _ in self.ctxt_cams if _.image_name in angle2]
        # x0_proj2 = th.cat([_.original_image for _ in cams2])[:, None] - 2
        # x0_t_proj2 = self.p_sample(x0_proj2, p, th.tensor(angle2, device=x0_proj2.device))
        # loss_diff2 = self.get_perceptual_loss(x0_proj2, x0_t_proj2)
        # print(loss_diff.item(), loss_diff2.item())
        # image_logger({'loss_diff': th.cat([x0_proj, x0_t_proj]).cpu(), 'loss_diff2': th.cat([x0_proj2, x0_t_proj2]).cpu()}, path='./image.png')
        # loss dict
        loss_dict = {"mse": loss_mse, "ssim": loss_ssim, "tv": loss_tv}#, "diff": loss_diff}
        weight_dict = {"mse": self.optimization_params.lambda_mse,
                       "ssim": self.optimization_params.lambda_dssim,
                       "tv": self.optimization_params.lambda_tv,}
                    #    "diff": self.optimization_params.lambda_diff}

        if not do_backward:
            return loss_dict, weight_dict, render_pkgs
        
        self.progress += 1
        loss = sum([v * weight_dict[k] for k, v in loss_dict.items()])
        loss.backward()

        self.update_gaussians(render_pkgs)
        self.gaussians.update_learning_rate(self.progress)
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)

        return loss_dict, weight_dict  

    def add_cam(self, k=4, p=.5):
        x0_proj = (self.forward(self.trgt_cams)[:, None] - 2).to(self.suppl_device)
        # compute confidence
        ## todo
        x0_t_proj = self.p_sample(x0_proj, p, self.pi_trgt)
        loss = th.nn.functional.mse_loss(x0_proj, x0_t_proj, reduction='none').view(x0_proj.shape[0], -1).mean(-1)
        loss += self.get_perceptual_loss(x0_proj, x0_t_proj)
        angle = random.choices(self.pi_trgt, k=k, weights=th.nn.functional.normalize(th.exp(loss*10), dim=-1).cpu().data.numpy())
        
        new_ctxt_cams = cameraList_from_camInfos(
            self._make_caminfo_from_pi(np.array(angle), x0_t_proj[angle, 0] + 2),
            args=SimpleNamespace(data_device=self.device)
        )
        self.ctxt_cams = self.ctxt_cams[:self.raw_ctxt_cam_ptr] + new_ctxt_cams
        
    def replace_cam(self, k=4, p=.5):
        x0_proj = (self.forward(self.trgt_cams)[:, None] - 2).to(self.suppl_device)
        # compute confidence
        ## todo
        x0_t_proj = self.p_sample(x0_proj, p, self.pi_trgt)
        loss = th.nn.functional.mse_loss(x0_proj, x0_t_proj, reduction='none').view(x0_proj.shape[0], -1).mean(-1)
        loss += self.get_perceptual_loss(x0_proj, x0_t_proj)
        # angle = random.choices(self.pi_trgt, k=k, weights=th.nn.functional.normalize(th.exp(loss*10), dim=-1).cpu().data.numpy())
        angle = loss.argsort()[-k:].cpu().data.numpy().tolist()
        
        new_ctxt_cams = cameraList_from_camInfos(
            self._make_caminfo_from_pi(np.array(angle), x0_t_proj[angle, 0] + 2),
            args=SimpleNamespace(data_device=self.device)
        )
        for i in range(len(self.ctxt_cams)):
            if self.ctxt_cams[i].image_name in angle:
                self.ctxt_cams[i] = new_ctxt_cams[angle.index(self.ctxt_cams[i].image_name)]

    def replace_cam_v2(self, k=4, p=30):
        with th.no_grad():
            x0_proj = self.forward(self.ctxt_cams)[:, None].to(self.suppl_device) - 2
        x0_proj_ctxt = th.cat([cam.original_image for cam in self.ctxt_cams], dim=0)[:, None].to(self.suppl_device) - 2
        # x0_proj_ctxt = x0_proj_ctxt.clone().requires_grad_(True)
        loss = th.nn.functional.mse_loss(x0_proj, x0_proj_ctxt, reduction='none')
        eps = (loss > loss.mean()).float()
        eps = th.nn.functional.max_pool3d(eps.transpose(0, 1)[None], kernel_size=3, stride=1, padding=1)[0].transpose(0, 1)
        eps[0] = 0
        m = 1 - eps
        # x0_proj_ctxt = self.opt1(x0_proj_ctxt, th.autograd.grad(loss.mean(), x0_proj_ctxt)[0], 1e-1)

        # self.consistent_mask = (loss < loss.mean()).long()
        noise = th.randn_like(x0_proj) * eps + x0_proj * m
        # x0_t_proj = self.p_sample(x0_proj, p, self.pi_trgt, noise=noise.to(self.suppl_device))
        # x0_t_proj = x0_t_proj * (1 - self.consistent_mask) + x0_proj * self.consistent_mask
        x0_t_proj_ctxt = self.p_sample(x0_proj, p, self.pi_trgt, noise=noise.to(self.suppl_device))
        x0_t_proj_ctxt = x0_proj_ctxt * m + x0_t_proj_ctxt * eps
        # x0_t_proj_ctxt = x0_t_proj_ctxt * (1 - self.consistent_mask) + x0_proj_ctxt * self.consistent_mask
        for cam in self.ctxt_cams:
            if cam.image_name > 0:
                cam.original_image = (x0_t_proj_ctxt[cam.image_name] + 2).to(self.device)

    def adaptive_conf(self, cams=None, obs=None):
        cams = cams if cams is not None else self.ctxt_cams
        obs = obs if obs is not None else th.cat([cam.original_image for cam in cams], dim=0)
        with th.no_grad():
            render_pkgs = self.forward(cams, return_pkgs=True)
        obs_hat = th.cat([_['render'] for _ in render_pkgs], dim=0)
        # mse loss
        self.view_conf = th.clip(self.view_conf, 1e-9, 1)
        self.pixel_conf = th.clip(self.pixel_conf, 1e-9, 1)
        loss_mse = (th.nn.functional.mse_loss(obs_hat, obs, reduction='none') * self.pixel_conf[[cam.image_name for cam in cams]]).sum()
        loss_ssim = 1 - (ssim(obs_hat, obs, size_average=False) * self.view_conf[[cam.image_name for cam in cams]]).sum()
        loss_uni = (th.nn.functional.mse_loss(self.pixel_conf.sum(), th.tensor(1., device=self.device)) +\
              th.nn.functional.mse_loss(self.view_conf.sum(), th.tensor(1., device=self.device))) * 10
        loss = self.optimization_params.lambda_mse * loss_mse + self.optimization_params.lambda_dssim * loss_ssim + loss_uni
        grad1 = th.autograd.grad(loss, self.pixel_conf, retain_graph=True)[0]
        grad2 = th.autograd.grad(loss, self.view_conf)[0]
        # self.pixel_conf = self.pixel_conf - grad1 * self.pixel_conf.mean()
        # self.view_conf = self.view_conf - grad2 * self.view_conf.mean()
        self.pixel_conf = self.pixel_conf_opt(self.pixel_conf, grad1, self.pixel_conf.mean())
        self.view_conf = self.view_conf_opt(self.view_conf, grad2, self.view_conf.mean())
        # mask = 1 - 0.9 * th.stack([loss_mse[i] > loss_mse[i].max() * 0.9 for i in range(loss_mse.shape[0])], dim=0)
        # self.pixel_conf[[cam.image_name for cam in cams]] = mask * self.pixel_conf[[cam.image_name for cam in cams]]


@th.enable_grad()
def debug_loop(
    projector: R2GaussianProjectorWrapper, 
    log_dir='/mnt/data_1/dlr/data/cache/r2gaussian',
    use_single_cam_per_iter=False,
    vol_data=None,
):
    cam_stack = []
    log_interval = 2000 if use_single_cam_per_iter else 200
    iterator = tqdm(range(projector.steps), desc="debug gaussian loop")
    
    import pyvista as pv
    import SimpleITK as sitk
    import tensorboardX as tbx
    from skimage.transform import resize
    from torchvision.utils import make_grid
    from inference.metric_cal import MetricLog

    metric_logger = MetricLog()
    writer = tbx.SummaryWriter(log_dir=os.path.join(log_dir, 'logs'),)
    plotter3d = pv.Plotter(off_screen=True, line_smoothing=True, window_size=[800, 800])
    if vol_data is not None:
        vol = sitk.GetArrayFromImage(sitk.ReadImage(vol_data))
        minmax = lambda x: (x - x.min()) / (x.max() - x.min())
        vol = resize(vol, projector.scene.scanner_cfg["nVoxel"][::-1], order=3)

    def _write_file_no_metadata(file, path):
        assert file.ndim == 3, f"file should have a dimension of 3"
        if not isinstance(file, np.ndarray): file = file.data.cpu().numpy()
        sitk.WriteImage(sitk.GetImageFromArray(file.astype(np.float32)), path)

    @th.no_grad()
    def _log(i):
        if log_dir is not None:
            _write_file_no_metadata(projector.forward(projector.trgt_cams), os.path.join(log_dir, f'r2gaussian_forward_ep{i}.nii.gz'))
            _write_file_no_metadata(projector.backward(), os.path.join(log_dir, f'r2gaussian_backward_ep{i}.nii.gz'))
            projector.gaussians.save_ply(os.path.join(log_dir, f"r2gaussian_gaussians_ep{i}.ply"))
    
    loss_record = []
    best_loss = float('inf')
    for i in iterator:
        kw  = {}
        if use_single_cam_per_iter:
            if not cam_stack:
                cam_stack = projector.ctxt_cams.copy()
            # cam = [cam_stack.pop(random.randint(0, len(cam_stack) - 1))]
            cam = [cam_stack[random.choice([0, 45, 90, 135])] if i < 500 else cam_stack[random.randint(0, len(cam_stack) - 1)]]
        else:
            cam = projector.ctxt_cams
        loss_dict, weight_dict = projector.update(cams=cam, do_backward=True)
        loss = sum([v * weight_dict[k] for k, v in loss_dict.items()])
        # if loss < 1e-2:
        #     projector.add_cam(k=12, p=max(.01, .5 - (projector.progress / projector.steps) * 2))
        best_loss = min(loss.item(), best_loss)
        projector.gaussians.update_learning_rate(i)
        writer.add_scalars('loss', loss_dict | {'total': loss.item()}, global_step=i)
        
        # if len(loss_record) > 50:
        #     loss_record.pop(0)
        # loss_record.append(loss.item())
        # check if loss pleataus
        # if i > 100 and len(loss_record) > 40 and max(loss_record) - min(loss_record) < 1e-2:
        if i % 1000 == 0 and i > 0:
            projector.replace_cam_v2()
            loss_record = []

        iterator.set_postfix(
            {k: f"{(v.item() * weight_dict[k]):.4f}" for k, v in loss_dict.items()} | kw
        )
        with th.no_grad():
            if i % log_interval == 0:
                o_hat = projector.forward()[:, None].cpu()
                o_gt = th.cat([cam.original_image for cam in projector.ctxt_cams], dim=0)[:, None].cpu()
                o_hat_trgt = projector.forward(cams=projector.trgt_cams)[:, None].cpu()
                o_gt_trgt = th.cat([cam.original_image for cam in projector.trgt_cams], dim=0)[:, None].cpu()
                metric_dict_ctxt = {
                    'psnr_ctxt': metric_logger.psnr(o_hat, o_gt),
                    'ssim_ctxt': metric_logger.ssim(o_hat, o_gt),
                    'rmse_ctxt': metric_logger.rmse(o_hat, o_gt),
                }
                metric_dict_trgt = {
                    'psnr_trgt': metric_logger.psnr(o_hat_trgt, o_gt_trgt),
                    'ssim_trgt': metric_logger.ssim(o_hat_trgt, o_gt_trgt),
                    'rmse_trgt': metric_logger.rmse(o_hat_trgt, o_gt_trgt)
                }
                writer.add_scalars('metrics/ctxt', metric_dict_ctxt, global_step=i)
                writer.add_scalars('metrics/trgt', metric_dict_trgt, global_step=i)
                if vol_data is not None:
                    s_gt = th.tensor(minmax(vol))[None, None]
                    s_hat = minmax(projector.backward().cpu())[None, None]
                    metric_dict_vol = {
                        'psnr_vol': metric_logger.psnr(s_hat, s_gt),
                        'ssim_vol': metric_logger.ssim(s_hat, s_gt),
                        'rmse_vol': metric_logger.rmse(s_hat, s_gt)
                    }
                    writer.add_scalars('metrics/vol', metric_dict_vol, global_step=i)
                trgt_views = random.sample(range(len(o_hat_trgt)), k=16)
                writer.add_image('image/forward_ctxt', make_grid(o_hat, nrow=8, normalize=True), global_step=i)
                writer.add_image('image/forward_ctxt_gt', make_grid(o_gt, nrow=8, normalize=True), global_step=i)
                writer.add_image('image/forward_trgt', make_grid(o_hat_trgt[trgt_views], nrow=8, normalize=True), global_step=i)
                writer.add_image('image/forward_trgt_gt', make_grid(o_gt_trgt[trgt_views], nrow=8, normalize=True), global_step=i)
                # plot 3d
                plotter3d.clear()
                plotter3d.add_volume(
                    projector.backward().cpu().data.numpy(), 
                    cmap='viridis', 
                    opacity="linear", 
                )
                screenshot = plotter3d.screenshot(transparent_background=False, return_img=True)
                writer.add_image('image/backward', np.array(screenshot).transpose(2, 0, 1), global_step=i)
            if i % log_interval == 0:
                _log(i)
                if loss == best_loss: _log('best')