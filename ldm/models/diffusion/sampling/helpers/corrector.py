import gc
import torch
import random
import numpy as np
import torch.nn as nn
import pyvista as pv
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from typing import TYPE_CHECKING
from types import SimpleNamespace
from ldm.modules.diffusionmodules.util import zero_module
from ldm.modules.losses.functional import ssim, tv_3d_loss, hlcc
from ldm.modules.losses.lpips import LPIPS
from projectors.leap import LeaptorchProjectorWrapper
from projectors.noiser import TigreNoiser, AngleNoiser
if TYPE_CHECKING:
    from test import R2GaussianProjectorWrapper, LeaptorchProjectorWrapper
    from ldm.models.diffusion.sampling.ddim import DDIMStepSolver
    from ldm.models.diffusion.sampling.ddim import DDIMSampler
    from ldm.models.diffusion.ddpm import LatentDiffusion
    
    
def _minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)
    

class SGDOptimizer:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x, grad, scale):
        if torch.isnan(grad).sum() > 0:
            return x
        return x - scale * grad


class AdamOptimizer:
    def __init__(self, etas=(0.9, 0.999), varepsilon=1e-8) -> None:
        super().__init__()
        self.etas = etas
        self.m = None
        self.v = None
        self.varepsilon = varepsilon

    def __call__(self, x, grad, scale):
        if torch.isnan(grad).sum() > 0:
            return x
        if self.m is None:
            self.m = torch.zeros_like(grad).type_as(grad)
        else:
            self.m = (1 - self.etas[0]) * grad + self.etas[0] * self.m
        if self.v is None:
            self.v = torch.zeros_like(grad).type_as(grad)
        else:
            self.v = (1 - self.etas[1]) * grad**2 + self.etas[1] * self.v

        self.m_hat = self.m / (1 - self.etas[0])
        self.v_hat = self.v / (1 - self.etas[1])

        return x - scale * self.m_hat / (torch.sqrt(self.v_hat) + self.varepsilon)


class ScoreCorrector:
    def __init__(self, scale=1e-2, device=torch.device('cuda:0'),
                 logdir=None, log_image_interval=100, log_metric_interval=1):
        self.device = device
        self.scale = torch.ones(1, device=self.device) * scale
        self.logger = SummaryWriter(logdir=logdir)

        self.steps = 0
        self.log_image_interval = log_image_interval
        self.log_metric_interval = log_metric_interval

    @staticmethod
    def write_file_no_metadata(file, suffix=''):
        import SimpleITK as sitk
        assert file.ndim == 3, f"file should have a dimension of 3"
        if not isinstance(file, np.ndarray): file = file.data.cpu().numpy()
        sitk.WriteImage(sitk.GetImageFromArray(file.astype(np.float32)), f"/mnt/data_1/dlr/data/cache/temp{suffix}.nii.gz")
        return 1

    def move_to_device(self, data, device=None):
        device = device or self.device
        if data is None: return None
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, device=device)
        elif isinstance(data, (torch.Tensor, torch.nn.Module)):
            data = data.to(device)
        elif isinstance(data, list):
            data = [self.move_to_device(d) for d in data]
        elif isinstance(data, dict):
            data = {k: self.move_to_device(v) for k, v in data.items()}
        return data

    def modify_score(self,):
        raise NotImplementedError('implement this fn in child classes')

    def step(self, **kw):
        x_prev, metric_logs, image_logs = self.modify_score(**kw)
        if self.steps % self.log_image_interval == 0:
            for k, v in image_logs.items():
                self.logger.add_image(f'images/{k}', v, self.steps)
        if self.steps % self.log_metric_interval == 0:
            metric_logs = {k: v for k, v in metric_logs.items() if isinstance(v, (int, float))}
            self.logger.add_scalars('scalars', metric_logs, self.steps)
        self.steps += 1
        return x_prev


class ReconstructionCorrector(ScoreCorrector):
    def __init__(self,
                 logdir,
                 scale=1e-2,
                 opt='adam',
                 projector=None,
                 gpu_id=None,
                 pi_ctxt=None, pi_trgt=None,
                 o=None,
                 view_noise_eq_dose=1.,
                 angle_uncertainty=0.0,):
        super().__init__(scale=scale, device=gpu_id, logdir=logdir)
        
        if opt == 'adam': 
            self.opt = AdamOptimizer()
        elif opt == 'none':
            self.opt = SGDOptimizer()
        self.opt2 = AdamOptimizer()

        self.o = self.move_to_device(o)
        self.o_ctxt = TigreNoiser(self.o[pi_ctxt], dose=view_noise_eq_dose).noisy_projections
        if view_noise_eq_dose < 1:
            print('using noisy views, dose=', view_noise_eq_dose, 'difference norm:', (self.o_ctxt - self.o[pi_ctxt]).abs().mean())

        pi_ctxt_ = AngleNoiser(pi_ctxt, pi=angle_uncertainty).noisy_angles
        if angle_uncertainty > 0:
            print('using noisy angles, uncertainty=', angle_uncertainty, '\n old pi_ctxt:', pi_ctxt, '\n new pi_ctxt:', pi_ctxt_)
        self.pis = [pi_ctxt_, pi_trgt]

        self.projector = projector
        self.plotter3d = pv.Plotter(off_screen=True, line_smoothing=True, window_size=[800, 800])

    @torch.no_grad()
    def log_images(self, s, o_dict):
        logs = {}
        if self.steps % self.log_image_interval != 0: return logs

        for k, v in o_dict.items():
            logs[f'{k}'] = make_grid(v[:, None].cpu().data, nrow=8, normalize=True)
        
        if s is not None:
            self.plotter3d.clear()
            self.plotter3d.add_volume(
                s.cpu().data.numpy(), 
                cmap='viridis', 
                opacity="linear", 
            )
            screenshot = self.plotter3d.screenshot(transparent_background=False, return_img=True)
            logs['3d'] = screenshot.transpose(2, 0, 1)
        return logs


class ProjReconCorrector(ReconstructionCorrector):
    def __init__(self, 
                 logdir,
                 scale=1e-2,
                 opt='adam',
                 projector:"LeaptorchProjectorWrapper"=None,
                 o=None,
                 shape=None,
                 pi_ctxt=None, pi_trgt=None,
                 norm=None, gaussian=None,
                 gpu_id=torch.device('cuda:6'), **kw
                 ):
        super().__init__(
            logdir=logdir,
            scale=scale,
            opt=opt,
            projector=projector,
            gpu_id=gpu_id,
            pi_ctxt=pi_ctxt, 
            pi_trgt=pi_trgt,
            o=o, **kw
        )
        
        self.norm = norm
        self.shape = shape
        self.prg_cnt = [0]
        self.projector = projector
        self.gaussian = gaussian
        self.offset = torch.zeros_like(self.o).requires_grad_(True)
        self.opt2 = AdamOptimizer()
        
        self.weight = torch.eye(len(self.pis[1]), device=self.device, ).requires_grad_(True)

    @torch.enable_grad()      
    def update(self, cams=None, obs=None):
        self.gaussian.progress += 1
        loss_list = []

        for _ in range(1000):
            cams = cams if cams is not None else self.ctxt_cams
            obs = obs if obs is not None else torch.cat([cam.original_image for cam in cams], dim=0)
            render_pkgs = self.gaussian.forward(cams, return_pkgs=True)
            obs_hat = torch.cat([_['render'] for _ in render_pkgs], dim=0)
            # mse loss
            loss_mse = torch.nn.functional.mse_loss(obs_hat, obs)
            # ssim loss
            loss_ssim = 1 - ssim(obs_hat[:, None], obs[:, None])
            loss_dict = {"mse": loss_mse, "ssim": loss_ssim,}
            weight_dict = {"mse": self.gaussian.optimization_params.lambda_mse,
                        "ssim": self.gaussian.optimization_params.lambda_dssim,}
            
            loss = sum([v * weight_dict[k] for k, v in loss_dict.items()])
            loss.backward()

            if len(loss_list) > 100:
                loss_list.pop(0)
                if max(loss_list) - min(loss_list) < 1e-2:
                    break
            loss_list.append(loss.item())

            self.gaussian.update_gaussians(render_pkgs)
            self.gaussian.gaussians.update_learning_rate(self.gaussian.progress)
            self.gaussian.gaussians.optimizer.step()
            self.gaussian.gaussians.optimizer.zero_grad(set_to_none=True)

        return loss_dict, weight_dict  
        
    def _get_loss(self, pred_x0, o_hat):
        o_hat_ctxt = o_hat[self.pis[0]]
        # ce loss
        loss_ce = torch.nn.functional.mse_loss(o_hat_ctxt, self.norm.forward(self.o_ctxt))
        loss_cons = torch.nn.functional.mse_loss(pred_x0, o_hat.squeeze(1))
        # loss_ssim = 1 - ssim(o_hat_ctxt, self.norm.forward(self.o_ctxt))
        loss_tv = tv_3d_loss(pred_x0, reduction="mean")
        loss = loss_ce + loss_cons + 1e-2 * loss_tv
        return loss

    def modify_score(self, 
                     ddim_sampler: "DDIMSampler", 
                     ddim_solver: "DDIMStepSolver",
                     score, 
                     x, 
                     index=None, 
                     x_prev=None, 
                     dir_xt=None,
                     noise=None,
                     **kw):
        b, *_, device = *x.shape, x.device
        use_original_steps = ddim_sampler.ddim_solver.use_original_steps
        x, score, x_prev = map(self.move_to_device, [x, score, x_prev])
        x = x.requires_grad_(True)
        progress = 1 - index / len(ddim_solver.timesteps)
        
        alphas = ddim_sampler.model.alphas_cumprod if use_original_steps else ddim_sampler.ddim_alphas
        alphas_prev = ddim_sampler.model.alphas_cumprod_prev if use_original_steps else ddim_sampler.ddim_alphas_prev
        sigmas = ddim_sampler.model.ddim_sigmas_for_original_num_steps if use_original_steps else ddim_sampler.ddim_sigmas
        a_t = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, alphas[index], device=self.device)
        a_prev = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, alphas_prev[index], device=self.device)
        sqrt_one_minus_alphas = ddim_sampler.model.sqrt_one_minus_alphas_cumprod if use_original_steps else ddim_sampler.ddim_sqrt_one_minus_alphas
        sqrt_one_minus_at = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, sqrt_one_minus_alphas[index], device=self.device)
        sigma_t = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, sigmas[index], device=self.device)

        score = score.requires_grad_(True)
        # score_hat = self.NAAtNtx(score.squeeze(1)).unsqueeze(1)
        # loss1 = torch.nn.functional.mse_loss(score_hat, score)
        # grad = torch.autograd.grad(loss1, score)[0]
        # score = self.opt(score, grad, self.scale)
        # score = score_hat
        pred_x0 = (x - sqrt_one_minus_at * score) / a_t.sqrt()
        # pred_x0 = self.norm.backward(pred_x0).squeeze(1) #+ self.offset.unsqueeze(1)
        i_pi_ctxt_in_pi_trgt = [min(self.pis[1], key=lambda i: abs(i - round(_))) for _ in self.pis[0]]
        # pred_x0 = (pred_x0 + torch.roll(pred_x0, shifts=1, dims=0) + torch.roll(pred_x0, shifts=-1, dims=0)) / 3
        # pred_x0 = (self.weight.T @ pred_x0.view(pred_x0.shape[0], -1)).view(*pred_x0.shape)
        # x_prev = a_prev.sqrt() * pred_x0 + (1. - a_prev - sigma_t ** 2).sqrt() * score + self.move_to_device(noise)

        # temp = torch.zeros((360,) + pred_x0.shape[-2:], device=self.device, dtype=torch.float32)
        # # pis = random.sample(self.pis[1], k=96)
        # # temp = torch.nn.functional.interpolate(self.norm.backward(pred_x0[pis]).squeeze(1)[None, None], self.o.shape, mode='trilinear')[0, 0]
        # temp[self.pis[1]] = self.norm.backward(pred_x0).squeeze(1)
        # pred_x0[i_pi_ctxt_in_pi_trgt] = self.o_ctxt
        pred_x0 = self.norm.backward(pred_x0).squeeze(1)
        
        # cam_stack = np.random.choice(self.gaussian.trgt_cams, size=2, replace=False,).tolist()
        # o_stack = (pred_x0 + 2)[[cam_stack[_].image_name for _ in range(len(cam_stack))]]
        # self.update(cams=cam_stack, obs=o_stack.detach(),)
        # o_hat = self.gaussian.forward(self.gaussian.trgt_cams).detach() - 2
        # o_hat_ctxt = o_hat[i_pi_ctxt_in_pi_trgt]

        s = self.projector.Atx_full(pred_x0)
        o_hat = self.projector.Ax_full(s)
        o_hat = self.norm.forward(o_hat)
        o_hat_ctxt = o_hat[i_pi_ctxt_in_pi_trgt]

        pred_x0 = self.norm.forward(pred_x0)

        loss = torch.nn.functional.mse_loss(o_hat_ctxt, self.norm.forward(self.o_ctxt)) +\
            torch.nn.functional.mse_loss(pred_x0, o_hat)
        x_prev = self.opt(x_prev, torch.autograd.grad(loss, x, retain_graph=True)[0], self.scale)
        # x_prev = a_prev.sqrt() * o_hat + (1. - a_prev - sigma_t ** 2).sqrt() * score + self.move_to_device(noise)


        logs = {}
        logs['o'] = torch.nn.functional.mse_loss(_minmax(o_hat[[round(_) for _ in self.pis[1] if _ in list(range(360))]]), 
                                                 _minmax(self.o)[[round(_) for _ in self.pis[1] if _ in list(range(360))]]).item()
        logs['oc'] = torch.nn.functional.mse_loss(_minmax(o_hat_ctxt), _minmax(self.o_ctxt)).item()
        logs['c'] = torch.nn.functional.mse_loss(pred_x0, o_hat).item()
        logs['l'] = loss.item()
        # if progress > .1 and loss_c.std() > 2.5e-3 and progress - self.prg_cnt[-1] > .1:
        #     self.prg_cnt.append(progress)
        #     self._modify_conditional(_add=[loss_c.argmax(0).item()])
        # logs['x'] = f'{self.pis[0]}' + f"{loss_c.std().item():.4f}"
        ddim_solver.iterator.set_postfix(logs)

        imgs = {}
        # angles = torch.randperm(o_hat.shape[0])[:16]
        # imgs['o'] = o_hat[angles].cpu().data.squeeze()
        # imgs['t'] = pred_x0[angles].cpu().data.squeeze()
        # imgs['o_gt'] = self.o[angles].cpu().data.squeeze()
        
        if round(progress * 1000) % 100 == 0:
            import SimpleITK as sitk
            sitk.WriteImage(sitk.GetImageFromArray(pred_x0.squeeze().cpu().data.numpy()), f"/home/dlr/data/cache/o_hat_leapct_{round(progress*1000)}.nii.gz")
        
        x_prev = self.move_to_device(x_prev, device)
        return x_prev, logs, self.log_images(o_hat.squeeze(), imgs)
    

class HelgasonLudwigCorrector(ScoreCorrector):
    def __init__(self, 
                 opt='adam',
                 orders=1,
                 o=None,
                 pi_ctxt=None,
                 pi_trgt=None,
                 scale=1e-2,
                 gpu_id=torch.device('cuda:6')
                 ):
        super().__init__(scale=scale, device=gpu_id)
        self.orders = orders
        self.pi_ctxt = torch.tensor(pi_ctxt, device=self.device, dtype=torch.long)
        self.pi_trgt = torch.tensor(pi_trgt, device=self.device, dtype=torch.long)
        self.o_ctxt = o[self.pi_ctxt]
        if isinstance(self.orders, int):
            self.orders = [self.orders]
        if self.pi_trgt is None:
            self.pi_trgt = torch.linspace(0, 2*torch.pi, 360, device=self.device)
        if opt == 'adam':
            self.opt = AdamOptimizer()
        elif opt == 'none':
            self.opt = SGDOptimizer()

    def deg_rad_trans(self, deg=None, rad=None):
        if deg is not None:
            return deg * torch.pi / 180
        elif rad is not None:
            return rad * 180 / torch.pi
        else:
            raise ValueError("Either 'deg' or 'rad' must be provided.")

    def zeroth_order(self, x: torch.Tensor):
        # compute the helgason-Ludwig 0th consistency
        x = x.sum(dim=[-1, -2])
        loss = x.var(-1).sqrt()
        return {'zeroth_order_loss': loss.mean()}
    
    def nth_order(self, x: torch.Tensor, n=1):
        # compute the helgason-Ludwig nth consistency
        # t = torch.linspace(-1, 1, x.shape[2], device=x.device).view(1, 1, -1) ** n
        # m = (x * t).sum([-1, -2])
        # m_gt = torch.nn.functional.interpolate((self.o_ctxt * t).sum([-1, -2])[None, None], m.shape, mode='linear')[0, 0]
        # x1 = torch.stack([torch.ones_like(self.deg_rad_trans(deg=self.pi_trgt))] + \
        #                 [torch.cos(_ * self.deg_rad_trans(deg=self.pi_trgt)) for _ in range(1, n + 1)] + \
        #                 [torch.sin(_ * self.deg_rad_trans(deg=self.pi_trgt)) for _ in range(1, n + 1)], dim=-1)
        # lhs1, rhs1 = x1.transpose(0, 1) @ x1, x1.transpose(0, 1) @ m
        # coef1 = torch.linalg.solve(lhs1, rhs1)
        # loss = ((m_gt - x1 @ coef1) ** 2).sqrt().mean()
        
        # mathematically better but perform worse
        # m_gt = (self.o_ctxt * t).sum([-1, -2])
        # m_hat_gt = (x[:self.o_ctxt.shape[0]] * t).sum([-1, -2])
        # x1 = torch.stack([torch.ones_like(self.deg_rad_trans(deg=self.pi_ctxt))] + \
        #                 [torch.cos(_ * self.deg_rad_trans(deg=self.pi_ctxt)) for _ in range(1, n + 1)] + \
        #                 [torch.sin(_ * self.deg_rad_trans(deg=self.pi_ctxt)) for _ in range(1, n + 1)], dim=-1)
        # lhs1, rhs1 = x1.transpose(0, 1) @ x1, x1.transpose(0, 1) @ m_hat_gt
        # coef1 = torch.linalg.solve(lhs1, rhs1)
        # loss = ((m_gt - x1 @ coef1) ** 2).sqrt().mean()
        
        pi = torch.tensor(self.pi_trgt, device=x.device) * torch.pi / 180
        t = torch.linspace(-1, 1, x.shape[2], device=x.device).view(1, 1, -1) ** n
        y = (x * t).sum([-1], keepdim=True).permute(1, 0, 2)  # hxmx1
        A = torch.stack([torch.ones_like(pi)] + \
                        [torch.cos(_ * pi) for _ in range(1, n + 1)] + \
                        [torch.sin(_ * pi) for _ in range(1, n + 1)], dim=-1)[None].repeat_interleave(y.shape[0], 0)  # hxmx2n
        AtA, Aty = A.transpose(1, 2) @ A, A.transpose(1, 2) @ y
        coef2 = torch.linalg.solve(AtA, Aty)
        loss = ((y - A @ coef2) ** 2).sqrt().mean()
        return loss

    def modify_score(self,
                     ddim_sampler: "DDIMSampler",
                     ddim_solver: "DDIMStepSolver", 
                     score, 
                     x, 
                     index=None, 
                     x_prev=None, 
                     **kw):
        b, *_, device = *x.shape, x.device
        use_original_steps = ddim_sampler.ddim_solver.use_original_steps
        x, score, x_prev = map(self.move_to_device, [x, score, x_prev])
        
        alphas = ddim_sampler.model.alphas_cumprod if use_original_steps else ddim_sampler.ddim_alphas
        a_t = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, alphas[index], device=device)
        sqrt_one_minus_alphas = ddim_sampler.model.sqrt_one_minus_alphas_cumprod if use_original_steps else ddim_sampler.ddim_sqrt_one_minus_alphas
        sqrt_one_minus_at = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, sqrt_one_minus_alphas[index], device=device)

        pred_x0 = (x - sqrt_one_minus_at * score) / a_t.sqrt()

        loss_dict = {}
        pred_x0 = pred_x0.squeeze()
        for order in self.orders:
            loss_dict[f'{order}th'] = self.nth_order(pred_x0, n=order)
        loss = sum(loss_dict.values())

        ddim_solver.iterator.set_postfix({k: v.item() for k, v in loss_dict.items()} |\
                                         {'total_loss': loss.item()})
        grad = torch.autograd.grad(loss, x)[0]
        x_prev = self.opt(x_prev, grad, self.scale)

        return x_prev
    
    
class PcdReconCorrector(ReconstructionCorrector):
    def __init__(self,
                 logdir,
                 scale=1e-2,
                 opt='adam',
                 projector:"R2GaussianProjectorWrapper"=None,
                 o=None,
                 pi_ctxt=None, pi_trgt=None,
                 n_cam_ctxt=1, n_cam_trgt=6,
                 normalization=None,
                 gpu_id=torch.device('cuda:6'),):

        super().__init__(
            scale=scale,
            opt=opt,
            projector=projector,
            gpu_id=gpu_id,
            pi_ctxt=pi_ctxt, 
            pi_trgt=pi_trgt,
            o=o,
            logdir=logdir
        )
        self.shape = o.shape
        self.n_cam_ctxt = n_cam_ctxt
        self.n_cam_trgt = n_cam_trgt
        self.opt2 = AdamOptimizer()

        self.norm = normalization
        self.i = 0
        self.loss = torch.nn.MSELoss()
        self.lpips = LPIPS().eval().to(self.device)
        self.leap_projector = LeaptorchProjectorWrapper(self.geometry, self.device, batch_size=1)
    
    @torch.no_grad()
    def p_sample(self, model:"LatentDiffusion", x, t, c):
        if isinstance(t, int): t = torch.tensor(t, device=x.device)
        elif isinstance(t, float): t = torch.tensor(round(t * model.num_timesteps), device=x.device)
        # v1
        t = t.repeat(x.shape[0])
        c = model.get_learned_conditioning(torch.tensor(c, device=x.device)[:, None])
        xt = model.q_sample(x, t)
        eps = model.apply_model(xt, t, c)
        x0 = model.predict_start_from_noise(xt, t, eps)
        return x0.to(self.device)

    def _norm(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def get_perceptual_loss(self, x, y):
        x = self._norm(x)
        y = self._norm(y)
        return self.lpips(x, y).mean()

    @torch.enable_grad()      
    def update(self, cams=None, obs=None):
        cams = cams if cams is not None else self.ctxt_cams
        obs = obs if obs is not None else torch.cat([cam.original_image for cam in cams], dim=0)
        render_pkgs = self.forward(cams, return_pkgs=True)
        obs_hat = torch.cat([_['render'] for _ in render_pkgs], dim=0)
        # mse loss
        loss_mse = torch.nn.functional.mse_loss(obs_hat, obs)
        # ssim loss
        loss_ssim = 1 - ssim(obs_hat[:, None], obs[:, None])
        # tv loss
        tv_vol_size = self.projector.optimization_params.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(self.projector.scene.scanner_cfg["dVoxel"]) * tv_vol_nVoxel
        tv_vol_center = (self.projector.optimization_params.bbox[0] + tv_vol_sVoxel / 2) + (
            self.projector.optimization_params.bbox[1] - tv_vol_sVoxel - self.projector.optimization_params.bbox[0]
        ) * torch.rand(3)
        vol_pred = self.backward(
            tv_vol_center,
            tv_vol_nVoxel,
            tv_vol_sVoxel,
        )
        loss_tv = tv_3d_loss(vol_pred, reduction="mean")
        loss_dict = {"mse": loss_mse, "ssim": loss_ssim, "tv": loss_tv}
        weight_dict = {"mse": self.projector.optimization_params.lambda_mse,
                       "ssim": self.projector.optimization_params.lambda_dssim,
                       "tv": self.projector.optimization_params.lambda_tv,}
        
        self.progress += 1
        loss = sum([v * weight_dict[k] for k, v in loss_dict.items()])
        loss.backward()

        self.projector.update_gaussians(render_pkgs)
        self.projector.gaussians.update_learning_rate(self.projector.progress)
        self.projector.gaussians.optimizer.step()
        self.projector.gaussians.optimizer.zero_grad(set_to_none=True)

        return loss_dict, weight_dict   
        
    def modify_score(self, 
                     ddim_sampler: "DDIMSampler", 
                     ddim_solver: "DDIMStepSolver",
                     score, 
                     x, 
                     index=None, 
                     x_prev=None, 
                     dir_xt=None,
                     noise=None,
                     **kw):
        # index: max -> 0
        b, *_, device = *x.shape, x.device
        use_original_steps = ddim_sampler.ddim_solver.use_original_steps
        x, score, x_prev = map(self.move_to_device, [x, score, x_prev])
        x = x.detach_().requires_grad_(True)
        progress = 1 - index / len(ddim_solver.timesteps)
        
        alphas = ddim_sampler.model.alphas_cumprod if use_original_steps else ddim_sampler.ddim_alphas
        alphas_prev = ddim_sampler.model.alphas_cumprod_prev if use_original_steps else ddim_sampler.ddim_alphas_prev
        a_t = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, alphas[index], device=self.device)
        sigmas = ddim_sampler.model.ddim_sigmas_for_original_num_steps if use_original_steps else ddim_sampler.ddim_sigmas
        a_prev = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, alphas_prev[index], device=self.device)
        sqrt_one_minus_alphas = ddim_sampler.model.sqrt_one_minus_alphas_cumprod if use_original_steps else ddim_sampler.ddim_sqrt_one_minus_alphas
        sqrt_one_minus_at = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, sqrt_one_minus_alphas[index], device=self.device)
        sigma_t = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, sigmas[index], device=self.device)

        logs = {}
        # xt -> x0,t
        x0t = (x - sqrt_one_minus_at * score) / a_t.sqrt()
        x0t = x0t.squeeze()
        # x0,t->r0,t
        steps, best_loss, loss, tolerance = 1 if progress < 0.99 else 1000, 1e5, 1e5, 10
        x0t_norm = x0t.detach() + 2
        x0t_norm[self.pis[0]] = self.o_ctxt + 2
        w = [progress * 1] * len(self.projector.trgt_cams)
        while steps > 0 or loss < best_loss:
            steps -= 1
            self.i += 1
            best_loss = min(best_loss, loss)

            # cam_stack = np.random.choice(self.projector.ctxt_cams, size=self.n_cam_ctxt, replace=True,).tolist() +\
            #     np.random.choice(self.projector.trgt_cams, size=self.n_cam_trgt, replace=False,).tolist()
            cam_stack = np.random.choice(self.projector.trgt_cams, size=self.n_cam_trgt, replace=False,).tolist()
            o_stack = x0t_norm[[cam_stack[_].image_name for _ in range(len(cam_stack))]]
            self.update(cams=cam_stack, obs=o_stack,)

        
        x0t_ = self.projector.forward(self.projector.trgt_cams).detach() - 2
        # grad = torch.autograd.grad(self.loss(x0t_[self.pis[0]], self.o_ctxt), x0t_)[0]
        # x_prev = self.opt(x_prev, grad, self.scale)

        # loss = torch.nn.functional.mse_loss(x0t, x0t_)
        # grad = torch.autograd.grad(loss, x)[0]
        # x_prev = self.opt(x_prev, grad, self.scale)
        x_prev = a_prev.sqrt() * x0t_.unsqueeze(1) + (1. - a_prev - sigma_t ** 2).sqrt() * score + self.move_to_device(noise)
        # x0t_ = x0t_.requires_grad_(True)

        logs['o'] = torch.nn.functional.mse_loss(x0t_, self.o).item()
        logs['oc'] = torch.nn.functional.mse_loss(x0t_[self.pis[0]], self.o_ctxt).item()
        logs['c'] = torch.nn.functional.mse_loss(x0t, x0t_).item()
        logs['l'] = loss.item()

        x_prev = self.move_to_device(x_prev, device)
        ddim_solver.iterator.set_postfix(logs)

        return x_prev, logs, {}


class ProjReconCorrectorV2(ReconstructionCorrector):
    def __init__(self, 
                 logdir,
                 scale=1e-2,
                 opt='adam',
                 projector:"LeaptorchProjectorWrapper"=None,
                 o=None,
                 shape=None,
                 pi_ctxt=None, pi_trgt=None,
                 norm=None,
                 geometry=None,
                 gpu_id=torch.device('cuda:6'),
                 ):
        super().__init__(
            logdir=logdir,
            scale=scale,
            opt=opt,
            projector=None,
            gpu_id=gpu_id,
            pi_ctxt=pi_ctxt, 
            pi_trgt=pi_trgt,
            o=o
        )
        
        self.norm = norm
        self.shape = shape
        self.prg_cnt = [0]
        self.geometry = geometry
        self.offset = torch.zeros((256, 256, 256), device=self.device).requires_grad_(True)
        self.opt2 = AdamOptimizer()

    def _update_projector(self, pis=None):
        self.projector = LeaptorchProjectorWrapper(self.geometry, self.device, batch_size=1, pis=pis)
        self.Ax = self.projector.forward
        self.Atx = self.projector.backward

    def modify_score(self, 
                     ddim_sampler: "DDIMSampler", 
                     ddim_solver: "DDIMStepSolver",
                     score, 
                     x, 
                     index=None, 
                     x_prev=None, 
                     dir_xt=None,
                     noise=None,
                     **kw):
        b, *_, device = *x.shape, x.device
        use_original_steps = ddim_sampler.ddim_solver.use_original_steps
        x, score, x_prev = map(self.move_to_device, [x, score, x_prev])
        x = x.requires_grad_(True)
        progress = 1 - index / len(ddim_solver.timesteps)
        
        alphas = ddim_sampler.model.alphas_cumprod if use_original_steps else ddim_sampler.ddim_alphas
        alphas_prev = ddim_sampler.model.alphas_cumprod_prev if use_original_steps else ddim_sampler.ddim_alphas_prev
        sigmas = ddim_sampler.model.ddim_sigmas_for_original_num_steps if use_original_steps else ddim_sampler.ddim_sigmas
        a_t = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, alphas[index], device=self.device)
        a_prev = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, alphas_prev[index], device=self.device)
        sqrt_one_minus_alphas = ddim_sampler.model.sqrt_one_minus_alphas_cumprod if use_original_steps else ddim_sampler.ddim_sqrt_one_minus_alphas
        sqrt_one_minus_at = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, sqrt_one_minus_alphas[index], device=self.device)
        sigma_t = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, sigmas[index], device=self.device)

        pred_x0 = ((x - sqrt_one_minus_at * score) / a_t.sqrt()).squeeze(1)
        
        # self._update_projector()
        # temp = self.norm.backward(pred_x0)
        # # offset_forward = self.Ax(self.offset)

        # s = self.Atx(temp)
        # o_hat_full = self.Ax(s)
        # o_hat_full = self.norm.forward(o_hat_full)
        # o_hat_ctxt_full = o_hat_full[self.pis[0]]
        
        pis = random.sample(self.pis[1], k=96)
        pis = sorted(pis + [_ for _ in self.pis[0] if _ not in pis])
        i_pi_ctxt = [pis.index(_) for _ in self.pis[0]]
        self._update_projector(pis=pis)
        temp = self.norm.backward(pred_x0[pis])
        
        s = self.Atx(temp)
        
        projector = LeaptorchProjectorWrapper(self.geometry, self.device, batch_size=1, )
        o_hat_partial = projector.forward(s)
        o_hat_partial = self.norm.forward(o_hat_partial)
        o_hat_ctxt_partial = o_hat_partial[self.pis[0]]
        
        loss_dict = {
            # 'consistency': torch.nn.functional.mse_loss(o_hat_full, pred_x0),
            # 'context': torch.nn.functional.mse_loss(o_hat_ctxt_full , self.norm.forward(self.o_ctxt)),
            'consistency_partial': torch.nn.functional.mse_loss(o_hat_partial, pred_x0),
            'context_partial': torch.nn.functional.mse_loss(o_hat_ctxt_partial, self.norm.forward(self.o_ctxt)),
        }
        loss = sum([v for k, v in loss_dict.items()])
        
        # x_prev = a_prev.sqrt() * o_hat_full + (1. - a_prev - sigma_t ** 2).sqrt() * score + self.move_to_device(noise)
        grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
        x_prev = self.opt(x_prev, grad, self.scale)
        # grad2 = torch.autograd.grad(loss, self.offset)[0]
        # self.offset = self.opt2(self.offset, grad2, self.scale)

        logs = {}
        logs['o'] = torch.nn.functional.mse_loss(o_hat_partial, self.norm.forward(self.o)).item()
        logs['oc'] = torch.nn.functional.mse_loss(o_hat_ctxt_partial, self.norm.forward(self.o_ctxt)).item()
        logs['l'] = loss.item()
        ddim_solver.iterator.set_postfix(logs)
        
        if round(progress * 1000) % 100 == 0:
            import SimpleITK as sitk
            sitk.WriteImage(sitk.GetImageFromArray(temp.squeeze().cpu().data.numpy()), f"/home/dlr/data/cache/o_hat_leapct2_{round(progress*1000)}.nii.gz")
        
        x_prev = self.move_to_device(x_prev, device)
        return x_prev, logs, self.log_images(None, {})