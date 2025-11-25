## this file is for standalone testing of a trained ckpt
import os
import sys
import json
import random
import numpy as np
import torch
import SimpleITK as sitk

from tqdm import tqdm
from functools import partial
from collections import defaultdict
from skimage.transform import resize
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from inference.metric_cal import MetricLog
from ldm.data.make_dataset.settings import Geometry
from ldm.data.make_dataset.from_latents import ProjectionNorm

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.append('/home/dlr/code/diffusion/r2_gaussian/')
try:
    from projectors.leap import LeaptorchProjectorWrapper
except ImportError as e:
    print(e)
    print(f"leaptorch is not available")

try:
    from projectors.r2gaussian import (
        R2GaussianProjectorWrapper,
        R2GaussianV2,
        debug_loop
    )
except ImportError as e:
    print(e)
    print("r2gaussian is not available")

try:
    from projectors.tigre import TigreProjectorWrapper
except ImportError as e:
    print(e)
    print("tigre is not available")

from ldm.data.make_dataset.settings import *
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.sampling.ddim import DDIMSampler
from ldm.models.diffusion.sampling.helpers.solver import DDIMStepInverter
from ldm.models.diffusion.sampling.helpers.corrector import ProjReconCorrector, ProjReconCorrectorV2, PcdReconCorrector
    

def write_file_no_metadata(file, path):
    assert file.ndim == 3, f"file should have a dimension of 3"
    if not isinstance(file, np.ndarray): file = file.data.cpu().numpy()
    sitk.WriteImage(sitk.GetImageFromArray(file.astype(np.float32)), path)
    return 1


class Projector:
    def __init__(self, geometry: Geometry, device=None, name='leap', 
                 pi_ctxt=None, pi_trgt=None):
        self.geometry = geometry
        self.pis = [pi_ctxt, pi_trgt]
        self.device = device or torch.device('cpu')
        projector = LeaptorchProjectorWrapper if name == 'leap' else TigreProjectorWrapper
        self._A = projector(geometry=geometry, device=device, pis=pi_ctxt)
        self._B = projector(geometry=geometry, device=device, pis=pi_trgt)
        
    def Ax(self, x, **kw):
        return self._A.forward(x, **kw)

    def Ax_full(self, x, **kw):
        return self._B.forward(x, **kw)
    
    def Atx(self, y, **kw):
        return self._A.backward(y, **kw)
    
    def Atx_full(self, y, **kw):
        return self._B.backward(y, **kw)


class Tester:
    def __init__(
        self,
        inputs=None,
        config=None,
        ckpt_path=None,
        use_dataset=False,
        geometry=geometry0825,
        normalizer=None,
        gpu_device=[0],
        test_fn_params={}
    ):
        assert inputs or config, f"either inputs or config must be destined"
        self.available_devices = [torch.device(f'cuda:{dev}') for dev in gpu_device]
        self.device = self.available_devices[0]
        config = [OmegaConf.load(os.path.join(config, _)) for _ in sorted(os.listdir(config))]
        self.config = OmegaConf.merge(*config)
        self.test_fn_params = test_fn_params
        if inputs is None: use_dataset = True
        
        if use_dataset:
            self.data = self.make_dataset()
        else:
            if isinstance(inputs, dict): self.data = instantiate_from_config(inputs)
            elif isinstance(inputs, list): raise NotImplementedError("for now cannot directly read data paths, please provide a dataset class")
            elif isinstance(inputs, str):
                assert os.path.exists(inputs), f"if inputs is a string, then it should be present in the fs, got {inputs}"
                with open(inputs) as f:
                    if inputs.endswith('json'): self.data = json.load(f)
                    elif inputs.endswith('yaml'): self.data = OmegaConf.to_container(OmegaConf.load(f))
                self.data = instantiate_from_config(self.data)
        
        self.geometry = geometry
        self.normalizer = normalizer
        self.metric_logger = MetricLog()
        self.model = self.make_model(ckpt_path)
        self.model: LatentDiffusion = self.model.eval().to(self.device)
            
    def make_dataset(self):
        data_conf = self.config['data']['params']
        test_data = OmegaConf.merge(data_conf.get('common', {}), data_conf.get('test', data_conf['validation']))
        # test_data['params']['data_dir'] = '/home/dlr/data/datasets/synthetic/projs_set0701'
        return instantiate_from_config(test_data)
    
    def make_model(self, ckpt_path=None):
        model_conf = self.config['model']
        # if 'test_target' in model_conf:
        #     test_model = model_conf['test_target']
        if 'train_target' in model_conf:
            test_model = model_conf['train_target']
        else:
            test_model = model_conf['target']
        if ckpt_path is not None: model_conf["params"]["ckpt_path"] = ckpt_path
        return instantiate_from_config({"target": test_model, "params": model_conf["params"]})
    
    def move_to_device(self, data, device=None):
        device = device or self.device
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, device=device)
        elif isinstance(data, torch.Tensor):
            data = data.to(device)
        elif isinstance(data, list):
            data = [self.move_to_device(d) for d in data]
        elif isinstance(data, dict):
            data = {k: self.move_to_device(v) for k, v in data.items()}
        return data

    def read_data(self, 
        batch, 
        n_context=6, 
        n_target=90,
        p_angle_context=.5, 
        p_angle_target=1,
        select_method='random',
    ):
        # choose context views
        proj, vol = batch.get('o', batch.get('projection')), batch.get('image', batch.get('volume'))
        # volume
        if vol is None:
            casename = batch.get('casename')
            vol_path = os.path.join('/home/dlr/data/datasets/synthetic/projs_set0825', 'val', f'{casename}.npz')
            if os.path.exists(vol_path):
                vol = torch.tensor(np.load(vol_path)['image']).to(proj.device)
            if vol is None:
                vol_path = os.path.join('/home/dlr/data/datasets/SinoCT', 'val', f'{casename}.npz')
                if os.path.exists(vol_path):
                    vol = torch.tensor(np.load(vol_path)['image']).to(proj.device)
            if vol is None:
                projector = TigreProjectorWrapper(self.geometry, )
                vol = torch.tensor(projector.backward(proj.cpu().numpy(), with_leapct=False)).to(proj.device)
        # context views
        if select_method == 'random':
            i_context = torch.randperm(round(p_angle_context * proj.shape[0]))
            i_context = torch.sort(i_context[:n_context])[0].numpy().tolist()
        elif select_method == 'uniform':
            i_context = torch.linspace(0, round(p_angle_context * proj.shape[0])*(n_context-1)/n_context, n_context).long().numpy().tolist()
            pi_context = torch.linspace(0, p_angle_context * 360, n_context + 1)[:n_context].numpy().tolist()
        elif select_method == 'active':
            # leave the context view selection to the model
            i_context = [0]
            pi_context = [0]
        # target views
        # if n_target <= round(p_angle_target * proj.shape[0]):
        #     pi_target = torch.arange(round(p_angle_target * proj.shape[0]))[::round(p_angle_target * proj.shape[0])//n_target][:n_target].numpy().tolist()
        #     pi_target = [_ / (proj.shape[0] / 360) for _ in pi_target]
        #     i_target = torch.linspace(0, proj.shape[0], n_target, dtype=int).numpy().tolist()
        # else:
        i_target = torch.linspace(0, round(p_angle_target * proj.shape[0])*(n_target-1)/n_target, n_target).long().numpy().tolist()
        pi_target = [_ / (proj.shape[0] / 360) for _ in i_target]
        # i_target = torch.arange(0, n_target).numpy().tolist()
        # learned conditioning
        cond = self.model.get_learned_conditioning(torch.tensor(pi_target, device=self.device))
        return (i_context, pi_context, i_target, pi_target), proj, vol, cond

    def test(self, method='leaptorch', n_case=5, **kw):
        metrics, agg = {}, defaultdict(lambda: defaultdict(list))
        idx = kw.pop('idx', 0)
        # now we will save each experiment setting to a different folder
        save_dir = os.path.join(f"/home/dlr/data/xsvct", f"{method}", f"{kw['n1']}")
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        for i in range(idx, n_case + idx):
            if method == 'leaptorch':
                metrics[i] = self.test_singlefile_2dproj_to_3d(idx=i, **kw)
            elif method == 'r2gaussian':
                metrics[i] = self.test_singlefile_2dproj_to_3d_r2gaussian(idx=i, **kw)
            elif method == 'fbp':
                metrics[i] = self.test_physical_models(idx=i, **kw)

        for case in metrics.values():
            for var, metric_dict in case.items():
                for metric, val in metric_dict.items():
                    agg[var][metric].append(val)
        agg['mean'] = {
            var: {metric: round(np.array(vals).mean(), 2) for metric, vals in metric_dict.items()}
            for var, metric_dict in agg.items()
        }
        agg['std'] = {
            var: {metric: round(np.array(vals).std(), 2) for metric, vals in metric_dict.items()}
            for var, metric_dict in agg.items() if var != 'mean'
        }
        agg = dict(agg)
        for key, mean, std in zip(agg['mean'].keys(), agg['mean'].values(), agg['std'].values()):
            for k, m, s in zip(mean.keys(), mean.values(), std.values()):
                print(f"{k} of {key}: {float(m):.2f} ± {float(s):.2f}")
        return agg

    @torch.no_grad()
    def test_singlefile(self, idx=0, use_ddim=True):
        from ldm.models.diffusion.ddpm import LatentDiffusion
        from ldm.models.diffusion.sampling.ddim import DDIMSampler
        assert isinstance(self.model, LatentDiffusion), f"diffusion is an instance of LatentDiffusion"

        N = 36
        S = 100
        batch = self.move_to_device(self.data[idx])
        x, c, cameras = self.model.get_input(batch)
        N = min(N, x.shape[0])

        with self.model.ema_scope():
            ddim_sampler = DDIMSampler(self.model, max_batch=32)
            shape = (self.model.channels, ) + tuple(self.model.infer_image_size)
            samples, intermediates = ddim_sampler.sample(
                S=S, 
                batch_size=N,
                shape=shape,
                # ddim_use_original_steps=True,
                verbose=True,
                conditioning=c,
                max_batch=56,
                eta=1,
                cameras=cameras,
            )
        write_file_no_metadata(samples.squeeze(), f"/home/dlr/data/cache/o_ddim_no_infer_cond.nii.gz")
        
        logs = {'x_hat': samples.squeeze(), 'x_gt': x.squeeze()}
        self.compute_metrics(logs)
        return samples
    
    @torch.no_grad()
    def test_singlefile_2dproj_to_3d(self, idx=0, n1=1, n2=360, view_noise_eq_dose=1, angle_uncertainty=0):
        batch = self.move_to_device(self.data[idx])
        casename = batch.get('casename', idx)
        print(f"processing case {casename}")
        (i_ctxt, pi_ctxt, i_trgt, pi_trgt), o, s, cond = self.read_data(batch, n_context=n1, n_target=n2, p_angle_target=1, select_method='uniform')
        o = (o - o.min()) / (o.max() - o.min())

        device = self.available_devices[1]
        projector = Projector(self.geometry, device, pi_ctxt=pi_ctxt, pi_trgt=pi_trgt)
        # y = projector.Atx(o[pi_ctxt]).data.cpu().numpy()
        # gaussian = R2GaussianProjectorWrapper(
        #     geometry0825, 
        #     device, 
        #     s_init=None,#y[::-1].transpose(2, 1, 0)*15,
        #     o=self.move_to_device(ProjectionNorm().forward(o), device), 
        #     steps=1e3,
        #     pi_ctxt=i_ctxt,
        #     pi_trgt=i_trgt,
        # )
        score_corrector = ProjReconCorrector(
                projector=projector,
                shape=o.shape,
                o=o[torch.linspace(0, o.shape[0]-1, len(pi_trgt), dtype=int).numpy().tolist()],
                norm=self.normalizer,
                gaussian=None,
                pi_ctxt=pi_ctxt, pi_trgt=pi_trgt,
                gpu_id=device,
                scale=3e-2 if 'SinoCT' in casename else 1e-2,
                logdir=f'/home/dlr/data/cache/logs/leapct/{n1}slices',
                view_noise_eq_dose=view_noise_eq_dose,
                angle_uncertainty=angle_uncertainty,
            )
        ddim_sampler1 = DDIMSampler(
            self.model, 
            score_corrector=score_corrector,
            # score_corrector=ProjReconCorrectorV2(
            #     shape=o.shape,
            #     o=o,
            #     geometry=geometry0825,
            #     norm=ProjectionNorm(),
            #     pi_ctxt=pi_ctxt, pi_trgt=pi_trgt,
            #     gpu_id=device,
            #     scale=1e-2,
            #     logdir=f'/home/dlr/data/cache/logs/leapct/{n1}slices'),
            max_batch=128 if 'SinoCT' in casename else 50,
            schedule='quad',
        )
        shape = (self.model.channels, ) + tuple(self.model.infer_image_size)
        o_hat, _ = ddim_sampler1.sample(
            S=100, 
            batch_size=len(pi_trgt), 
            shape=shape, 
            verbose=False, 
            conditioning=cond.view(-1, 1, *cond.shape[2:]),
            repeats=1,
            ddim_use_original_steps=True,
            eta=1,
            # preset=np.array([0] * 100),
            # x_T=x_T
            # repeat_noise=True,
        )
        o_hat = o_hat.squeeze(1)
        if view_noise_eq_dose == 1 and angle_uncertainty == 0:
            i_pi_ctxt_in_pi_trgt = [round(i * o_hat.shape[0] / o.shape[0]) for i in i_ctxt]
            o_hat[i_pi_ctxt_in_pi_trgt] = self.normalizer.forward(o[i_ctxt])
            # qualitative results
            write_file_no_metadata(o_hat, os.path.join(self.save_dir, f"{casename}_o.nii.gz"))
            s_hat = torch.flip(projector.Atx_full(o_hat), (0, 1, 2))
            write_file_no_metadata(s_hat, os.path.join(self.save_dir, f"{casename}_s.nii.gz"))
        else:
            write_file_no_metadata(o_hat, os.path.join(self.save_dir, f"{casename}_o_{view_noise_eq_dose:.4f}_{angle_uncertainty:.4f}.nii.gz"))
            if view_noise_eq_dose < 1:
                write_file_no_metadata(score_corrector.o_ctxt, os.path.join(self.save_dir, f"{casename}_{view_noise_eq_dose:.4f}noisy_o_ctxt.nii.gz"))
            if angle_uncertainty > 0:
                print('\nraw angles:\n', pi_ctxt, '\nnoisy angles:\n', score_corrector.pis[0])
            s_hat = torch.flip(projector.Atx_full(o_hat), (0, 1, 2))
            write_file_no_metadata(s_hat, os.path.join(self.save_dir, f"{casename}_s_{view_noise_eq_dose:.4f}_{angle_uncertainty:.4f}.nii.gz"))
        o = o[i_trgt]
        # quantitative results
        if s is None: s = torch.zeros_like(s_hat); print('no s gt found, using zeros')
        o, o_hat, s, s_hat = map(lambda x: ((x - x.min()) / (x.max() - x.min() + 1e-5)).cpu(), (o, o_hat, s, s_hat))
        o, o_hat, s, s_hat = map(lambda x: x.cpu().numpy(), (o, o_hat, s, s_hat))
        _dict_ = {
            'o': {
                'ssim': ssim(o, o_hat, data_range=1.),
                'psnr': psnr(o, o_hat, data_range=1.),
                # 'lpips': self.metric_logger.lpips(o, o_hat),
                # 'rmse': self.metric_logger.rmse(o, o_hat),
            },
            's': {
                'ssim': ssim(s, s_hat, data_range=1.),
                'psnr': psnr(s, s_hat, data_range=1.),
                # 'lpips': self.metric_logger.lpips(s, s_hat),
                # 'rmse': self.metric_logger.rmse(s, s_hat),
            }
        }
        return _dict_

    @torch.no_grad()
    def test_singlefile_2dproj_to_3d_r2gaussian(self, idx=1, n1=1, n2=360):
        batch = self.move_to_device(self.data[idx])
        (pi_ctxt, pi_trgt), o, s, cond = self.read_data(batch, n_context=n1, n_target=n2, select_method='uniform')
        # init gaussian
        norm = ProjectionNorm(om=1, oM=3)
        fbp_projector = LeaptorchProjectorWrapper(geometry=geometry0825, device=torch.device("cuda:0"))
        x_like = torch.zeros_like(o)
        x_like[pi_ctxt] = o[pi_ctxt]
        y = fbp_projector.backward(norm.forward(x_like)).data.cpu().numpy()
        projector = R2GaussianProjectorWrapper(
            geometry0825, 
            torch.device("cuda:0"), 
            s_init=y[::-1].transpose(2, 1, 0)*15,
            o=self.move_to_device(norm.forward(o), torch.device("cuda:0")), 
            steps=1e5,
            pi_ctxt=pi_ctxt,
            pi_trgt=pi_trgt,
        )
        
        ddim_sampler = DDIMSampler(
            self.model, 
            score_corrector=PcdReconCorrector(
                projector=projector,
                o=ProjectionNorm().forward(o),
                gpu_id=torch.device("cuda:0"),
                scale=1e-2,
                normalization=ProjectionNorm(),
                pi_ctxt=pi_ctxt,
                pi_trgt=pi_trgt,
                logdir=f'/home/dlr/data/cache/logs/r2gaussian/{n1}slices'),
            max_batch=60,
        )
        shape = (self.model.channels, ) + tuple(self.model.infer_image_size)
        o_hat, _ = ddim_sampler.sample(
            S=200, 
            batch_size=len(pi_trgt), 
            shape=shape, 
            verbose=True, 
            conditioning=cond[:, pi_trgt].view(-1, 1, *cond.shape[2:]),
            eta=1,
            repeats=1,
            ddim_use_original_steps=True,
        )
        # qualitative results
        o_hat_r2, s_hat_r2 = projector.forward(projector.trgt_cams), projector.backward()
        write_file_no_metadata(o_hat.squeeze(), f"/home/dlr/data/cache/ddpm_o_{n1}ddim.nii.gz")
        write_file_no_metadata(o_hat_r2, f"/home/dlr/data/cache/r2_gaussian_o_{n1}ddim.nii.gz")
        write_file_no_metadata(s_hat_r2, f"/home/dlr/data/cache/r2_gaussian_s_{n1}ddim.nii.gz")
        # quantitative results
        o, o_hat, s, o_hat_r2, s_hat_r2 = \
            map(lambda x: ((x - x.min()) / (x.max() - x.min())).cpu(), (o, o_hat, s, o_hat_r2, s_hat_r2))
        o, o_hat_r2, s_hat_r2, s = map(lambda x: x[:, None], (o, o_hat_r2, s_hat_r2, s))
        _dict_ = {
            'o': {
                'ssim': self.metric_logger.ssim(o, o_hat),
                'psnr': self.metric_logger.psnr(o, o_hat),
                'lpips': self.metric_logger.lpips(o, o_hat),
                'rmse': self.metric_logger.rmse(o, o_hat),
            },
            'o_r2': {
                'ssim': self.metric_logger.ssim(o, o_hat_r2),
                'psnr': self.metric_logger.psnr(o, o_hat_r2),
                'lpips': self.metric_logger.lpips(o, o_hat_r2),
                'rmse': self.metric_logger.rmse(o, o_hat_r2),
            },
            's_r2': {
                'ssim': self.metric_logger.ssim(s, s_hat_r2),
                'psnr': self.metric_logger.psnr(s, s_hat_r2),
                'lpips': self.metric_logger.lpips(s, s_hat_r2),
                'rmse': self.metric_logger.rmse(s, s_hat_r2),
            }
        }
        print(_dict_)
        
    def test_r2gaussian(self, idx=0, nc=1):
        # batch = self.move_to_device(self.data[idx])
        # x, c = self.model.get_input(batch, self.model.first_stage_key)

        # n1, n2, s = 1, 360, 3001
        # indices = torch.randperm(batch.get(self.model.first_stage_key).shape[0])
        # # pi_ctxt = torch.sort(torch.randperm(90)[:n1])[0].numpy().tolist()
        # pi_ctxt = torch.sort(indices[:n1])[0].numpy().tolist()
        # pi_trgt = torch.sort(indices[:n2])[0].numpy().tolist()

        # fbp_projector = TigreProjectorWrapper(geometry=geometry0701, pis=np.array(pi_ctxt))
        # y = fbp_projector.backward(x[pi_ctxt].data.cpu().numpy())

        s = 10001
        file = f'/mnt/data_1/dlr/data/cache/leapct_o_{nc}ddim.nii.gz'
        x = self.move_to_device(sitk.GetArrayFromImage(sitk.ReadImage(file)))
        # x_aat = self.move_to_device(sitk.GetArrayFromImage(sitk.ReadImage(file.replace('.nii.gz', '_aat.nii.gz'))))
        # pixel_conf = torch.nn.functional.mse_loss(x, x_aat, reduction='none')
        # pixel_conf = pixel_conf.max() - pixel_conf
        # # view_conf = torch.sin(torch.linspace(0, 2*torch.pi, 360)).abs()
        # pixel_conf = torch.ones_like(pixel_conf)
        # view_conf = torch.ones(360)
        # view_conf[0] = 2
        # pixel_conf /= pixel_conf.sum()
        # view_conf /= view_conf.sum()
        y = self.move_to_device(sitk.GetArrayFromImage(sitk.ReadImage(file.replace('leapct_o', 'leapct_s'))))
        pi_ctxt = pi_trgt = [i for i in range(360)]
        
        # gaussian = R2GaussianProjectorWrapper(
        #     geometry=geometry0701,
        #     device=self.device,
        #     s_init=((y - y.min()) / (y.max() - y.min())).transpose(2, 1, 0),
        #     o=self.move_to_device(x, self.device),
        #     steps=s,
        #     pi_ctxt=pi_ctxt,
        #     pi_trgt=pi_trgt,
        # )
        gaussian = R2GaussianV2(
            geometry=geometry0701,
            device=self.device,
            s_init=((y - y.min()) / (y.max() - y.min())).cpu().numpy().transpose(2, 1, 0),
            o=self.move_to_device(x, self.device),
            steps=s,
            pi_ctxt=pi_ctxt,
            pi_trgt=pi_trgt,
            ckpt_dir='/home/dlr/data/ldm/proj_256_256/',
            # pixel_conf=pixel_conf,
            # view_conf=view_conf,
        )
        # gaussian.gaussians.load_ply('/mnt/data_1/dlr/data/cache/r2gaussian/n1_1_tune_on_processed/r2gaussian_gaussians_ep400.ply')

        # debug_loop(gaussian, 
        #            log_dir=f'/home/dlr/data/cache/r2gaussian/n1_{n1}',
        #            vol_data=f"/home/dlr/data/datasets/CT-RATE/imagesTr/{batch['casename'][len('CT-RATE_'):]}.nii.gz")

        debug_loop(gaussian,
                   log_dir=f'/home/dlr/data/cache/r2gaussian/n{nc}_tune_on_processed',
                   use_single_cam_per_iter=True)
    
    def test_physical_models(self, idx=0, n1=16, n2=360, alg='fdk'):
        batch = self.move_to_device(self.data[idx])
        casename = batch.get('casename', idx)
        print(casename)
        (i_ctxt, pi_ctxt, i_trgt, pi_trgt), o, s, cond = self.read_data(batch, n_context=n1, n_target=n2, p_angle_target=1, select_method='uniform')
        _o = o.cpu().numpy()
        _o = (_o - _o.min()) / (_o.max() - _o.min())
        projector_backward = TigreProjectorWrapper(
            geometry=self.geometry,
            pis=np.array(pi_ctxt)
        )
        projector_forward = TigreProjectorWrapper(
            geometry=self.geometry,
        )
        
        o_ctxt = _o[i_ctxt]
        s_hat = projector_backward.backward(o_ctxt, alg=alg, with_leapct=False)
        o_hat = projector_forward.forward(s_hat)[:, ::-1].copy()
        for layer in range(o_hat.shape[0]):
            if np.abs(o_hat[layer]).max() > 100:
                o_hat[layer] = (o_hat[(layer + 1) % o_hat.shape[0]] + o_hat[(layer - 1) % o_hat.shape[0]]) / 2

        o_hat, s_hat = self.move_to_device(o_hat), self.move_to_device(s_hat)
        if s is None: s = torch.zeros_like(s_hat); print('no s gt found, using zeros')
        o, o_hat, s, s_hat = map(lambda x: ((x - x.min()) / (x.max() - x.min() + 1e-5)).cpu().data.numpy(), (o, o_hat, s, s_hat))

        write_file_no_metadata(o_hat, os.path.join(self.save_dir, f"{casename}_o_{alg}.nii.gz"))
        write_file_no_metadata(s_hat, os.path.join(self.save_dir, f"{casename}_s_{alg}.nii.gz"))
        _dict_ = {
            'o': {
                'ssim': ssim(o, o_hat, data_range=1),
                'psnr': psnr(o, o_hat, data_range=1),
                # 'lpips': self.metric_logger.lpips(*map(lambda t: torch.tensor(t, device=self.device), (o, o_hat))),
            },
            's': {
                'ssim': ssim(s, s_hat, data_range=1),
                'psnr': psnr(s, s_hat, data_range=1),
                # 'lpips': self.metric_logger.lpips(*map(lambda t: torch.tensor(t, device=self.device), (s, s_hat)), reset=True),
            }
        }
        return _dict_


if __name__ == "__main__":
    metrics = {}
    # test_folder = '/home/dlr/data/ldm/proj_256_256/'
    test_folder = '/home/dlr/data/ldm/proj_64_256/'
    # test_folder = '/home/dlr/data/ldm/proj_2d_256_256_1each/'
    # test_folder = '/home/dlr/data/ldm/proj_256_256_todai/'
    lung_data_wo_norm = {
        "target": "ldm.data.make_dataset.from_latents.LatentsDataset",
        "params": {
            "split": "val",
            "data_dir": "/home/dlr/data/datasets/synthetic/projs_set0701"
        }
    }
    lung_data_w_norm = {
        "target": "ldm.data.synthetic.SVCTDataset",
        "params": {
            "split": 'val',
            "data_dir": '/home/dlr/data/datasets/synthetic/projs_set0825',
        }
    }
    brain_data_wo_norm = {
        "target": "ldm.data.synthetic.SVCTDataset",
        "params": {
            "split": 'val',
            "data_dir": '/home/dlr/data/datasets/SinoCT',
            "n_angles_total": 984,
            "data_dir": "/home/dlr/data/datasets/SinoCT",
            "resize_to": [984, 64, 256],
            "input_projection_min_max": None
        }
    }
    natural_data = {
        "target": "ldm.data.synthetic.SVCTDataset",
        "params": {
            "split": 'val',
            "data_dir": '/home/dlr/data/datasets/synthetic/projs_set0825_todai',
            "n_angles": 360,
            "input_projection_min_max": (0, 20),
            "input_volume_min_max": (0, 0.4),
        }
    }

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nc', type=int, nargs='+', default=[8], help='context views')
    parser.add_argument('--N', type=int, default=1, help='n cases eval')
    parser.add_argument('--idx', type=int, default=0, help='index of first case')
    parser.add_argument('--dsname', type=str, default='todai', help='which dataset to use')
    parser.add_argument('--dose', type=float, default=1, help='dose level')
    parser.add_argument('--uncty', type=float, default=0, help='angle uncertainty level')
    args = parser.parse_args()
    
    gpus = [1, 0]
    cfg_ct_rate = {
        'config': '/home/dlr/data/ldm/proj_256_256/configs',
        'ckpt_path': '/home/dlr/data/ldm/proj_256_256/checkpoints/last-v1.ckpt',
        'use_dataset': False,
        'inputs': lung_data_wo_norm,
        'gpu_device': gpus,
        'geometry': geometry0825,
        'normalizer': ProjectionNorm(),
    }
    cfg_sino_ct = {
        'config': '/home/dlr/data/ldm/proj_64_256/configs',
        'ckpt_path': '/home/dlr/data/ldm/proj_64_256/checkpoints/last.ckpt',
        'use_dataset': False,
        'inputs': brain_data_wo_norm,
        'gpu_device': gpus,
        'geometry': geometry_brain,
        'normalizer': ProjectionNorm(pm=0, pM=5),
    }
    cfg_todai_objects = {
        'config': '/home/dlr/data/ldm/proj_256_256/configs',
        'ckpt_path': '/home/dlr/data/ldm/proj_256_256/checkpoints/last-v1.ckpt',
        'use_dataset': False,
        'inputs': natural_data,
        'gpu_device': gpus,
        'geometry': geometry0825_todai,
        'normalizer': ProjectionNorm(pm=0.5, pM=1),
    }
    # tester = Tester(**cfg_ct_rate)
    tester = Tester(**(cfg_sino_ct if args.dsname == 'sinoct' else cfg_ct_rate if args.dsname == 'ctrate' else cfg_todai_objects))
    # tester = Tester(**cfg_todai_objects)
    
    # tester = Tester(config=os.path.join(test_folder, 'configs'),
    #                 ckpt_path=os.path.join(test_folder, 'checkpoints', 'last.ckpt'),
    #                 use_dataset=False,
    #                 inputs=brain_data_w_norm,
    #                 gpu_device=[1, 0],
    #                 geometry=geometry_brain)
    # tester = TesterMultiChn(config=os.path.join(test_folder, 'configs'),
    #                         ckpt_path=os.path.join(test_folder, 'checkpoints', 'last.ckpt'),
    #                         use_dataset=True,
    #                         gpu_device=[0, 1])
    # tester = Tester(inputs=vol_data,
    #                 config=os.path.join(test_folder, 'configs'),
    #                 ckpt_path=os.path.join(test_folder, 'checkpoints', 'last.ckpt'),
    #                 use_dataset=False,
    #                 gpu_device=torch.device('cuda:7'))
    for nc in args.nc:
        print(f"=== {nc} ===")
        # tester.test_singlefile()
        metrics[nc] = tester.test(method='leaptorch', n1=nc, n_case=args.N, idx=args.idx, view_noise_eq_dose=args.dose, angle_uncertainty=args.uncty)
        # metrics[nc] = tester.test(method='fbp', alg='cgls', n1=nc, n_case=args.N, idx=args.idx)
        # tester.test_singlefile_2dproj_to_3d_r2gaussian()
        # tester.test_r2gaussian(nc=nc)
    # print(json.dumps(metrics, indent=4))