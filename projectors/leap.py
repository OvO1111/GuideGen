import torch as th
import numpy as np
from leaptorch import ( 
    Projector,
    FBPFunctionGPU, 
    FBPReverseFunctionGPU,
    ProjectorFunctionGPU,
    BackProjectorFunctionGPU
)

from ldm.data.make_dataset.settings import Geometry
from ldm.data.make_dataset.from_latents import ProjectionNorm
from ldm.modules.diffusionmodules.util import instantiate_from_config


class LeaptorchProjectorWrapper:
    def __init__(self, 
                 geometry: Geometry, 
                 device, 
                 batch_size=1, 
                 pis=None
                 ):
        self.device = device
        self.projector = Projector(
            forward_project=False,
            use_static=False,
            use_gpu=True,
            gpu_device=self.device,
            batch_size=batch_size
        )
        self.projector.leapct.set_truncatedScan(True)
        if not isinstance(geometry, Geometry): geometry = instantiate_from_config(geometry)
        self.geometry = geometry
        fn = 'set_parallelbeam' if 'parallel' in geometry['mode'] else 'set_conebeam'
        getattr(self.projector, fn)(**dict(
            numAngles=geometry.total_angles if pis is None else len(pis),
            numRows=geometry.n_detector[0], 
            numCols=geometry.n_detector[1],
            pixelHeight=geometry.d_detector[0],
            pixelWidth=geometry.d_detector[1],
            centerRow=geometry.n_detector[0] // 2,
            centerCol=geometry.n_detector[1] // 2,
            phis=(
                (np.linspace(0, geometry.max_angle, geometry.total_angles) + geometry.start_angle)
                if pis is None else np.array(pis)
            ).astype(np.float32),
        ) | ({} if 'parallel' in geometry['mode'] and 'cone' not in geometry['mode'] else dict(
            sod=geometry.distance_source_origin,
            sdd=geometry.distance_source_detector,
        )))
        self.projector.set_volume(
            numX=geometry.n_voxel[2],
            numY=geometry.n_voxel[1],
            numZ=geometry.n_voxel[0],
            voxelWidth=geometry.d_voxel[-1],
            voxelHeight=geometry.d_voxel[0],
            offsetX=geometry.offset_origin[0],
            offsetY=geometry.offset_origin[1],
            offsetZ=geometry.offset_origin[2],
        )

    def forward(self, x, use_fbp=False):
        # ct -> proj
        if not isinstance(x, th.Tensor): x = th.tensor(x)
        if x.device != self.device: x = x.to(self.device)
        
        ndim = x.ndim
        if x.ndim == 3: x = x[None]
        assert x.shape[-3:] == self.projector.vol_data.shape[-3:], \
            f"shape mismatch in x({x.shape}) and vol_data({self.projector.vol_data.shape})"
        args = (
            x.contiguous(),
            self.projector.proj_data,
            self.projector.vol_data,
            self.projector.param_id_t
        )
        if use_fbp:
            raise NotImplementedError("You should not use FBP for forward projection")
        else:
            y = ProjectorFunctionGPU.apply(*args)
        # y = th.flip(y, dims=[-2, -1])
        if ndim == 3: y = y[0]
        return y

    def backward(self, x, use_fbp=True):
        # proj -> ct
        if not isinstance(x, th.Tensor): x = th.tensor(x)
        if x.device != self.device: x = x.to(self.device)

        ndim = x.ndim
        if x.ndim == 3: x = x[None]
        assert x.shape[-3:] == self.projector.proj_data.shape[-3:], \
            f"shape mismatch in x({x.shape}) and proj_data({self.projector.proj_data.shape})"
        args = (
            x.contiguous(),
            self.projector.proj_data,
            self.projector.vol_data,
            self.projector.param_id_t
        )
        if use_fbp:
            y = FBPFunctionGPU.apply(*args)
        else:
            y = BackProjectorFunctionGPU.apply(*args)
        # y = th.flip(y, dims=[-2,])
        if ndim == 3: y = y[0]
        return y