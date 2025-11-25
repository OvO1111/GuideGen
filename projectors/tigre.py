import tigre
import tigre.algorithms as algs

import numpy as np
from ldm.data.make_dataset.settings import Geometry


class TigreProjectorWrapper:
    def __init__(self,
                 geometry: Geometry,
                 pis=None, **kw):
        if 'parallel' in geometry.mode:
            self.geometry = tigre.geometry(mode="parallel", nVoxel=np.array(geometry.nVoxel[::-1]))
        elif 'cone' in geometry.mode:
            self.geometry = tigre.geometry(mode="cone")
        else:
            raise NotImplementedError(f"not supported mode {geometry.mode}")
        
        self.geometry.DSD = geometry.distance_source_detector
        self.geometry.DSO = geometry.distance_source_origin
        self.geometry.nDetector = np.array(geometry.n_detector)
        self.geometry.sDetector = np.array(geometry.s_detector)
        self.geometry.dDetector = np.array(geometry.dDetector)
        self.geometry.nVoxel = np.array(geometry.n_voxel)
        self.geometry.sVoxel = np.array(geometry.s_voxel)
        self.geometry.dVoxel = np.array(geometry.dVoxel)
        self.geometry.offOrigin = np.array(geometry.offset_origin[::-1])
        self.geometry.offDetector = np.array(geometry.offset_detector[::-1] + [0])
        self.geometry.accuracy = 1
        self.geometry.filter = None

        if pis is None: pis = np.linspace(0, 360, geometry.total_angles, endpoint=False)
        self.pis = -np.array(pis) * np.pi / 180 - np.pi / 2  # to radian and start from -90 degree

    def forward(self, x):
        y = tigre.Ax(x + np.abs(x.min()), self.geometry, self.pis)
        return y
    
    def backward(self, x, alg='fdk', with_leapct=True):
        x = x[:, ::-1]  # if forward uses leapct (as in our datasets)
        if alg == 'fdk':
            y = algs.fdk(x, self.geometry, self.pis)
        elif alg == 'cgls': 
            y, _ = algs.cgls(x, self.geometry, self.pis, 10, computel2=True)
        elif alg == 'ossart':
            y = algs.ossart(x, self.geometry, self.pis, 60)
        if with_leapct: y = y.transpose(0, 2, 1)  # if forward uses leapct
        return y