from typing import NamedTuple
from copy import deepcopy
import numpy as np


hypers0610 = {
    'mode': 'set_parallelbeam',
    'filter': None,
    'distance_source_detector': 1400,
    'distance_source_origin': 1100,
    'n_detector': [256, 512],   # (v,u) resolution
    's_detector': [2, 2],     # size of (u,v) 
    'n_voxel': [256, 512, 512],
    's_voxel': [2, 2, 2],
    'offset_origin': [0, 0, 0],
    'offset_detector': [0, 0],
    'accuracy': 0.5,            # of forward projection
    'total_angles': 360,
    'max_angle': 360,
    'start_angle': 0,
    'noise': {'gaussian': {'mean': 0, 'std': 10}, 'poisson': {'mean': 10000}}
}
hypers0701 = {
    'mode': 'set_parallelbeam',
    'filter': None,
    'distance_source_detector': 7,
    'distance_source_origin': 5,
    'n_detector': [256, 256],   # (v,u) resolution
    's_detector': [2, 2],     # size of (u,v) 
    'n_voxel': [256, 256, 256],
    's_voxel': [2, 2, 2],
    'offset_origin': [0, 0, 0],
    'offset_detector': [0, 0],
    'accuracy': 0.5,            # of forward projection
    'total_angles': 360,
    'max_angle': 360,
    'start_angle': 0,
    'noise': {'gaussian': {'mean': 0, 'std': 10}, 'poisson': {'mean': 10000}}
}
hypers0612 = {
    'mode': 'set_parallelbeam',
    'filter': None,
    'distance_source_detector': 1400,
    'distance_source_origin': 1100,
    'n_detector': [128, 256],   # (v,u) resolution
    's_detector': [2, 2],     # size of (u,v) 
    'n_voxel': [128, 256, 256],
    's_voxel': [2, 2, 2],
    'offset_origin': [0, 0, 0],
    'offset_detector': [0, 0],
    'accuracy': 0.5,            # of forward projection
    'total_angles': 360,
    'max_angle': 360,
    'start_angle': 0,
    'noise': {'gaussian': {'mean': 0, 'std': 10}, 'poisson': {'mean': 10000}}
}


class Geometry(NamedTuple):
    mode: str
    distance_source_detector: float
    distance_source_origin: float
    n_detector: list[int]
    s_detector: list[float]
    n_voxel: list[int]
    s_voxel: list[float]
    offset_origin: list[float]
    offset_detector: list[float]
    total_angles: int = 360
    max_angle: int = 360
    start_angle: int = 0

    @property
    def d_voxel(self):
        return [s / n for n, s in zip(self.n_voxel, self.s_voxel)]
    
    @property
    def d_detector(self):
        return [s / n for n, s in zip(self.n_detector, self.s_detector)]
    
    def _parse(self, x):
        if x in self._fields: return x
        elif x == 'nDetector': x = 'n_detector'
        elif x == 'sDetector': x = 's_detector'             # size all detector pixels = nDetector * dDetector
        elif x == 'nVoxel': x = 'n_voxel'                   
        elif x == 'sVoxel': x = 's_voxel'                   # size all voxel = nVoxel * dVoxel
        elif x == 'offOrigin': x = 'offset_origin'
        elif x == 'offDetector': x = 'offset_detector'
        elif x == 'DSD': x = 'distance_source_detector'
        elif x == 'DSO': x = 'distance_source_origin'
        elif x == 'dVoxel': x = 'd_voxel'                   # size per voxel
        elif x == 'dDetector': x = 'd_detector'             # size per detector pixel
        return x
    
    def __getattr__(self, x):
        return object.__getattribute__(self, self._parse(x))
    
    def __getitem__(self, x):
        return object.__getattribute__(self, self._parse(x))

    def get(self, x, defval=None):
        x = self._parse(x)
        if x not in self._fields: return defval
        return object.__getattribute__(self, x)
    
    def project_points(self, points, angle):
        # points: [N, 3] ranging from [0, 1]
        # d_points: [N, 2] ranging from [-1, 1]

        d1 = self.distance_source_origin
        d2 = self.distance_source_detector

        points = deepcopy(points).astype(float)
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2] # [-0.5, 0.5]
        points *= self.n_voxel[0] * self.d_voxel[0] # mm

        angle = -1 * angle # inverse direction
        rot_M = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [             0,               0, 1]
        ])
        points = points @ rot_M.T
        
        coeff = (d2) / (d1 - points[:, 0]) # N,
        d_points = points[:, [2, 1]] * coeff[:, None] # [N, 2] float
        d_points /= (self.n_detector[0] * self.d_detector[0])
        d_points *= 2 # NOTE: some points may fall outside [-1, 1]
        return d_points
        

geometry0610 = Geometry(
    mode='set_parallelbeam',
    distance_source_detector=1400.,
    distance_source_origin=1100.,
    n_detector=[256, 512],
    s_detector=[2., 2.],  # first axis is up-down (z)
    n_voxel=[256, 512, 512],
    s_voxel=[2., 2., 2.],
    offset_origin=[0., 0., 0.],
    offset_detector=[0., 0.],
    total_angles=360,
    max_angle=360,
    start_angle=0,
)

geometry0701 = Geometry(
    mode='set_parallelbeam',
    distance_source_detector=7.,
    distance_source_origin=5.,
    n_detector=[256, 256],
    s_detector=[2., 2.],  # first axis is up-down (z)
    n_voxel=[256, 256, 256],
    s_voxel=[2., 2., 2.],
    offset_origin=[0., 0., 0.],
    offset_detector=[0., 0.],
    total_angles=360,
    max_angle=360,
    start_angle=0,
)

geometry0714 = Geometry(
    mode='parallel',
    distance_source_detector=7,
    distance_source_origin=5,
    n_detector=[256, 256],
    s_detector=[256, 256],
    n_voxel=[256, 256, 256],
    s_voxel=[256, 256, 256],
    offset_origin=[0, 0, 0],
    offset_detector=[0, 0],
)

geometry0825 = Geometry(
    mode='set_parallelbeam',
    distance_source_detector=10.,
    distance_source_origin=5.,
    n_detector=[256, 256],
    s_detector=[2, 2],  # first axis is up-down (z)
    n_voxel=[256, 256, 256],
    s_voxel=[2, 2, 2],
    offset_origin=[0., 0., 0.],
    offset_detector=[0., 0.],
    total_angles=360,
    max_angle=360,
    start_angle=0,
)

geometry0825_todai = Geometry(
    mode='set_parallelbeam',
    distance_source_detector=10.,
    distance_source_origin=5.,
    n_detector=[256, 256],
    s_detector=[2, 2],  # first axis is up-down (z)
    n_voxel=[256, 256, 256],
    s_voxel=[2, 2, 2],
    offset_origin=[0., 0., 0.],
    offset_detector=[0., 0.],
    total_angles=850,
    max_angle=360,
    start_angle=0,
)

geometry0825_onerow = Geometry(
    mode='set_parallelbeam',
    distance_source_detector=7.,
    distance_source_origin=5.,
    n_detector=[1, 256],
    s_detector=[1., 2.],  # first axis is up-down (z)
    n_voxel=[1, 256, 256],
    s_voxel=[1., 2., 2.],
    offset_origin=[0., 0., 0.],
    offset_detector=[0., 0.],
    total_angles=360,
    max_angle=360,
    start_angle=0,
)

geometry_brain = Geometry(
    mode='set_conebeam',
    distance_source_detector=15000,
    distance_source_origin=7500,
    n_detector=[64, 256],
    s_detector=[1., 1],  # first axis is up-down (z)
    n_voxel=[64, 256, 256],
    s_voxel=[1, 1, 1],
    offset_origin=[0., 0., 0.],
    offset_detector=[0., 0.],
    total_angles=984,
    max_angle=360,
    start_angle=0,
)