import random, sys
import torchio as tio
import os, torch, numpy as np
from torch.utils.data import default_collate

sys.path.append('/home/dlr/code/diffusion/')

from copy import deepcopy
from ldm.util import instantiate_from_config
from ldm.data.make_dataset.settings import geometry0825


def to_torch(x, **kw):
    return torch.tensor(x, **kw)


class SVCTDataset(torch.utils.data.Dataset):
    def __init__(self,
        data_dir='/hot_data/dlr/datasets/synthetic/projs_set0825',
        split='train',
        include_cases=None,
        max_size=None,
        n_angles_total=360,
        n_angles=32,
        batch_size=32,
        n_points=10000,
        network_depth=3,
        output_size=(256, 256, 256),
        output_keys=None,
        input_projection_min_max=(0.5e5, 3.5e5),
        input_volume_min_max=(-1024, 2048),
        fixed_angle_per_case=True,
        resize_to=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.files = [f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('npz')]
        if include_cases is not None:
            self.files = [f for f in self.files if any([c in f for c in include_cases])]
        self.files = self.files[:max_size]
        
        self.batch_size = batch_size
        self.n_angles = n_angles
        self.n_points = n_points
        self.geometry = geometry0825
        self.output_size = output_size
        self.output_keys = output_keys #+ [f'points_2d_depth_{l}' for l in range(network_depth)]
        self.network_depth = network_depth
        self.fixed_angle_per_case = fixed_angle_per_case
        if self.fixed_angle_per_case:
            self.file_angle = [random.choices(range(0, n_angles_total), k=self.n_angles) for _ in range(len(self.files))]
        self.transforms = tio.Compose([
            tio.RescaleIntensity(out_min_max=(-1, 1), in_min_max=input_projection_min_max, include=['projection'])
            if input_projection_min_max is not None else tio.Lambda(lambda x: x) ,
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=input_volume_min_max, include=['volume'])
            if input_volume_min_max is not None else tio.Lambda(lambda x: x, include=['projection']) ,
        ] + ([] if resize_to is None else [tio.Resize(resize_to, include=['projection'])]))
        
    def __len__(self, ):
        return len(self.files)
    
    def project_points(self, points, angle, layer=0):
        d1 = self.geometry.distance_source_origin
        d2 = self.geometry.distance_source_detector
        n_voxel = self.geometry.n_voxel[0]
        d_voxel = self.geometry.d_voxel[0]
        n_detector = self.geometry.n_detector[0]
        d_detector = self.geometry.d_detector[0] * 2 ** layer

        points = deepcopy(points).astype(float)
        points[:, 1:] -= 0.5 # [-0.5, 0.5]
        points[:, 0] = 0.5 - points[:, 0] # [-0.5, 0.5]
        # points -= 0.5
        points *= n_voxel * d_voxel # mm

        angle = -1 * angle # inverse direction
        rot_M = np.array([
            [            1,              0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle),  np.cos(angle)],
        ])
        # rot_M = np.array([
        #     [np.cos(angle), -np.sin(angle), 0],
        #     [np.sin(angle),  np.cos(angle), 0],
        #     [             0,               0, 1]
        # ])
        points = points @ rot_M.T

        coeff = (d2) / (d1 - points[:, 0])  # N,
        d_points = points[:, [0, 1]] * (coeff[:, None] if 'cone' in self.geometry.mode else 1.0) # [N, 2] float parallel coeff = 1
        d_points /= (n_detector * d_detector)
        d_points *= 2 # NOTE: some points may fall outside [-1, 1]
        return d_points
    
    def get_points(self, points_3d, angles, layer=0):
        points_2d = []
        for a in angles:
            p = self.project_points(points_3d, a, layer)
            points_2d.append(p)
        points_2d = np.stack(points_2d, axis=0) 
        return points_3d, points_2d
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.split, self.files[idx])
        sample_set = np.load(file_path)
        
        sample = tio.Subject(
            projection=tio.ScalarImage(tensor=to_torch(sample_set['projection'][None])),
            # volume=tio.ScalarImage(tensor=to_torch(sample_set['image'][None]))
        )
        sample = self.transforms(sample)
        sample = {k: getattr(sample, k).data[0] for k in sample}
        sample['angle_in_degrees'] = torch.linspace(0, 360-360/sample['projection'].shape[0], steps=sample['projection'].shape[0], )
        sample['angle_in_radians'] = -sample['angle_in_degrees'] * torch.pi / 180 - torch.pi
        
        selected_angles = torch.randperm(len(sample['angle_in_degrees']))[:self.batch_size] \
            if not self.fixed_angle_per_case else np.array(self.file_angle[idx])[np.random.permutation(len(self.file_angle[idx]))[:self.batch_size]].tolist()
        sample['selected_projection'] = sample['projection'][selected_angles]
        sample['selected_angle_in_degrees'] = sample['angle_in_degrees'][selected_angles]
        sample['selected_angle_in_radians'] = sample['angle_in_radians'][selected_angles]
        
        if self.n_points > 0:
            D, H, W = self.output_size
            d = np.random.randint(0, D)
            y, z = np.meshgrid(np.linspace(0, 1, H, endpoint=False), np.linspace(0, 1, W, endpoint=False))
            points_3d = np.stack([
                np.full(H * W, d / D),
                y.ravel(),
                z.ravel()
            ], axis=1)
            sample[f'points_3d'] = to_torch(points_3d)
            for l in range(self.network_depth):
                points_3d, points_2d = self.get_points(points_3d, sample['selected_angle_in_radians'], layer=l)
                sample[f'points_2d_depth_{l}'] = to_torch(points_2d)
            sample[f'selected_coord'] = torch.tensor(d)
            sample[f'points_2d'] = torch.cat([sample[f'points_2d_depth_{l}'] for l in range(self.network_depth)], dim=0)
        
        sample['casename'] = self.files[idx].split('/')[-1].split('.')[0]
        if self.output_keys is not None: output = {k: sample[k] for k in self.output_keys}
        else: output = sample
        return output

    def collate(self, batch):
        collated = default_collate(batch)
        rearranged = {}
        for k, v in collated.items():
            if not isinstance(v, torch.Tensor):
                rearranged[k] = v
            elif 'proj' in k or 'angle' in k:
                rearranged[k] = v.view(-1, 1, *v.shape[2:])
        return rearranged


class SVCTDatasetRepeat(SVCTDataset):
    def __init__(self, repeat=1, **kw):
        super().__init__(**kw)
        self.repeat = repeat
        self.files = self.files * repeat
        self.file_angle = self.file_angle * repeat if self.fixed_angle_per_case else None
    
    def __len__(self, ):
        return len(self.files)