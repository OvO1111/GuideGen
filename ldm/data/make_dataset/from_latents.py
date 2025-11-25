import omegaconf, sys
import torchio as tio
import os, torch as th, numpy as np
import SimpleITK as sitk, pytorch_lightning as pl

sys.path.append('/home/dlr/code/diffusion/')
from argparse import ArgumentParser
from ldm.util import instantiate_from_config
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from main import DataModuleFromConfig
from itertools import chain
from tqdm import tqdm

from ldm.data.make_dataset.settings import *


def move_to_device(tensor, device):
    if isinstance(tensor, th.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, np.ndarray):
        return th.tensor(tensor, device=device)
    elif isinstance(tensor, list):
        return [move_to_device(item, device) for item in tensor]
    elif isinstance(tensor, dict):
        return {key: move_to_device(value, device) for key, value in tensor.items()}
    elif isinstance(tensor, th.nn.Module):
        return tensor.to(device).eval()
    else:
        raise TypeError(f"Unsupported type: {type(tensor)}")
    

def move_from_device(tensor):
    if isinstance(tensor, th.Tensor):
        return tensor.cpu().data.numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        return [move_from_device(item) for item in tensor]
    elif isinstance(tensor, dict):
        return {key: move_from_device(value) for key, value in tensor.items()}
    else:
        raise TypeError(f"Unsupported type: {type(tensor)}")
    
    
def maybe_mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


def main(split='train', get=['image', 'projection']):
    from leaptorch import Projector
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/autoencoder/2dproj.yaml",
        help="Path to the config file for the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/dlr/data/datasets/synthetic/projs_set0825",
        help="Directory to save the dataset.",
    )
    
    args = parser.parse_args()
    config = omegaconf.OmegaConf.load(args.config)
    
    hypers = eval(f"geometry{args.output_dir.split('projs_set')[-1]}")
    config.data.params.train['max_size'] = None
    config.data.params.validation['max_size'] = None
    # config.data.params.test['max_size'] = None
    train_dataset: th.utils.data.Dataset = instantiate_from_config(config.data.params.train)
    val_dataset: th.utils.data.Dataset = instantiate_from_config(config.data.params.validation)
    # test_dataset: th.utils.data.Dataset = instantiate_from_config(config.data.params.test)
    
    config.model['target'] = config.model.train_target
    config.model['params'].update(config.model['test_only_params'])
    model: AutoencoderKL = instantiate_from_config(config.model)
    
    model = move_to_device(model, th.device('cuda:0'))
    
    dataset = locals().get(f"{split}_dataset")
    dataset.transforms = tio.Compose([
        tio.Resize(hypers['n_voxel'], image_interpolation='linear'),
        tio.CropOrPad(hypers['n_voxel'], only_crop=True),
        tio.Resize(hypers['n_voxel'], image_interpolation='linear')
    ])
    dataloader = th.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )
    output_dir = maybe_mkdir(os.path.join(args.output_dir, split))
    
    projector = Projector(forward_project=True, use_static=False, use_gpu=True, gpu_device=th.device('cuda:0'), batch_size=1)
    getattr(projector, hypers['mode'])(
        numAngles=hypers['total_angles'],
        numRows=hypers['n_detector'][0], 
        numCols=hypers['n_detector'][1],
        pixelHeight=hypers['s_detector'][0],
        pixelWidth=hypers['s_detector'][1],
        centerRow=hypers['n_detector'][0] // 2,
        centerCol=hypers['n_detector'][1] // 2,
        phis=(
                np.linspace(0, hypers.get('max_angle', 360), hypers['total_angles'])
            + hypers["start_angle"]
        ).astype(np.float32),
        # sod=hypers['distance_source_origin'],
        # sdd=hypers['distance_source_detector'],
    )
    projector.set_volume(
        numX=hypers['n_voxel'][2],
        numY=hypers['n_voxel'][1],
        numZ=hypers['n_voxel'][0],
        voxelWidth=hypers['s_voxel'][0],
        voxelHeight=hypers['s_voxel'][1],
        offsetX=hypers['offset_origin'][0],
        offsetY=hypers['offset_origin'][1],
        offsetZ=hypers['offset_origin'][2],
    )

    iterator = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, batch in iterator:
        save_dict = {}
        # get latents from image
        inputs = move_to_device(batch[model.image_key], th.device('cuda:0'))
        if 'image' in get:
            if model.is_conditional: conditions = model.get_input(batch, model.cond_key)
            else: conditions = None
            posterior = model.encode_to_latents(inputs, conditions)
            # save_dict['image'] = move_from_device(inputs)[0]
            save_dict['posterior'] = move_from_device(posterior)[0]
        
        # get projections from image
        if 'projection' in get:
            projs = projector(inputs[0] + th.abs(inputs[0].min()))
            save_dict['image'] = move_from_device(inputs[0, 0])
            save_dict['projection'] = move_from_device(projs)[0, ..., ::-1, ::-1]
            save_dict['maximum'] = inputs[0].max().item()
            save_dict['minimum'] = inputs[0].min().item()
            save_dict["casename"] = np.array(batch["casename"])
        
        # save_dict = {
        #     "image": move_from_device(inputs)[0],
        #     "posterior": move_from_device(posterior)[0],
        #     "projection": move_from_device(projs)[0, ..., ::-1, ::-1],
        # }
        save_path = os.path.join(output_dir, f"{'_'.join(batch['casename'][0].split('/')[-2:])}".split('.')[0] + '.npz')
        np.savez(save_path, **save_dict)
        iterator.set_postfix_str(f"saved to {save_path}")
        
        
class ProjectionNorm:
    def __init__(self, pm=0.5e5, pM=3.5e5, om=-1, oM=1):
        self.pm = pm
        self.pM = pM
        self.om = om
        self.oM = oM

    def forward(self, x, pm=None, pM=None, om=None, oM=None):
        # from the result of stats: the 1-sigma interval of proj is roughly [pm, pM]
        # normalize this interval to [-1, 1] to be compatible with the normal noise
        pm = pm if pm is not None else self.pm
        pM = pM if pM is not None else self.pM
        om = om if om is not None else self.om
        oM = oM if oM is not None else self.oM
        x = (x - pm) / (pM - pm) * (oM - om) + om
        return x
    
    def backward(self, x, pm=None, pM=None, om=None, oM=None):
        pm = pm if pm is not None else self.pm
        pM = pM if pM is not None else self.pM
        om = om if om is not None else self.om
        oM = oM if oM is not None else self.oM
        x = (x - om) / (oM - om) * (pM - pm) + pm
        return x
    
    def disabled(self):
        self.pm, self.pM = 0, 1
        self.om, self.oM = 0, 1
        return self
        
        
class LatentsDataset(th.utils.data.Dataset):
    def __init__(self,
                 data_dir='/home/dlr/data/datasets/synthetic/projs_set0610',
                 split='train',
                 max_size=None,
                 shape=None,   # (o_ctxt, o_trgt, o_final)
                 normalizer=None,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.shape = list(shape) if shape is not None else None
        self.files = [f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.npz')][:max_size]
        
        self.transforms = {}
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.split, self.files[idx])
        data = np.load(file_path)
        sample = {}
        # only projection
        projection = th.tensor(data['projection'], dtype=th.float32)
        
        if self.shape is not None:
            index = th.randperm(projection.shape[0])[:sum(self.shape)]
            sample['o_ctxt'] = projection[index[:self.shape[0]]]
        else:
            index = th.arange(0, 360)
        sample['index'] = index
        sample['index_raw'] = index
        sample['o'] = projection[index]
        
        sample['casename'] = self.files[idx].split('.')[0]
        return sample

        
    
if __name__ == "__main__":
    main(split='train', get=['projection'])
    main(split='val', get=['projection'])
    # ds = LatentsDataset()
    # debug = 1