from torch.utils.data import  Dataset
import sys
sys.path.append('/ailab/user/dailinrui-hdd/code/latentdiffusion/')
from ldm.data.utils import identity, window_norm, TorchioForegroundCropper, TorchioSequentialTransformer, LabelParser, OrganTypeBase
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from inference.utils import image_logger
import torch
import h5py, json
import torchio as tio
import os, numpy as np
from collections import OrderedDict
from functools import reduce, partial


OrganTypes = [
    OrganTypeBase("Background", 0),
    OrganTypeBase("Spleen", 1),
    OrganTypeBase("Kidney", 2),
    OrganTypeBase("Liver", 3),
    OrganTypeBase("Stomach", 4),
    OrganTypeBase("Pancreas", 5),
    OrganTypeBase("Lung", 6),
    OrganTypeBase("SmallBowel", 7),
    OrganTypeBase("Duodenum", 8),
    OrganTypeBase("Colon", 9),
    OrganTypeBase("UrinaryBladder", 10),
    OrganTypeBase("Heart", 11),
    OrganTypeBase("Vertebrae", 12),
    OrganTypeBase("Rib", 13),
    OrganTypeBase("Adrenal", 14),
    OrganTypeBase("PortalVeinAndSplenicVein", 15),
    OrganTypeBase("Esophagus", 16),
    OrganTypeBase("Aorta", 17),
    OrganTypeBase("InferiorVenaCava", 18),
    OrganTypeBase("Gallbladder", 19),
]


class GatheredEnsembleDataset(Dataset):
    def __init__(self, base='/ailab/user/dailinrui-hdd/data/datasets/ensemble', 
                 split="train", 
                 resize_to=(128,128,128), 
                 max_size=None, include_ds=None, include_cases=None):
        self.transforms = {
            "crop": TorchioForegroundCropper(crop_level="mask_foreground", 
                                             crop_anchor="totalseg",
                                             crop_kwargs=dict(foreground_hu_lb=1e-3,
                                                              foreground_mask_label=None,
                                                              outline=(0, 0, 0))),
            "resize": tio.Resize(resize_to) if resize_to is not None else tio.Lambda(identity),
            "norm": tio.Lambda(partial(window_norm, window_pos=0, window_width=2000), include=['image']),
        }
        self.base = base
        self.split = split
        
        self.train_keys = os.listdir(os.path.join(self.base, 'train'))
        self.val_keys = self.test_keys = os.listdir(os.path.join(self.base, 'val'))
        self.split_keys = getattr(self, f"{split}_keys")[:max_size]
        
        with open(os.path.join(base, 'mapping.json')) as f:
            mappings = json.load(f)
        
        if include_cases is not None:
            self.split_keys = [_ for _ in self.split_keys if _ in include_cases]
        else:
            if include_ds is not None:
                self.split_keys = [_ for _ in self.split_keys if reduce(lambda x, y: x | y, [x in mappings[_] for x in include_ds])]
        
    def __len__(self): return len(self.split_keys)
    
    def __getitem__(self, idx):
        sample = h5py.File(os.path.join(self.base, 'train' if self.split == 'train' else 'val', self.split_keys[idx]))
        attrs = sample.attrs
        ds = {k: sample[k][:] for k in sample.keys()}
        ds['prompt_context'] = ds["prompt_context"][0]
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=ds['image']),
                              totalseg=tio.LabelMap(tensor=ds['totalseg']),
                              mask=tio.LabelMap(tensor=ds['mask']))
        subject = self.transforms['crop'](subject)
        subject = self.transforms['resize'](subject)
        subject = self.transforms['norm'](subject)
        subject = self.transforms.get('augmentation', lambda x: x)(subject)
        
        sample = dict(**attrs) | ds
        sample.update({k: getattr(subject, k).data for k in subject.keys()})
        sample.update({"cond": torch.cat([sample['totalseg'], sample['mask']], dim=0)})
        # if sample['mask'].max() > 1: sample['mask'] = (sample['mask'] == 2).float()  # kits
        return sample
    
    
class GatheredEnsembleDatasetForLatentGeneration(Dataset):
    def __init__(self, base='/ailab/user/dailinrui-hdd/data/datasets/ensemble', 
                 split="train", 
                 max_size=None, include_ds=None, include_cases=None):
        self.base = base
        self.split = split
        
        self.train_keys = os.listdir(os.path.join(self.base, 'train'))
        self.val_keys = self.test_keys = os.listdir(os.path.join(self.base, 'val'))
        self.split_keys = getattr(self, f"{split}_keys")[:max_size]
        
        with open(os.path.join(base, 'mapping.json')) as f:
            mappings = json.load(f)
        
        if include_cases is not None:
            self.split_keys = [_ for _ in self.split_keys if _ in include_cases]
        else:
            if include_ds is not None:
                self.split_keys = [_ for _ in self.split_keys if reduce(lambda x, y: x | y, [x in mappings[_] for x in include_ds])]
        
    def __len__(self): return len(self.split_keys)
    
    def __getitem__(self, idx):
        sample = h5py.File(os.path.join(self.base, 'train' if self.split == 'train' else 'val', self.split_keys[idx]))
        attrs = sample.attrs
        ds = {k: sample[k][:] for k in sample.keys()}
        ds['prompt_context'] = ds["prompt_context"][0]
        
        case = {"image_latents": DiagonalGaussianDistribution(torch.tensor(sample['image_latents'][:])).sample()[0]}
        case.update({
            "cond": torch.cat([torch.tensor(sample['totalseg'][:]), torch.tensor(sample['mask'][:])], dim=0),
            "prompt_context": torch.tensor(sample['prompt_context'][:]).float()[0]
        })
        case['cond'] = torch.nn.functional.interpolate(case['cond'][None], size=(32, 128, 128), mode='trilinear')[0]
        # if sample['mask'].max() > 1: sample['mask'] = (sample['mask'] == 2).float()  # kits
        return case

    
class GatheredDatasetForClassification(GatheredEnsembleDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.transforms['norm'] = tio.Lambda(partial(window_norm, window_pos=60, window_width=360), include=['image'])
        self.transforms['augmentation'] = TorchioSequentialTransformer(OrderedDict({
            "first": tio.OneOf({
                tio.RandomAnisotropy(0, downsampling=(1.5, 5), image_interpolation='linear', include=['image']): 2,
                tio.RandomAnisotropy((1,2), downsampling=(1.5, 5), image_interpolation='linear', include=['image']): 2,
                tio.RandomNoise(include=['image']): 1,
                tio.Lambda(identity): 5
            }),
            "second": tio.OneOf({
                tio.RandomGamma(include=['image']): 5,
                tio.Lambda(identity): 5
            })
        }))
        
        
class GatheredDatasetForGeneration(GatheredEnsembleDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.transforms['norm'] = tio.Lambda(partial(window_norm, window_pos=0, window_width=2400), include=['image'])


class GatheredDatasetForMaskGeneration(GatheredEnsembleDataset):
    def __init__(self, num_classes=20, **kw):
        super().__init__(**kw)
        self.transforms['norm'] = tio.RescaleIntensity(in_min_max=(0, num_classes), out_min_max=(0, 1), include=['totalseg'])
        
    def __getitem__(self, idx):
        sample = h5py.File(os.path.join(self.base, 'train' if self.split == 'train' else 'val', self.split_keys[idx]))
        attrs = sample.attrs
        ds = {k: sample[k][:] for k in sample.keys()}
        ds['prompt_context'] = ds["prompt_context"][0]
        
        subject = tio.Subject(totalseg=tio.ScalarImage(tensor=ds['totalseg']),
                              mask=tio.LabelMap(tensor=ds['mask'].astype(np.uint8) if ds['mask'].max() == 1 else (ds['mask'] == 1).astype(np.uint8)))
        subject = self.transforms['crop'](subject)
        subject = self.transforms['resize'](subject)
        subject = self.transforms['norm'](subject)
        subject = self.transforms.get('augmentation', lambda x: x)(subject)
        
        sample = dict(**attrs) | ds
        sample.update({k: getattr(subject, k).data for k in subject.keys()})
        return sample
    
    
class MedSynDataset(GatheredEnsembleDataset):
    def __getitem__(self, idx):
        sample = h5py.File(os.path.join(self.base, 'train' if self.split == 'train' else 'val', self.split_keys[idx]))
        attrs = sample.attrs
        ds = {k: sample[k][:] for k in sample.keys()}
        ds['prompt_context'] = ds["prompt_context"][0]
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=ds['image']),
                              totalseg=tio.LabelMap(tensor=ds['totalseg']),
                              mask=tio.LabelMap(tensor=ds['mask'].astype(np.uint8) if ds['mask'].max() == 1 else (ds['mask'] == 2).astype(np.uint8)))
        subject = self.transforms['crop'](subject)
        subject = self.transforms['resize'](subject)
        subject = self.transforms['norm'](subject)
        subject = self.transforms.get('augmentation', lambda x: x)(subject)
        sample =  {"data": torch.cat([subject.image.data, subject.totalseg.data / 10 - 1, subject.mask.data], dim=0),
                   "prompt_context": torch.tensor(ds['prompt_context'])} | dict(**attrs)
        return sample
    
    
class SemanticSynthesizerDataset(Dataset):
    def __init__(self, 
                 base='/ailab/user/dailinrui-hdd/data/datasets/ensemble',
                 resize_to=(128,)*3,
                 split='train',
                 max_num=None):
        self.base = f"{base}/{split}_v2"
        self.ds = [os.path.join(self.base, f) for f in os.listdir(self.base)][:max_num]
        self.transforms = {
            "crop": TorchioForegroundCropper(crop_level="mask_foreground", 
                                             crop_anchor="totalseg",
                                             crop_kwargs=dict(foreground_hu_lb=1e-3,
                                                              foreground_mask_label=None,
                                                              outline=(0, 0, 0))),
            "resize": tio.Resize(resize_to) if resize_to is not None else tio.Lambda(identity),
            "norm": tio.Lambda(partial(window_norm, window_pos=0, window_width=2000), include=['image']),
        }
        self.parser = LabelParser(totalseg_version='v1')
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        sample = h5py.File(self.ds[idx])
        attrs = sample.attrs
        ds = {k: sample[k][:] for k in sample.keys()}
        ds['prompt_context'] = ds["prompt_context"][0]
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=ds['image']),
                              totalseg=tio.LabelMap(tensor=ds['totalseg']),
                              mask=tio.LabelMap(tensor=ds['mask'].astype(np.uint8)))
        subject = self.transforms['crop'](subject)
        subject = self.transforms['resize'](subject)
        subject = self.transforms['norm'](subject)
        subject = self.transforms.get('augmentation', lambda x: x)(subject)
        totalseg_and_tumorseg = self.parser.totalseg2mask(subject.totalseg.data, OrganTypes)
        totalseg_and_tumorseg[subject.mask.data > 0] = 20
        sample =  {"mask": totalseg_and_tumorseg.long(),
                   "prompt_context": torch.tensor(ds['prompt_context'])} | dict(**attrs)
        return sample
    
    
def gather():
    from tqdm import tqdm
    from pathlib import Path
    from scipy.ndimage import label
    from ldm.modules.encoders.modules import FrozenBERTEmbedder
    from ldm.models.autoencoder import AutoencoderKL
    
    embedder = FrozenBERTEmbedder().cuda()
    encoder = AutoencoderKL(
        ddconfig=dict(
            double_z=False,
            z_channels=4,
            resolution=[128, 512, 512],
            in_channels=1,
            out_ch=1,
            ch=4,
            ch_mult=[1, 1, 2],
            num_res_blocks=2,
            attn_resolutions=[],
            dropout=0.0,
            attn_type='none'
        ),
        lossconfig=dict(
            target='torch.nn.Identity'
        ),
        embed_dim=4,
        dims=3,
        image_key='image',
        ckpt_path='/ailab/user/dailinrui-hdd/data/ldm/aekl_128_512_512/checkpoints/last.ckpt'
    ).cuda()
    transforms = {
        "crop": TorchioForegroundCropper(crop_level="mask_foreground", 
                                            crop_anchor="totalseg",
                                            crop_kwargs=dict(foreground_hu_lb=1e-3,
                                                            foreground_mask_label=None,
                                                            outline=(0, 0, 0))),
        "resize": tio.Resize((128, 512, 512)),
        "norm": tio.Lambda(partial(window_norm, window_pos=0, window_width=2000), include=['image']),
    }
    
    base = '/ailab/user/dailinrui-hdd/data/datasets/ensemble/val'
    for file in tqdm(Path(base).glob('*.h5'), total=len(os.listdir(base))):
        # read h5
        h5 = h5py.File(file, 'r+')
        sample = {'dataset': {k: h5[k][:] for k in h5.keys()}, 'attrs': {k: h5.attrs[k] for k in h5.attrs.keys()}}
        
        # # process mask
        # if sample['dataset']['mask'].max() > 1: sample['dataset']['mask'] = (sample['dataset']['mask'] > 1).astype(sample['dataset']['mask'].dtype)
        # size_tumors = sample['dataset']['mask'].sum()
        # _, n_tumors = label(sample['dataset']['mask'])
        
        # # process prompt
        # prompt = sample['attrs']['prompt'] + f"。该患者肿瘤较{'大' if size_tumors > 40000 else '小'}，共有{n_tumors}个原发肿瘤"
        # feature = embedder(prompt).data.cpu().numpy()
        # sample['dataset']['prompt_context'] = feature
        # sample['attrs']['prompt'] = prompt
        
        # process image to latents
        image_norm = tio.Subject(image=tio.ScalarImage(tensor=sample['dataset']['image']),
                                 totalseg=tio.ScalarImage(tensor=sample['dataset']['totalseg']),)
        image_norm = transforms['crop'](image_norm)
        image_norm = transforms['resize'](image_norm)
        image_norm = transforms['norm'](image_norm)
        image_norm = image_norm.image.data[None].cuda()
        # image_latents = encoder.encode_to_latents(image_norm)
        image_latents = torch.tensor(sample['dataset']['image_latents']).cuda()
        sample['dataset']['image_latents'] = image_latents.cpu().data.numpy()
        
        # image_logger({'image': image_norm.cpu(),
        #               'image_latents': image_latents.cpu(),
        #               'image_latents_recon': encoder.decode_from_latents(image_latents).cpu(),
        #               'image_latents_recon_ds': encoder.decode(DiagonalGaussianDistribution(image_latents).sample()).cpu(),}, path='./test.png', log_separate=False)
        
        # modify h5
        # for dataset in sample['dataset']:
        #     h5[dataset][...] = sample['dataset'][dataset]
        # for attr in sample['attrs']:
        #     h5.attrs[attr] = sample['attrs'][attr]
        if not 'image_latents' in h5:
            h5.create_dataset(name='image_latents', data=sample['dataset']['image_latents'])
        else:
            h5['image_latents'][...] = sample['dataset']['image_latents']
        h5.close()


if __name__ == "__main__":
    # sbatch -D $(pwd) -J pp -o ./outs/pp.txt -p smart_health_02 -N 1 -n 1 --cpus-per-task=16 --gpus=4 --mem=128G --wrap "python /ailab/user/dailinrui-hdd/code/latentdiffusion/ldm/data/guidegen.py"
    # sbatch -D $(pwd) -J msk -o ./outs/masksyn.txt -p smart_health_02 -N 1 -n 1 --cpus-per-task=24 --gpus=8 --mem=400G --wrap "python /ailab/user/dailinrui-hdd/code/latentdiffusion/main.py -t --base /ailab/user/dailinrui-hdd/code/latentdiffusion/configs/categorical-diffusion/rebuttal.yaml --debug --name guidegen_tcss_21label"
    gather()