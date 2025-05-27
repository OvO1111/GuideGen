import os, glob, numpy as np

import json, torch, torchio as tio
from functools import reduce
from omegaconf.omegaconf import DictConfig, ListConfig
from SimpleITK import GetArrayFromImage, GetImageFromArray, ReadImage
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    RandSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
)

from monai.data.dataset import CacheDataset
from torch.utils.data import default_collate, Dataset
from ldm.data.utils import LabelParser, OrganTypeBase, TorchioForegroundCropper


def read_text(path, process_fn=lambda x: x):
    if path.endswith('json'):
        with open(path) as f:
            text = json.load(f)
    elif reduce(lambda x, y: x | y, [path.endswith(x) for x in ['txt', 'list', 'out']]):
        with open(path) as f:
            text = "\n".join(f.readlines()).strip()
    return process_fn(text)


class SimpleDataset:
    def __init__(self, train_dict=None, val_dict=None, test_dict=None, any_dict=None,
                 split='train', output_keys=['image', 'label'], image_keys=['image'], label_keys=[], 
                 max_size=None, use_aug=False, cache_num=0, patch_size=[128, 128, 128], **kw):
        """
        makes a dataset dict from
        image: [$path1, $path2, ...]
        label: [$path1, $path2, ...]
        """
        self.train_dict = train_dict
        self.val_dict = val_dict
        self.test_dict = test_dict
        if self.train_dict is None and self.val_dict is None and self.test_dict is None:
            assert any_dict is not None, "at least give it one list to work with"
            self.__dict__[f"{split}_dict"] = any_dict
        
        self.split = split
        self.use_aug = use_aug
        self.patch_size = patch_size
        self.output_keys = output_keys
        assert split in ['train', 'val', 'test']
        if isinstance(self.__dict__[f"{split}_dict"], (dict, DictConfig)):
            _dict = {}
            for k, v in self.__dict__[f"{split}_dict"].items():
                with open(v) as f:
                    _dict[k] = [_.strip() for _ in f.readlines()]
            self.__dict__[f"{split}_dict"] = _dict
        else:
            with open(v) as f:
                self.__dict__[f"{split}_dict"] = json.load(self.__dict__[f"{split}_dict"])
        
        self.split_list = []
        output_keys = [key for key in output_keys if key in self.__dict__[f"{split}_dict"].keys()]
        if len(image_keys) == 0: 
            image_keys = [key for key in self.output_keys if 'image' in key or 'img' in key]
        if len(label_keys) == 0:
            label_keys = [key for key in self.output_keys if 'label' in key or 'seg' in key or 'mask' in key]
        for args in zip(*self.__dict__[f"{split}_dict"].values()):
            self.split_list.append({ik: iv if ik in image_keys + label_keys else read_text(iv) for ik, iv in zip(self.__dict__[f"{split}_dict"].keys(), args)})
            
        assert reduce(lambda x, y: x | y, [x in self.__dict__[f"{split}_dict"] for x in output_keys])
        
        if max_size is not None:     
            self.__dict__[f"{split}_dict"] = {k: v for ikv, (k, v) in enumerate(self.__dict__[f"{split}_dict"], 1) if ikv <= max_size}
        preprocess = self.get_preprocess(image_keys, label_keys, **kw)
        
        self.parser = LabelParser(totalseg_version="v1")
        self.dataset = CacheDataset(self.split_list, transform=preprocess, cache_num=cache_num, num_workers=8)
        
    def get_preprocess(self, image_keys, label_keys, **kw):
        if not isinstance(image_keys, (list, ListConfig)): image_keys = [image_keys]
        if not isinstance(label_keys,  (list, ListConfig)): label_keys = [label_keys]
        transforms = []
        transforms.extend([
            LoadImaged(keys=image_keys + label_keys, allow_missing_keys=True),
            EnsureChannelFirstd(keys=image_keys + label_keys, channel_dim="no_channel", allow_missing_keys=True),
            # Orientationd(keys=image_keys + label_keys, axcodes="RAS"),
            Spacingd(keys=image_keys + label_keys, pixdim=[1, 1, 1], allow_missing_keys=True,
                     mode=["bilinear" if key in image_keys else "nearest" for key in image_keys + label_keys]),
            CropForegroundd(keys=image_keys + label_keys, source_key=kw.get("crop_by_fg_key", image_keys[0]), 
                            k_divisible=self.patch_size, allow_missing_keys=True, mode='minimum'),
            # SpatialPadd(keys=image_keys + label_keys, spatial_size=self.patch_size, mode='constant', allow_missing_keys=True),
            ScaleIntensityRanged(keys=image_keys, a_min=kw.get('window_min', -1000), a_max=kw.get('window_max', 1000), b_min=-1, b_max=1)
        ])
        if self.split == "train" and self.use_aug:
            transforms.extend([
                RandRotate90d(keys=image_keys + label_keys, prob=0.10, max_k=3, allow_missing_keys=True),
                RandShiftIntensityd(keys=image_keys, offsets=0.1, prob=0.2, allow_missing_keys=True),
                RandZoomd(keys=label_keys + image_keys, prob=0.1, allow_missing_keys=True),
                RandFlipd(keys=image_keys + label_keys, prob=0.1, allow_missing_keys=True)
            ])
            if self.patch_size is not None:
                transforms.append(RandSpatialCropd(keys=image_keys + label_keys, roi_size=self.patch_size, allow_missing_keys=True, random_center=True))
        elif self.split == "train":
            transforms.append(
                RandCropByPosNegLabeld(keys=image_keys + label_keys, label_key=kw.get("crop_by_label_key", label_keys[0]),
                                       spatial_size=self.patch_size, pos=kw.get("crop_by_label_seqclass", [2])[-1], neg=kw.get("crop_by_label_seqclass", [1])[0])
            )
        else:
            # transforms.append(
            #     RandCropByLabelClassesd(keys=image_keys + label_keys, label_key=kw.get("crop_by_label_key", label_keys[0]),
            #                             spatial_size=self.patch_size, ratios=kw.get("crop_by_label_seqclass", [0, 1]), num_classes=kw.get("crop_by_label_nclass", 2)),
            # )
            transforms.append(RandSpatialCropd(keys=image_keys + label_keys, roi_size=self.patch_size, allow_missing_keys=True, random_center=True))
        transforms.append(ToTensord(keys=image_keys + label_keys, allow_missing_keys=True))
        return Compose(transforms)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        trunc_sample = {k: v for k, v in sample.items() if k in self.output_keys}
        return trunc_sample
    
    def collate(self, batch):
        return default_collate(batch)
    


class DummyDataset(Dataset):
    def __init__(self, output_size, **kw):
        self.output_size = tuple(output_size)
        super().__init__()
    
    def __len__(self):
        return 100
    
    def __getitem__(self, _):
        output_size = (1,) + self.output_size if len(self.output_size) == 3 else self.output_size
        return {"image": torch.ones(output_size).float(),
                "label": torch.ones((1,) + self.output_size[1:]).long(),
                "text": "this is a dummy dataset",
                "aux": torch.ones((len("this is a dummy dataset"), 768)).float()}
        
        
class MSDDataset(Dataset):
    def __init__(self, base_dir, split='train',
                 output_size=(128, 512, 512), max_size=None):
        self.split = split
        self.base_dir = base_dir
        self.ds = {'train': [], 'val': [], 'test': []}
        super().__init__()
        
        if isinstance(self.base_dir, str):
            self.base_dir = [self.base_dir]
            
        for dir in self.base_dir:
            with open(os.path.join(dir, 'train.txt')) as f1, open(os.path.join(dir, 'val.txt')) as f2, open(os.path.join(dir, 'test.txt')) as f3:
                self.ds['train'].extend([os.path.join(dir, x.strip()) for x in f1.readlines()])
                self.ds['val'].extend([os.path.join(dir, x.strip()) for x in f2.readlines()])
                self.ds['test'].extend([os.path.join(dir, x.strip()) for x in f3.readlines()])
        
        self.ds = {k: [_.strip() for _ in v] for k, v in self.ds.items()}
        self.split_ds = self.ds[self.split][:max_size]
        
        self.load_fn = lambda x: GetArrayFromImage(ReadImage(x))
        self.transforms = {
            "resize": tio.Resize(output_size, image_interpolation='trilinear', label_interpolation='nearest'),
        }
        
    def __len__(self):
        return len(self.split_ds)
    
    def __getitem__(self, idx):
        path = self.split_ds[idx]
        dirname, basename = os.path.dirname(path), os.path.basename(path)
        if basename.endswith('.nii.gz'): basename = basename[:-7]
        
        image = self.load_fn(os.path.join(dirname, 'imagesTr', basename + '_0000.nii.gz'))
        label = self.load_fn(os.path.join(dirname, 'labelsTr', basename + '.nii.gz'))
        
        sample = {"image": image, "label": label}
                
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.tensor(sample['image'][None].astype(np.float32))),
            label=tio.LabelMap(tensor=torch.tensor(sample['label'][None].astype(np.int64))),
        )
        subject = self.transforms(subject)
        sample = {k: v.data for k, v in subject.items()}
        
        return sample
        