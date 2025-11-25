import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy
import random
import numpy as np
from einops import rearrange, repeat
from ldm.models.template import BasePytorchLightningTrainer
from ldm.modules.diffusionmodules.model import Downsample, Upsample, make_attn, checkpoint
from torch.optim.lr_scheduler import LinearLR, LambdaLR
from scipy.ndimage import label
import medpy.metric.binary as bin

from monai.networks.nets.unet import UNet 
from monai.networks.nets.vnet import VNet
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.basic_unetplusplus import BasicUNetPlusPlus
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.nets.dynunet import DynUNet


class Segmentator(BasePytorchLightningTrainer):
    def __init__(self,
                 num_classes,
                 image_key="image",
                 seg_key="mask",
                 backbone_name=None,
                 in_chns=1,
                 crop_input=False,
                 ckpt_path=None,
                 ignore_keys=[],
                 only_load_model=True,
                 monitor='val/loss',
                 dims=3,
                 image_size=(128, 128, 128)):
        super().__init__()
        self.dims = dims
        self.image_key = image_key
        self.seg_key = seg_key
        self.crop_input = crop_input
        self.image_size = image_size
        self.num_classes = num_classes
        self.in_chns = in_chns
        self.backbone_name = backbone_name
        self._get_model(backbone_name)
        
        self.monitor = monitor
        if ckpt_path is not None: self.init_from_ckpt(ckpt_path, ignore_keys, only_load_model)
        
    def get_input(self, batch, *keys):
        images = [batch[k] for k in keys]
        if self.crop_input:
            if self.training:
                return self.get_input_train(images)
            else:
                return self.get_input_val(images)
        return images
        
    def get_input_val(self, images):
        # return generator of patches
        image = images[0]
        (b, c, w, h, d) = image.shape
        outline = [self.image_size[i] * (image.shape[2 + i] // self.image_size[i] + 1) - image.shape[2 + i] for i in range(3)]
        padder = (lambda x: x) if sum(outline) == 0 else lambda x: torch.nn.functional.pad(x, (0, outline[2], 0, outline[1], 0, outline[0]), mode='constant', value=0)
        
        for i in range(0, w, self.image_size[0]):
            for j in range(0, h, self.image_size[1]):
                for k in range(0, d, self.image_size[2]):
                    patch = [padder(im)[:, :, i: i + self.image_size[0], j: j + self.image_size[1], k: k + self.image_size[2]] for im in images]
                    yield patch, (i, j, k)
                    
    def _patch_cropper(self, _image, _mode="random", _pad_value=0, _hazy=False):
        _image = _image[0]
        _output_size = self.image_size
        # maybe pad image if _output_size is larger than _image.shape
        ph = max((_output_size[0] - _image.shape[1]) // 2 + 3, 0)
        pw = max((_output_size[1] - _image.shape[2]) // 2 + 3, 0)
        pd = max((_output_size[2] - _image.shape[3]) // 2 + 3, 0)
        padder = lambda x: torch.nn.functional.pad(x, (pd, pd, ph, ph, pw, pw), mode='constant', value=0)
        _image = padder(_image)
        # padder = identity
            
        if _mode == "random":
            h_center = random.randint(ph, _image.shape[1] - ph)
            w_center = random.randint(pw, _image.shape[2] - pw)
            d_center = random.randint(pd, _image.shape[3] - pd)

        elif _mode == "foreground":
            if _image.sum() == 0: return self._patch_cropper(_image, _output_size, "random", _pad_value)
            
            _hl, _hr = torch.where(torch.any(torch.any(_image, 2), 2))[-1][[0, -1]]
            _wl, _wr = torch.where(torch.any(torch.any(_image, 1), -1))[-1][[0, -1]]
            _dl, _dr = torch.where(torch.any(torch.any(_image, 1), 1))[-1][[0, -1]]
            if _hazy: _label = _image
            else:
                _label, _n = label(_image[:, _hl: _hr + 1, _wl: _wr + 1, _dl: _dr + 1].data.cpu().numpy())
                _label = torch.tensor(_label == random.randint(1, _n), dtype=_image.dtype, device=_image.device)
            hl, hr = torch.where(torch.any(torch.any(_label, 2), 2))[-1][[0, -1]] + _hl
            wl, wr = torch.where(torch.any(torch.any(_label, 1), -1))[-1][[0, -1]] + _wl
            dl, dr = torch.where(torch.any(torch.any(_label, 1), 1))[-1][[0, -1]] + _dl
            hc, hd = (hl + hr) // 2, hr - hl + 1
            wc, wd = (wl + wr) // 2, wr - wl + 1
            dc, dd = (dl + dr) // 2, dr - dl + 1
            
            h_center = random.randint(hc - hd // 4, max(hc + hd // 4, hc - hd // 4 + 1))
            w_center = random.randint(wc - wd // 4, max(wc + wd // 4, wc - wd // 4 + 1))
            d_center = random.randint(dc - dd // 4, max(dc + dd // 4, dc - dd // 4 + 1))
        
        h_left = max(0, h_center - _output_size[0] // 2)
        h_right = min(_image.shape[1], h_center + _output_size[0] // 2)
        h_offset = (max(0, _output_size[0] // 2 - h_center), max(0, h_center + _output_size[0] // 2 - _image.shape[1]))
        w_left = max(0, w_center - _output_size[1] // 2)
        w_right = min(_image.shape[2], w_center + _output_size[1] // 2)
        w_offset = (max(0, _output_size[1] // 2 - w_center), max(0, w_center + _output_size[1] // 2 - _image.shape[2]))
        d_left = max(0, d_center - _output_size[2] // 2)
        d_right = min(_image.shape[3], d_center + _output_size[2] // 2) 
        d_offset = (max(0, _output_size[2] // 2 - d_center), max(0, d_center + _output_size[2] // 2 - _image.shape[3]))
        cropper = lambda x: x[:, :, h_left: h_right, w_left: w_right, d_left: d_right]
        post_padder = lambda x: torch.nn.functional.pad(x, 
                                                        (*d_offset, *w_offset, *h_offset),
                                                        mode='constant', value=_pad_value)
            
        return padder, cropper, post_padder 
        
    def get_input_train(self, images, fg_prob=0.7):
        mode = "foreground" if random.random() < fg_prob else "random"
        padder, cropper, post_padder = self._patch_cropper(images[1], _mode=mode, _pad_value=0)  # index 0 is image, 1 is seg
        inputs = [post_padder(cropper(padder(im))) for im in images]
        return inputs if len(inputs) > 1 else inputs[0]
        
    def _get_model(self, model_name,):
        in_channels = self.in_chns
        
        if model_name == 'unet':
            self.model = UNet(self.dims, in_channels, self.num_classes, 
                              channels=(16, 32, 64, 128, 256),
                              strides=(2, 2, 2, 2))
        elif model_name == 'vnet':
            self.model = VNet(self.dims, in_channels, self.num_classes)
        elif model_name == 'unetr':
            self.model = UNETR(spatial_dims=self.dims,
                               feature_size=64,
                               in_channels=in_channels,
                               out_channels=self.num_classes,
                               img_size=self.image_size)
        elif model_name == 'swinunetr':
            self.model = SwinUNETR(img_size=self.image_size,
                                   feature_size=48,
                                   num_heads=(4, 8, 16, 32),
                                   in_channels=in_channels,
                                   out_channels=self.num_classes,)
        elif model_name == 'unetpp':
            self.model = BasicUNetPlusPlus(self.dims, in_channels, self.num_classes,)
        
    
    def _multiclass_metrics(self, x, y, prefix=""):
        logs = {}
        for m in ["dc"]:#, "hd95"]:
            for i in range(1, self.num_classes):
                if (x == i).sum() == 0 or (y == i).sum() == 0: result = 0
                else: result = getattr(bin, m, lambda *a: 0)(x == i, y == i)
                logs[f"{prefix}/{m}/{i}"] = result
            # logs[f"{prefix}/{m}/mean"] = sum([logs[f"{prefix}/{m}/{j}"] for j in range(1, self.num_classes)]) / (self.num_classes - 1)
        return logs
    
    def _dice_loss(self, x, y, is_y_one_hot=False, num_classes=-1):
        smooth = 1e-5
        x = x.softmax(1)
        if not is_y_one_hot:
            y = rearrange(F.one_hot(y, num_classes), "b ... c -> b c ...")
        intersect = torch.sum(x * y)
        x_sum = torch.sum(x * x)
        y_sum = torch.sum(y * y)
        loss = 1 - (2 * intersect) / (x_sum + y_sum + smooth)
        return loss
    
    def _ce_loss(self, x, y, **kw):
        return F.cross_entropy(x, y, **kw)
    
    def get_loss(self, preds, y=None, one_hot_y=False):
        ce_loss = self._ce_loss(preds, y)
        dice_loss = self._dice_loss(preds, y, one_hot_y, self.num_classes) 
            
        loss = ce_loss + dice_loss
        return loss * 10
    
    def training_step(self, batch, batch_idx):
        loss_dict = {}
        image, seg = self.get_input(batch, self.image_key, self.seg_key)
        seg = seg[:, 0].long()
        prefix = "train" if self.training else "val"
        model_outputs = self.model(image)
        if self.backbone_name == 'unetpp': model_outputs = model_outputs[-1]
                
        loss = self.get_loss(model_outputs, seg, one_hot_y=0)
        loss_dict[f"{prefix}/loss"] = loss
        # metrics
        metric_dict = self._multiclass_metrics(model_outputs.argmax(1).cpu().numpy(),
                                               seg.cpu().numpy(),
                                               prefix)
        metric_dict['train/dc/mean'] = np.mean([metric_dict[f"train/dc/{i}"] for i in range(1, self.num_classes)])
        # metric_dict['train/hd95/mean'] = [metric_dict[f"train/hd95/{i}"] for i in range(1, self.num_classes)]
        self.log('train.dc.mean', metric_dict['train/dc/mean'], prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(metric_dict, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("global_step", self.global_step, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss_dict = {}
        metric_dict = {}
        
        for itr, ((image, seg), vertex) in enumerate(self.get_input(batch, self.image_key, self.seg_key)):
            seg = seg[:, 0].long()
            prefix = "train" if self.training else "val"
            model_outputs = self.model(image)
            if self.backbone_name == 'unetpp': model_outputs = model_outputs[-1]
                    
            loss = self.get_loss(model_outputs, seg, one_hot_y=0)
            loss_dict[f"{prefix}/loss"] = loss
            # metrics
            iter_metric = self._multiclass_metrics(model_outputs.softmax(1).argmax(1).cpu().numpy(),
                                                    seg.cpu().numpy(),
                                                    prefix)
            for k, v in iter_metric.items():
                if k not in metric_dict: metric_dict[k] = v
                else: metric_dict[k] = (metric_dict[k] * itr + v) / (itr + 1)
        
        metric_dict['val/dc/mean'] = np.mean([metric_dict[f"val/dc/{i}"] for i in range(1, self.num_classes)])
        # metric_dict['train/hd95/mean'] = [metric_dict[f"train/hd95/{i}"] for i in range(1, self.num_classes)]
        self.log('val.dc.mean', metric_dict['val/dc/mean'], prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)      
        self.log_dict(metric_dict, logger=True, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("global_step", self.global_step, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss
    
    def log_images(self, batch, **kw):
        logs = {}
        
        if self.training or not self.crop_input:
            image, seg = self.get_input(batch, self.image_key, self.seg_key)
            seg = seg.long()
            
            logs["inputs"] = image
            logs["gt"] = seg
            
            model_outputs = self.model(image)
            logs["seg"] = model_outputs.softmax(1).argmax(1, keepdim=True)
        else:
            images = torch.zeros_like(batch[self.image_key]) 
            segs = torch.zeros_like(batch[self.seg_key]) 
            model_outputs = torch.zeros_like(batch[self.seg_key]) 
            
            for (image, seg), vertex in self.get_input(batch, self.image_key, self.seg_key):   
                seg = seg.long()
                images[:, :, 
                    vertex[0]: vertex[0] + self.image_size[0], 
                    vertex[1]: vertex[1] + self.image_size[1], 
                    vertex[2]: vertex[2] + self.image_size[2]] =\
                    image[:, :, :min(vertex[0] + self.image_size[0], images.shape[-3]) - vertex[0], :min(vertex[1] + self.image_size[2], images.shape[-2]) - vertex[1], :min(images.shape[-1], vertex[2] + self.image_size[2]) - vertex[2]]
                segs[:, :, 
                    vertex[0]: vertex[0] + self.image_size[0], 
                    vertex[1]: vertex[1] + self.image_size[1], 
                    vertex[2]: vertex[2] + self.image_size[2]] =\
                    seg[:, :, :min(vertex[0] + self.image_size[0], images.shape[-3]) - vertex[0], :min(vertex[1] + self.image_size[2], images.shape[-2]) - vertex[1], :min(images.shape[-1], vertex[2] + self.image_size[2]) - vertex[2]]
                model_outputs[:, :, 
                    vertex[0]: vertex[0] + self.image_size[0], 
                    vertex[1]: vertex[1] + self.image_size[1], 
                    vertex[2]: vertex[2] + self.image_size[2]] =\
                    self.model(image).argmax(1, keepdim=True)[:, :, :min(vertex[0] + self.image_size[0], images.shape[-3]) - vertex[0], :min(vertex[1] + self.image_size[2], images.shape[-2]) - vertex[1], :min(images.shape[-1], vertex[2] + self.image_size[2]) - vertex[2]]
                

            logs["inputs"] = images
            logs['seg'] = model_outputs
            logs['gt'] = segs
        return logs
    
    def configure_optimizers(self,):
        opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0001)
        sch = LinearLR(opt, start_factor=1, end_factor=0, total_iters=self.trainer.max_epochs, verbose=1)
        return [opt], [sch]
    
    
def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def normalization(channels):
    return nn.GroupNorm(32, channels)


def zero_module(module, enabled=True):
    """
    Zero out the parameters of a module and return it.
    """
    if enabled:
        for p in module.parameters():
            p.detach().zero_()
    return module


def checkpoint(func, inputs, params, flag):
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) if torch.is_tensor(x) else x for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) if torch.is_tensor(x) else x for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_tensors_without_none = [x for x in ctx.input_tensors if torch.is_tensor(x)]
        input_tensors_indices = [ix for ix, x in enumerate(ctx.input_tensors) if not torch.is_tensor(x)]
        input_grads = torch.autograd.grad(
            output_tensors,
            input_tensors_without_none + [x for x in ctx.input_params if x.requires_grad],
            output_grads,
            allow_unused=True,
        )
        output_grads = [None] * (len(ctx.input_tensors) + len(ctx.input_params))
        ii = 0
        for ix in range(len(ctx.input_tensors) + len(ctx.input_params)):
            if ix not in input_tensors_indices and (ctx.input_tensors + ctx.input_params)[ix].requires_grad:
                output_grads[ix] = input_grads[ii]
                ii += 1
            else: output_grads[ix] = None

        output_grads = tuple(output_grads)
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + output_grads
    
    
class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (2, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)
    
    
class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2] * 2, x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
        

class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        h = self.out_layers(h)
        return self.skip_connection(x) + h
        
        
class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        use_checkpoint=True,
        resblock_updown=False,
        use_zero_module=True,
    ):
        super().__init__()

        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.input_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                ch = mult * model_channels
                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.input_block_chans = copy.deepcopy(input_block_chans)
        self.middle_block = nn.Sequential(
            ResBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
            ),
            ResBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                ch = model_channels * mult
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1), enabled=use_zero_module),
        )

    def forward(self, x):
        hs = []
        h = x.type(x.dtype)
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        z = self.middle_block(h)
        for module in self.output_blocks:
            z = torch.cat([z, hs.pop()], dim=1)
            z = module(z)
        z = z.type(x.dtype)
        z = self.out(z)
        return z