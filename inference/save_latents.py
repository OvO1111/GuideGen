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


def maybe_mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


def main():
    from leaptorch import Projector
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/autoencoder/ae.yaml",
        help="Path to the config file for the dataset.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/autoencoder/ae.ckpt",
        help="Path to the checkpoint file for the model.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        help="Directory to save the dataset.",
    )
    
    args = parser.parse_args()
    config = omegaconf.OmegaConf.load(args.config)
    
    config.data.params.train['max_size'] = None
    config.data.params.validation['max_size'] = None
    # config.data.params.test['max_size'] = None
    train_dataset: th.utils.data.Dataset = instantiate_from_config(config.data.params.train)
    val_dataset: th.utils.data.Dataset = instantiate_from_config(config.data.params.validation)
    test_dataset: th.utils.data.Dataset = instantiate_from_config(config.data.params.test)
    
    config.model['target'] = config.model.train_target
    config.model['params'].update(config.model['test_only_params'])
    model: AutoencoderKL = instantiate_from_config(config.model).cuda()
    
    dataloader = th.utils.data.DataLoader(
        chain(train_dataset, val_dataset, test_dataset), 
        batch_size=1, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )
    output_dir_train = maybe_mkdir(os.path.join(args.save_root, 'train'))
    output_dir_val = maybe_mkdir(os.path.join(args.save_root, 'val'))
    output_dir_test = maybe_mkdir(os.path.join(args.save_root, 'test'))

    iterator = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, batch in iterator:
        save_dict = {}
        # get latents from image
        inputs = batch[model.image_key].cuda()

        if model.is_conditional: conditions = model.get_input(batch, model.cond_key)
        else: conditions = None
        posterior = model.encode_to_latents(inputs, conditions)
        save_dict['posterior'] = posterior[0].cpu()

        if i < len(train_dataset):
            output_dir = output_dir_train
        elif i < len(train_dataset) + len(val_dataset):
            output_dir = output_dir_val
        else:
            output_dir = output_dir_test
        save_path = os.path.join(output_dir,
                                f"{'_'.join(batch['casename'][0].split('/')[-2:])}".split('.')[0] + '.npz')
        np.savez(save_path, **save_dict)
        iterator.set_postfix_str(f"saved to {save_path}")


main()