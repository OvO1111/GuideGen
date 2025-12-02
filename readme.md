# Official Implementation of GuideGen: A Text-Guided Framework for Full-torso Anatomy and CT Volume Generation (AAAI-26)

[[Arxiv paper](https://arxiv.org/abs/2403.07247)]

Steps to run the experiments:
- prepare env:
`conda env create -f environment.yml`
- train models (follow the instructions in `execute.sh`)
- test models (using the same cli instruction as in training minus the `-t` flag)

Before train/inference process, you also need to compile a config file, which consists of three major keys: `model`, `data` and other lightning-related modules inside `trainer`. You can find samples of them under `configs/` folder.

For reference, the dataset organization in my case is:<br>
`<dataset name>/`<br>
---- `train/`<br>
---- ---- `Case0001.nii.gz`<br>
---- `val/`<br>
---- ---- `Case0002.nii.gz`<br>
---- `test/`<br>
---- `totalseg/`<br>
---- ---- `Case0001.nii.gz`<br>
---- ---- `Case0002.nii.gz`<br>
---- `tumorseg/`<br>
You probably need to write your own `Dataset` class and config files according to your conventions.

# Datasets
- [BTCV Multi-organ Segmentation](https://www.synapse.org/Synapse:syn3193805/challenge/)
- [AMOS Multi-organ Segmentation](https://amos22.grand-challenge.org/)
- [MSD Tumor Segmentation](http://medicaldecathlon.com/)
- [KiTS Tumor Segmentation](https://kits-challenge.org/kits23/)
- [TCIA Archive](https://www.cancerimagingarchive.net/browse-collections/)

Please add appopriate references when you use these datasets

# References
If you use our work, please make sure to cite the following:
```
@article{dai2024guidegen,
  title={GuideGen: A Text-Guided Framework for Full-torso Anatomy and CT Volume Generation},
  author={Dai, Linrui and Zhang, Rongzhao and Yu, Yongrui and Zhang, Xiaofan},
  journal={arXiv preprint arXiv:2403.07247},
  year={2024}
}
```
Our codebase is based on Latent Diffusion and other public libraries, so you can also refer to their repositories for more details
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion)
- [taming-transformers](https://github.com/CompVis/taming-transformers)
- [Conditional Categorical Diffusion Models](https://github.com/LarsDoorenbos/ccdm-stochastic-segmentation)