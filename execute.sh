# train script
# stage 1: training categorical diffusion model
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=20000 main.py\
     -t \
     --debug \                                                                          # --debug flag tells the script to use tensorboard instead of wandb logger
     --gpus 0,1,2,3 \
     --base configs/diffusion/cdpm.yaml \
     --name cdpm \
     --resume_from_checkpoint /home/xxx/data/ldm/cdpm/checkpoints/last.ckpt \           # delete this line if you are not resuming from a checkpoint
     data.params.train.params.max_size=1000                                             # modifying config parameters in cli is also possible via omegaconf.merge(cfg, cli)


# stage 2: training autoencoder
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=20001 main.py\   # note that stage 1 and stage 2 can be trained in parallel
     -t \
     --debug \
     --gpus 0,1,2,3 \
     --base configs/autoencoder/ae.yaml \
     --name ae \

# use the trained autoencoder to save encoded latents for your dataset to save VRAM for stage 3
# if you have enough VRAM (80G VRAM is required for directly training on 256**3 volumes), you do not need to do this)
python inference/save_latents.py \
    --config configs/autoencoder/ae.yaml \
    --checkpoint /home/xxx/data/ldm/ae/checkpoints/last.ckpt \
    --save_root /home/xxx/data/your_dataset/latents

# stage 3: training latent diffusion model
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=20002 main.py\
     -t \
     --debug \
     --gpus 0,1,2,3 \
     --base configs/diffusion/ldm.yaml \
     --name ldm \