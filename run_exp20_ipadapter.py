import sys
import os
import copy
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join('..')))
import argparse
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusion_core.guiders.guidance_editing_ip import GuidanceEditingIP
from diffusion_core.utils import load_512, use_deterministic
from diffusion_core.schedulers.sample_schedulers import DDIMScheduler
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import load_image

import logging
log = logging.getLogger(__name__)

import hydra
from omegaconf import OmegaConf, DictConfig

def get_scheduler():
    return DDIMScheduler(
        num_inference_steps=50, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", set_alpha_to_one=False
    )

def generate_single(
        cnt_img_path: str, cnt_prompt: str, sty_img_path: str, edit_prompt: str,
        edit_cfg: dict, exp_cfg: dict, model: nn.Module, device, *args, **kwargs):
    cnt_img = Image.fromarray(load_512(cnt_img_path))
    sty_img = Image.fromarray(load_512(sty_img_path))

    guidance = GuidanceEditingIP(
        model, edit_cfg, exp_cfg['image_encoder_path'], exp_cfg['ip_ckpt'], device
    )
    res = guidance(
        image_gt=cnt_img, inv_prompt=cnt_prompt, trg_prompt=edit_prompt,
        ip_image=sty_img, scale=exp_cfg['exp_configs']['ip_scale'], verbose=True,
    )
    res = Image.fromarray(res)
    return res

@hydra.main(version_base=None, config_path='configs', config_name='exp20')
def run_experiment(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    if hydra_cfg["mode"].name == "RUN":
        run_path = hydra_cfg["run"]["dir"]
    elif hydra_cfg["mode"].name == "MULTIRUN":
        run_path = os.path.join(hydra_cfg["sweep"]["dir"], hydra_cfg["sweep"]["subdir"])
    else:
        raise NotImplementedError()
    
    log.info(f'Experiment run directory: {run_path}')

    if torch.cuda.is_available():
        torch.cuda.set_device(cfg['device'])
    use_deterministic()
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

    scheduler = get_scheduler()
    
    # load model
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        use_safetensors=True,
        scheduler=scheduler,
        vae = vae,
        add_watermarker=False,
    ).to(device)
    ldm_stable.disable_xformers_memory_efficient_attention()
    ldm_stable.disable_xformers_memory_efficient_attention()
    config = cfg['guidance_cfg']
    os.makedirs(os.path.join(run_path, 'output_imgs'))

    for sample_items in cfg['samples']:
        cnt_name = Path(sample_items['cnt_img_path']).stem
        sty_name = Path(sample_items['sty_img_path']).stem
        log.info(f'Processing cnt={cnt_name}; sty={sty_name}')

        g_config = copy.deepcopy(config)
        log.info(f'Attention guider: {g_config["guiders"][1]["name"]}')
        # for guiding_ix in range(cfg["exp_configs"]["attn_guider_start"], cfg["exp_configs"]["attn_guider_end"]):
        #     g_config["guiders"][1]["g_scale"][guiding_ix] = cfg["exp_configs"]["attn_guider_scale"]
        # log.info(f'Scales for attn guider:\n{g_config["guiders"][1]["g_scale"]}')
        res = generate_single(
            exp_cfg=cfg, edit_cfg=g_config, device=device, model=ldm_stable, **sample_items
        )
        res.save(os.path.join(run_path, 'output_imgs', f'{cnt_name}___{sty_name}.png'))

if __name__ == '__main__':
    run_experiment()
