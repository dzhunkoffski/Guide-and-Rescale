"""
Experiment 1: Use eq. 13 for stylisation, but for Phi* use Phi_sty (instead of Phi_cnt)
[ ]: Explore optimal guider scale values
[ ]: Explore when to apply Style guider (in terms of diffusion iter)
[ ]: Explore which features to use for style guiding
[ ]: explore how prompt affects on stylisation (try to use same edit prompt as source prompt, try to left style prompt as empty string)
"""
import os
from datetime import datetime
from pathlib import Path
import copy

import hydra
from omegaconf import OmegaConf, DictConfig

import torch
from torch import nn
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf

from diffusion_core.guiders import GuidanceEditing
from diffusion_core.utils import load_512, use_deterministic
from diffusion_core import diffusion_models_registry, diffusion_schedulers_registry

def log(msg: str):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f'{dt_string} === {msg}')

def get_scheduler(scheduler_name):
    if scheduler_name not in diffusion_schedulers_registry:
        raise ValueError(f"Incorrect scheduler type: {scheduler_name}, possible are {diffusion_schedulers_registry}")
    scheduler = diffusion_schedulers_registry[scheduler_name]()
    return scheduler

def get_model(scheduler, model_name, device):
    model = diffusion_models_registry[model_name](scheduler)
    model.to(device)
    return model

def generate_single(
        cnt_img_path: str, cnt_prompt: str,
        sty_img_path: str, sty_prompt: str,
        edit_prompt: str, edit_cfg: DictConfig, model: nn.Module):
    cnt_img = Image.fromarray(load_512(cnt_img_path))
    sty_img = Image.fromarray(load_512(sty_img_path))

    guidance = GuidanceEditing(model, edit_cfg)
    res = guidance.call_stylisation(
        image_gt=cnt_img, inv_prompt=cnt_prompt, trg_prompt=edit_prompt,
        control_image=sty_img, inv_control_prompt=sty_prompt, verbose=True
    )
    res = Image.fromarray(res)
    return res

@hydra.main(version_base=None, config_path='configs', config_name='exp1')
def run_experiment(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    if hydra_cfg["mode"].name == "RUN":
        run_path = hydra_cfg["run"]["dir"]
    elif hydra_cfg["mode"].name == "MULTIRUN":
        run_path = os.path.join(hydra_cfg["sweep"]["dir"], hydra_cfg["sweep"]["subdir"])
    else:
        raise NotImplementedError()
    
    log(f'[INFO]: Experiment run directory: {run_path}')

    if torch.cuda.is_available():
        torch.cuda.set_device(cfg["device"])
    use_deterministic()

    device = torch.device(cfg["device"] if torch.cuda.is_available() else 'cpu')

    scheduler = get_scheduler(cfg['scheduler_name'])
    model = get_model(scheduler, cfg['model_name'], device)
    config = cfg['guidance_cfg']

    os.makedirs(os.path.join(run_path, 'output_imgs'))

    for sample_items in cfg['samples']:
        cnt_name = Path(sample_items['cnt_img_path']).stem
        sty_name = Path(sample_items['sty_img_path']).stem
        log(f'Processing cnt={cnt_name}; sty={sty_name}')

        style_g_ix = cfg['exp_configs']['style_guider_ix']
        assert config['guiders'][style_g_ix]['name'] == 'style_features_map_l2'
        g_config = copy.deepcopy(config)
        for g_scale_ix in range(len(g_config['guiders'][style_g_ix]['g_scale'])):
            if g_scale_ix < cfg['exp_configs']['style_guider_scale_n_iters']:
                g_config['guiders'][style_g_ix]['g_scale'][g_scale_ix] = cfg['exp_configs']['style_guider_scale_default']
            g_config['guiders'][style_g_ix]['g_scale'][g_scale_ix] *= cfg['exp_configs']['style_guider_scale_multiplier']

        res = generate_single(
            edit_cfg=g_config, model=model,
            **sample_items
        )
        res.save(
            os.path.join(run_path, 'output_imgs', f'{cnt_name}___{sty_name}.png')
        )

if __name__ == '__main__':
    run_experiment()
