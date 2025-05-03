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

import logging
log = logging.getLogger(__name__)

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
        sty_img_path: str, sty_prompt: str, root_path: str,
        edit_prompt: str, edit_cfg: DictConfig, model: nn.Module,
        do_others_rescaling, others_rescaling_iter_start, others_rescaling_iter_end, others_rescaling_factor,
        *args, **kwargs):
    cnt_img = Image.fromarray(load_512(cnt_img_path))
    sty_img = Image.fromarray(load_512(sty_img_path))

    guidance = GuidanceEditing(
        model, edit_cfg, root_path=root_path,
        do_others_rescaling=do_others_rescaling, others_rescaling_iter_start=others_rescaling_iter_start,
        others_rescaling_iter_end=others_rescaling_iter_end, others_rescaling_factor=others_rescaling_factor
    )
    res = guidance.call_stylisation(
        image_gt=cnt_img, inv_prompt=cnt_prompt, trg_prompt=edit_prompt,
        control_image=sty_img, inv_control_prompt=sty_prompt, verbose=True
    )
    res = Image.fromarray(res)
    return res

def img_panel(img1: Image, img2: Image, img3: Image):
    img1 = img1.resize((256, 256))
    img2 = img2.resize((256, 256))
    img3 = img3.resize((256, 256))

    total_width = 256 * 3
    new_im = Image.new('RGB', (total_width, 256))

    x_offset = 0
    for im in [img1, img2, img3]:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.width
    
    return new_im

@hydra.main(version_base=None, config_path='configs', config_name='exp16')
def run_experiment(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    if hydra_cfg['mode'].name == 'RUN':
        run_path = hydra_cfg['run']['dir']
    elif hydra_cfg['mode'].name == 'MULTIRUN':
        run_path = os.path.join(hydra_cfg['sweep']['dir'], hydra_cfg['sweep']['subdir'])
    else:
        raise NotImplementedError()

    log.info(f'[INFO]: EXperiment run directory: {run_path}')

    if torch.cuda.is_available():
        torch.cuda.set_device(cfg['device'])
    use_deterministic()

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

    scheduler = get_scheduler(cfg['scheduler_name'])
    model = get_model(scheduler, cfg['model_name'], device)
    config = cfg['guidance_cfg']

    os.makedirs(os.path.join(run_path, 'output_imgs'))

    for sample_items in cfg['samples']:
        torch.cuda.empty_cache()
        cnt_name = Path(sample_items['cnt_img_path']).stem
        sty_name = Path(sample_items['sty_img_path']).stem
        sty_name = os.path.basename(os.path.dirname(sample_items['sty_img_path'])) + '___'  + sty_name
        log.info(f'Processing cnt={cnt_name}; sty={sty_name}')

        cnt_img = Image.open(sample_items['cnt_img_path'])
        sty_img = Image.open(sample_items["sty_img_path"])

        g_config = copy.deepcopy(config)

        # Prepare content guider
        log.info(f'Content guider: {g_config["guiders"][1]["name"]}')
        for guiding_ix in range(cfg["exp_configs"]["content_guider_start"], cfg["exp_configs"]["content_guider_end"]):
            g_config["guiders"][1]["g_scale"][guiding_ix] = cfg['exp_configs']['content_guider_scale']
        log.info(f'Scales for content guider:\n{g_config["guiders"][1]["g_scale"]}')

        # Prepare style guider
        log.info(f'Style guider: {g_config["guiders"][2]["name"]}')
        for guiding_ix in range(cfg['exp_configs']['style_guider_start'], cfg['exp_configs']['style_guider_end']):
            g_config['guiders'][2]['g_scale'][guiding_ix] = cfg['exp_configs']['style_guider_scale']
        log.info(f'Scales for style guider:\n{g_config["guiders"][2]["g_scale"]}')

        os.makedirs(os.path.join(run_path, f'{cnt_name}___{sty_name}'), exist_ok=True)
        res = generate_single(
            edit_cfg=g_config, model=model,
            root_path=os.path.join(run_path, f'{cnt_name}___{sty_name}'),
            **cfg['exp_configs'], **sample_items
        )
        res.save(os.path.join(run_path, 'output_imgs', f'{cnt_name}___{sty_name}.png'))
        panel = img_panel(cnt_img, sty_img, res)
        panel.save(os.path.join(run_path, 'output_imgs', f'panel___{cnt_name}___{sty_name}.png'))

if __name__ == '__main__':
    run_experiment()
