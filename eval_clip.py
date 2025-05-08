import os
import argparse
import glob
from pathlib import Path
from tqdm import tqdm
from typing import List

import numpy as np
from PIL import Image

import lpips

import torch
import torch.nn.functional as F
import torchvision
from torchvision.io import ImageReadMode
from torch.utils.data import Dataset, DataLoader

from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

import logging
log = logging.getLogger(__name__)

class ClipImgDataset(Dataset):
    def __init__(self, clip_id: str, output_list: List[str], style_root: str):
        self.style = style_root
        self.clip_img_processor = CLIPImageProcessor.from_pretrained(clip_id)

        # self.output_imgs = glob.glob(os.path.join(output_root, '**', '*.png'), recursive=True)
        self.output_imgs = output_list
        print(f'-----> Found {len(self.output_imgs)} images')
    
    def __len__(self):
        return len(self.output_imgs)
    
    def _get_style_path(self, output_img_path: str):
        fname = Path(output_img_path).stem
        style_type = fname.split('___')[1]
        style_no = fname.split('___')[-1]
        style_type = style_type.replace('%20', ' ')
        style_img_path = os.path.join(self.style, style_type, f'{style_no}.jpg')
        if not os.path.isfile(style_img_path):
            raise FileNotFoundError(style_img_path)
        return style_img_path
    
    def _read_clip_img(self, path: str):
        img = np.array(Image.open(path).convert('RGB'))
        img = self.clip_img_processor(img, return_tensors='pt')["pixel_values"]
        img = img.squeeze(0)
        return img

    def __getitem__(self, index):
        output_img_path = self.output_imgs[index]
        style_img_path = self._get_style_path(output_img_path)

        output_img = self._read_clip_img(output_img_path)
        style_img = self._read_clip_img(style_img_path)

        return output_img, style_img

def main(opt):
    device = torch.device(opt.device)
    print(f'------> Will run on device {device}')

    output_imgs = []
    output_imgs += list(glob.glob('outputs/2025-05-03/20-20-49___attn_v_batch1/output_imgs/**.png'))
    output_imgs += list(glob.glob('outputs/2025-05-03/20-21-28___attn_v_batch2/output_imgs/**.png'))
    output_imgs += list(glob.glob('outputs/2025-05-04/06-57-09___attn_v_batch2/output_imgs/**.png'))
    output_imgs += list(glob.glob('outputs/2025-05-04/06-57-22___attn_v_batch1/output_imgs/**.png'))
    output_imgs += list(glob.glob('outputs/2025-05-04/22-20-20___attn_v_batch1/output_imgs/**.png'))
    output_imgs += list(glob.glob('outputs/2025-05-05/08-58-53___attn_v_batch1/output_imgs/**.png'))
    output_imgs += list(glob.glob('outputs/2025-05-05/23-08-11___attn_v_batch1/output_imgs/**.png'))
    output_imgs += list(glob.glob('outputs/2025-05-06/09-53-30___attn_v_batch1/output_imgs/**.png'))
    true_output_imgs = []
    for oimg in output_imgs:
        if 'panel___' not in oimg:
            true_output_imgs.append(oimg)

    dataset = ClipImgDataset(
        clip_id=opt.clip_id, output_list=true_output_imgs, style_root=opt.style
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=opt.batch_size,
        shuffle=False, num_workers=4, drop_last=False
    )

    img_encoder = CLIPVisionModelWithProjection.from_pretrained(opt.clip_id).to(device)

    clip_score = 0.0
    total_seen = 0

    with torch.no_grad():
        for output, style in tqdm(dataloader):
            output = output.to(device)
            style = style.to(device)

            curr_batch_size = output.size(0)
            total_seen += curr_batch_size

            output_feats = img_encoder(pixel_values=output).image_embeds
            output_feats = output_feats / output_feats.norm(dim=1, keepdim=True)
            style_feats = img_encoder(pixel_values=style).image_embeds
            style_feats = style_feats / style_feats.norm(dim=1, keepdim=True)
            score = F.cosine_similarity(output_feats, style_feats).sum()

            clip_score += score

    print(f'-----> CLIP: {clip_score / total_seen}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--clip_id', type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument('--device', type=str, default='cuda:1')
    opt = parser.parse_args()
    main(opt)