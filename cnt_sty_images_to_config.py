import os
from pathlib import Path
import argparse
import glob
import yaml
from tqdm import tqdm

def run(cfg):
    cnt_imgs = list(glob.glob(os.path.join(cfg.cnt, '*.png'))) + list(glob.glob(os.path.join(cfg.cnt, '*.jpg'))) + list(glob.glob(os.path.join(cfg.cnt, '*.jpeg')))
    sty_imgs = list(glob.glob(os.path.join(cfg.sty, '*.jpg'))) + list(glob.glob(os.path.join(cfg.sty, '*.jpeg')))

    print(f'Found {len(cnt_imgs)} content images')
    print(f'Found {len(sty_imgs)} style images')

    samples = []
    for i, cnt_p in enumerate(cnt_imgs):
        for sty_p in sty_imgs:
            samples.append({
                'cnt_img_path': cnt_p, 'cnt_prompt': 'An image',
                'sty_img_path': sty_p, 'sty_prompt': '', 'edit_prompt': ''
            })
    
    with open(os.path.join(cfg.save_to, 'customset.yaml'), 'w') as fd:
        yaml.dump(samples, fd, default_flow_style=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', type=str)
    parser.add_argument('--sty', type=str)
    parser.add_argument('--save_to', type=str)
    args = parser.parse_args()

    run(args)
