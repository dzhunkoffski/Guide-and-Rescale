import os
import pickle
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Literal
import torchvision
import torchvision.transforms as T
from torchvision.models import vgg19, VGG19_Weights
from torchvision.utils import save_image

from diffusion_core.utils.class_registry import ClassRegistry
from diffusion_core.guiders.scale_schedulers import last_steps, first_steps
from diffusion_core.diffusion_utils import latent2image, image2latent

from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

opt_registry = ClassRegistry()

from utils.visualizers import visualize_self_attn

import logging
log = logging.getLogger(__name__)

class BaseGuider:
    def __init__(self):
        self.clear_outputs()
    
    @property
    def grad_guider(self):
        return hasattr(self, 'grad_fn')
    
    def __call__(self, data_dict):
        if self.grad_guider:
            return self.grad_fn(data_dict)
        else:
            return self.calc_energy(data_dict)
        
    def clear_outputs(self):
        if not self.grad_guider:
            self.output = self.single_output_clear()
    
    def single_output_clear(self):
        raise NotImplementedError()


@opt_registry.add_to_registry('cfg')
class ClassifierFreeGuidance(BaseGuider):
    def __init__(self, is_source_guidance=False):
        self.is_source_guidance = is_source_guidance
    
    def grad_fn(self, data_dict):
        prompt_unet = data_dict['src_prompt_unet'] if self.is_source_guidance else data_dict['trg_prompt_unet']
        return prompt_unet - data_dict['uncond_unet']


@opt_registry.add_to_registry('latents_diff')
class LatentsDiffGuidance(BaseGuider):
    """
    \| z_t* - z_t \|^2_2
    """
    def grad_fn(self, data_dict):
        return 2 * (data_dict['latent'] - data_dict['inv_latent'])

@opt_registry.add_to_registry('latents_sty_diff')
class LatentsDiffStyleGuidance(BaseGuider):
    """
    \| z_t* - z_t \|^2_2
    """
    def grad_fn(self, data_dict):
        return 2 * (data_dict['latent'] - data_dict['inv_ctrl_latent'])

@opt_registry.add_to_registry('style_features_map_l2')
class StyleFeaturesMapL2EnergyGuider(BaseGuider):
    def __init__(self, block='up'):
        assert block in ['down', 'up', 'mid', 'whole']
        self.block = block

    patched = True
    forward_hooks = ['cur_trg', 'sty_inv']
    def calc_energy(self, data_dict):
        return torch.mean(torch.pow(data_dict['style_features_map_l2_cur_trg'] - data_dict['style_features_map_l2_sty_inv'], 2))

    def model_patch(self, model, self_attn_layers_num=None):
        def hook_fn(module, input, output):
            self.output = output
        if self.block == 'mid':
            model.unet.mid_block.register_forward_hook(hook_fn)
        elif self.block == 'up':
            model.unet.up_blocks[1].resnets[1].register_forward_hook(hook_fn)
        elif self.block == 'down':
            model.unet.down_blocks[1].resnets[1].register_forward_hook(hook_fn)

    def single_output_clear(self):
        None
                
@opt_registry.add_to_registry('features_map_l2')
class FeaturesMapL2EnergyGuider(BaseGuider):
    def __init__(self, block='up'):
        assert block in ['down', 'up', 'mid', 'whole']
        self.block = block
        
    patched = True
    forward_hooks = ['cur_trg', 'inv_inv']
    def calc_energy(self, data_dict):
        return torch.mean(torch.pow(data_dict['features_map_l2_cur_trg'] - data_dict['features_map_l2_inv_inv'], 2))

    # XXX: model_patch - ???
    # XXX: register_foward_hook - ???
    def model_patch(self, model, self_attn_layers_num=None):
        def hook_fn(module, input, output):
            self.output = output 
        if self.block == 'mid':
            model.unet.mid_block.register_forward_hook(hook_fn)
        elif self.block == 'up':
            model.unet.up_blocks[1].resnets[1].register_forward_hook(hook_fn)
        elif self.block == 'down':
            model.unet.down_blocks[1].resnets[1].register_forward_hook(hook_fn)

    def single_output_clear(self):
        None

class StyleLossG(nn.Module):
    def __init__(self, target_feature):
        super(StyleLossG, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = target_feature

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class AdainLoss(nn.Module):
    def __init__(self, style_feature, content_feature):
        super().__init__()

        cnt_mean = content_feature.mean(dim=[0,2,3], keepdim=True)
        cnt_std = content_feature.std(dim=[0,2,3], keepdim=True)
        sty_mean = style_feature.mean(dim=[0,2,3], keepdim=True)
        sty_std = style_feature.std(dim=[0,2,3], keepdim=True)
        self.target_feat = (((content_feature - cnt_mean) / cnt_std) * sty_std + sty_mean).detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target_feat)
        return input

@opt_registry.add_to_registry('adain_style_guider_v1')
class AdainStyleGuiderV1(BaseGuider):
    def __init__(self, style_layers):
        self.t = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.encoder = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        self.style_layers = style_layers

    def single_output_clear(self):
        None

    def calc_energy(self, data_dict):
        z0_approx_cur = (data_dict['latent'] - torch.sqrt(1 - data_dict['alpha_t']) * data_dict['trg_prompt_unet']) / torch.sqrt(data_dict['alpha_t'])
        img_approx_cur = z0_approx_cur / data_dict['model'].vae.config.scaling_factor
        img_approx_cur = (data_dict['model'].vae.decode(img_approx_cur)['sample'] + 1) / 2
        img_approx_cur = img_approx_cur.clamp(0, 1)

        img_approx_cur = self.t(img_approx_cur)

        img_approx_sty = T.ToTensor()(data_dict['sty_img']).to(img_approx_cur.device).unsqueeze(0)
        img_approx_sty = self.t(img_approx_sty)

        img_approx_cnt = T.ToTensor()(data_dict['cnt_img']).to(img_approx_cur.device).unsqueeze(0)
        img_approx_cnt = self.t(img_approx_cnt)

        model = nn.Sequential(nn.Identity())
        adain_losses = []
        layer_ix = 0
        for layer in self.encoder.children():
            if isinstance(layer, nn.Conv2d):
                layer_ix += 1
                name = 'conv_{}'.format(layer_ix)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(layer_ix)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(layer_ix)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(layer_ix)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            layer = layer.to(img_approx_cur.device)
            model.add_module(name, layer)

            if name in self.style_layers:
                sty_feature = model(img_approx_sty)
                cnt_feature = model(img_approx_cnt)
                adain_loss = AdainLoss(sty_feature, cnt_feature)
                adain_loss.target_feat = adain_loss.target_feat.to(img_approx_cur.device)
                model.add_module("style_loss_{}".format(layer_ix), adain_loss)
                adain_losses.append(adain_loss)
        model = model.to(img_approx_cur.device)
        model(img_approx_cur)
        adaloss = 0.0
        for sl in adain_losses:
            adaloss += sl.loss
        return adaloss

@opt_registry.add_to_registry('perceptual_style_guider')
class PerceptualStyleGuider(BaseGuider):
    # https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    def __init__(self, style_layers, apply_gramm: bool = True):
        self.t = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.encoder = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        self.style_layers = style_layers
        self.apply_gramm = apply_gramm

    def single_output_clear(self):
        None

    def calc_energy(self, data_dict):
        log.info(data_dict['root_path'])
        z0_approx_cur = (data_dict['latent'] - torch.sqrt(1 - data_dict['alpha_t']) * data_dict['trg_prompt_unet']) / torch.sqrt(data_dict['alpha_t'])
        img_approx_cur = z0_approx_cur / data_dict['model'].vae.config.scaling_factor
        img_approx_cur = (data_dict['model'].vae.decode(img_approx_cur)['sample'] + 1) / 2
        img_approx_cur = img_approx_cur.clamp(0, 1)

        # with torch.no_grad():
        #     img_from_latent = ((data_dict['model'].vae.decode(data_dict['latent'] / data_dict['model'].vae.config.scaling_factor)['sample'] + 1) / 2).clamp(0,1)
        # if data_dict["diff_iter"] % 5 == 0 or data_dict['diff_iter'] == 49:
        #     save_image(img_approx_cur, os.path.join(data_dict['root_path'], f'approx_{data_dict["diff_iter"]}.png'))
        #     save_image(img_from_latent, os.path.join(data_dict['root_path'], f'latent_{data_dict["diff_iter"]}.png'))
        img_approx_cur = self.t(img_approx_cur)


        img_approx_sty = T.ToTensor()(data_dict['sty_img']).to(img_approx_cur.device).unsqueeze(0)
        img_approx_sty = self.t(img_approx_sty)
        model = nn.Sequential(nn.Identity())
        style_losses = []
        layer_ix = 0
        for layer in self.encoder.children():
            if isinstance(layer, nn.Conv2d):
                layer_ix += 1
                name = 'conv_{}'.format(layer_ix)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(layer_ix)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(layer_ix)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(layer_ix)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            layer = layer.to(img_approx_cur.device)
            model.add_module(name, layer)

            if name in self.style_layers:
                # print(model)
                sty_feature = model(img_approx_sty).detach()
                if self.apply_gramm:
                    stloss = StyleLossG(sty_feature)
                else:
                    stloss = StyleLoss(sty_feature)
                stloss.target = stloss.target.to(img_approx_cur.device)
                model.add_module("style_loss_{}".format(layer_ix), stloss)
                style_losses.append(stloss)
        model = model.to(img_approx_cur.device)
        model(img_approx_cur)
        style_loss = 0.0
        for sl in style_losses:
            style_loss += sl.loss
        return style_loss

@opt_registry.add_to_registry('clip_sty_diff_v1')
class ClipStyDiffGuidanceV1(BaseGuider):
    def __init__(self, dist: Literal['l1', 'l2', 'cos'], clip_id: str, device: str):
        self.dist = dist
        self.clip_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)
        self.t = T.Compose([
            T.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def single_output_clear(self):
        None

    def calc_energy(self, data_dict):
        z0_approx_cur = (data_dict['latent'] - torch.sqrt(1 - data_dict['alpha_t']) * data_dict['trg_prompt_unet']) / torch.sqrt(data_dict['alpha_t'])
        img_approx_cur = z0_approx_cur / data_dict['model'].vae.config.scaling_factor
        img_approx_cur = (data_dict['model'].vae.decode(img_approx_cur)['sample'] + 1) / 2
        img_approx_cur = img_approx_cur.clamp(0, 1)
        img_approx_cur = self.t(img_approx_cur)
        img_approx_cur = self.clip_encoder(pixel_values=img_approx_cur).image_embeds

        z0_approx_sty = (data_dict['inv_ctrl_latent'] - torch.sqrt(1 - data_dict['alpha_t']) * data_dict['sty_unet']) / torch.sqrt(data_dict['alpha_t'])
        img_approx_sty = z0_approx_sty / data_dict['model'].vae.config.scaling_factor
        img_approx_sty = (data_dict['model'].vae.decode(img_approx_sty)['sample'] + 1) / 2
        img_approx_sty = img_approx_sty.clamp(0,1)
        img_approx_sty = self.t(img_approx_sty)
        img_approx_sty = self.clip_encoder(pixel_values=img_approx_sty).image_embeds

        if self.dist == 'l1':
            return torch.mean(torch.pow(img_approx_cur - img_approx_sty, 1))
        elif self.dist == 'l2':
            return torch.mean(torch.pow(img_approx_cur - img_approx_sty, 2))
        
        img_approx_cur = img_approx_cur / img_approx_cur.norm(dim=1, keepdim=True)
        img_approx_sty = img_approx_sty / img_approx_sty.norm(dim=1, keepdim=True)
        return -F.cosine_similarity(img_approx_cur, img_approx_sty).sum()

@opt_registry.add_to_registry('clip_sty_diff_v2')
class ClipStyDiffGuidanceV2(ClipStyDiffGuidanceV1):
    def calc_energy(self, data_dict):
        z0_approx_cur = (data_dict['latent'] - torch.sqrt(1 - data_dict['alpha_t']) * data_dict['trg_prompt_unet']) / torch.sqrt(data_dict['alpha_t'])
        img_approx_cur = z0_approx_cur / data_dict['model'].vae.config.scaling_factor
        img_approx_cur = (data_dict['model'].vae.decode(img_approx_cur)['sample'] + 1) / 2
        img_approx_cur = img_approx_cur.clamp(0, 1)
        img_approx_cur = self.t(img_approx_cur)
        # log.info(f'cur {img_approx_cur.size()}')
        img_approx_cur = self.clip_encoder(pixel_values=img_approx_cur).image_embeds

        img_approx_sty = T.ToTensor()(data_dict['sty_img']).to(img_approx_cur.device).unsqueeze(0)
        img_approx_sty = self.t(img_approx_sty)
        # log.info(f'sty {img_approx_sty.size()}')
        img_approx_sty = self.clip_encoder(pixel_values=img_approx_sty).image_embeds
        
        img_approx_cur = img_approx_cur / img_approx_cur.norm(dim=1, keepdim=True)
        img_approx_sty = (img_approx_sty / img_approx_sty.norm(dim=1, keepdim=True)).detach()
        # if self.dist == 'l1':
        #     return torch.mean(torch.pow(img_approx_cur - img_approx_sty, 1))
        # elif self.dist == 'l2':
        #     return torch.mean(torch.pow(img_approx_cur - img_approx_sty, 2))
        if self.dist == 'l1':
            return F.l1_loss(img_approx_cur, img_approx_sty)
        if self.dist == 'l2':
            return F.mse_loss(img_approx_cur, img_approx_sty)
        return -F.cosine_similarity(img_approx_cur, img_approx_sty).sum()

@opt_registry.add_to_registry('self_attn_map_qkv_l2')
class SelfAttnMapQKVL2EnergyGuider(BaseGuider):
    def __init__(
            self, attn_map_scale: float, q_scale: float, kv_scale: float,
            attn_map_iter_start: int, attn_map_iter_end: int,
            qkv_iter_start: int, qkv_iter_end: int, layers_num: int):
        super().__init__()

        self.attn_map_scale = attn_map_scale
        self.q_scale = q_scale
        self.kv_scale = kv_scale
        self.layers_num = layers_num

        self.attn_map_iter_start = attn_map_iter_start
        self.attn_map_iter_end = attn_map_iter_end
        self.qkv_iter_start = qkv_iter_start
        self.qkv_iter_end = qkv_iter_end

    patched = True
    forward_hooks = ['cur_inv', 'inv_inv', 'sty_inv']
    def single_output_clear(self):
        return {
            "down_self": [], 'mid_self': [], 'up_self': [],
            "down_q": [], "mid_q": [], "up_q": [],
            "down_k": [], "mid_k": [], "up_k": [],
            "down_v": [], "mid_v": [], "up_v": []
        }

    def calc_energy(self, data_dict):
        result = torch.tensor(0., device=data_dict["latent"].device)
        for unet_place, data in data_dict["self_attn_map_qkv_l2_cur_inv"].items():
            if "_self" in unet_place and data_dict["diff_iter"] >= self.attn_map_iter_start and data_dict["diff_iter"] < self.attn_map_iter_end:
                for elem_idx, elem in enumerate(data):
                    result += self.attn_map_scale * torch.mean(
                        torch.pow(elem - data_dict["self_attn_map_qkv_l2_inv_inv"][unet_place][elem_idx], 2)
                    )
            elif "_q" in unet_place and data_dict["diff_iter"] >= self.qkv_iter_start and data_dict["diff_iter"] < self.qkv_iter_end:
                for elem_idx, elem in enumerate(data):
                    result += self.q_scale * torch.mean(
                        torch.pow(elem - data_dict["self_attn_map_qkv_l2_inv_inv"][unet_place][elem_idx], 2)
                    )
            elif '_k' or '_v' in unet_place and data_dict["diff_iter"] >= self.qkv_iter_start and data_dict["diff_iter"] < self.qkv_iter_end:
                for elem_idx, elem in enumerate(data):
                    result += self.kv_scale * torch.mean(
                        torch.pow(elem - data_dict["self_attn_map_qkv_l2_sty_inv"][unet_place][elem_idx], 2)
                    )
        self.single_output_clear()
        return result

    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self, place_unet):
            def patched_forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                
                ## Injection
                is_self = encoder_hidden_states is None
                
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)

                if is_self:
                    layer_ix = len(guider_self.output[f"{place_unet}_self"])
                    t1, t2 = guider_self.layers_num[f'{place_unet}_self'][0], guider_self.layers_num[f'{place_unet}_self'][1]
                    if layer_ix >= t1 and layer_ix < t2:
                        guider_self.output[f"{place_unet}_self"].append(attention_probs)
                    else:
                        guider_self.output[f"{place_unet}_self"].append(torch.tensor(0.0))

                    t1, t2 = guider_self.layers_num[f'{place_unet}_q'][0], guider_self.layers_num[f'{place_unet}_q'][1]
                    if layer_ix >= t1 and layer_ix < t2:
                        guider_self.output[f"{place_unet}_q"].append(query)
                    else:
                        guider_self.output[f'{place_unet}_q'].append(torch.tensor(0.0))

                    t1, t2 = guider_self.layers_num[f'{place_unet}_k'][0], guider_self.layers_num[f'{place_unet}_k'][1]
                    if layer_ix >= t1 and layer_ix < t2:
                        guider_self.output[f"{place_unet}_k"].append(key)
                    else:
                        guider_self.output[f'{place_unet}_k'].append(torch.tensor(0.0))
                    
                    t1, t2 = guider_self.layers_num[f'{place_unet}_v'][0], guider_self.layers_num[f'{place_unet}_v'][1]
                    if layer_ix >= t1 and layer_ix < t2:
                        guider_self.output[f"{place_unet}_v"].append(value)
                    else:
                        guider_self.output[f"{place_unet}_v"].append(torch.tensor(0.0))
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states
            return patched_forward
        
        def register_attn(module, place_in_unet, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    module.forward = new_forward_info(module, place_in_unet)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, place_in_unet, layers_num, cur_layers_num)
                return cur_layers_num

        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down", self_attn_layers_num[0])
            if "mid" in name:
                register_attn(net, "mid", self_attn_layers_num[1])
            if "up" in name:
                register_attn(net, "up", self_attn_layers_num[2])

@opt_registry.add_to_registry('self_attn_map_v_l2')
class SelfAttnMapVL2EnergyGuider(BaseGuider):
    def __init__(
            self, 
            attn_map_scale: float, v_scale: float,
            attn_map_iter_start: int, attn_map_iter_end: int,
            v_iter_start: int, v_iter_end: int,
            save_data_dict: bool, save_data_dir: str,
            layers_num: dict):
        super().__init__()
        self.attn_map_scale = attn_map_scale
        self.v_scale = v_scale

        self.attn_map_iter_start = attn_map_iter_start
        self.attn_map_iter_end = attn_map_iter_end
        self.v_iter_start = v_iter_start
        self.v_iter_end = v_iter_end

        self.save_data_dict = save_data_dict
        self.save_data_dir = save_data_dir

        self.layers_num = layers_num

    patched = True
    forward_hooks = ['cur_inv', 'inv_inv', 'sty_inv']
    def single_output_clear(self):
        return {
            "down_self": [], 'mid_self': [], 'up_self': [],
            "down_v": [], "mid_v": [], "up_v": []
        }

    def calc_energy(self, data_dict):
        result = torch.tensor(0., device=data_dict['latent'].device)
        # XXX: save data_dict to observe it later
        if self.save_data_dict:
            source = ['cur_inv', 'inv_inv', 'sty_inv']

            unet_parts = [
                'down_self', 'mid_self', 'up_self',
                'down_v', 'mid_v', 'up_v'
            ]
            for d in source:
                p = os.path.join(self.save_data_dir, d, str(data_dict['diff_iter']))
                os.makedirs(p)
                for part in unet_parts:
                    for i in range(len(data_dict[f'self_attn_map_v_l2_{d}'][part])):
                        img = visualize_self_attn(data_dict[f'self_attn_map_v_l2_{d}'][part][i])
                        img.save(os.path.join(p, f'{part}_{i}.png'))

        for unet_place, data in data_dict['self_attn_map_v_l2_cur_inv'].items():
            if '_self' in unet_place and data_dict['diff_iter'] >= self.attn_map_iter_start and data_dict['diff_iter'] < self.attn_map_iter_end:
                for elem_idx, elem in enumerate(data):
                    result += self.attn_map_scale * torch.mean(
                        torch.pow(
                            elem - data_dict['self_attn_map_v_l2_inv_inv'][unet_place][elem_idx], 2
                        )
                    )
            elif '_v' in unet_place and data_dict['diff_iter'] >= self.v_iter_start and data_dict['diff_iter'] < self.v_iter_end:
                # XXX: multiply by 0.5 (K, V) -> multiply by kv_scale
                for elem_idx, elem in enumerate(data):
                    result += self.v_scale * torch.mean(
                        torch.pow(
                            elem - data_dict['self_attn_map_v_l2_sty_inv'][unet_place][elem_idx], 2
                        )
                    )
        self.single_output_clear()
        return result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        # XXX: `self_attn_layers_num` by default is [(0, 6), (0, 1), (0, 9)] i.e. all
        def new_forward_info(self, place_unet):
            def patched_forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                
                ## Injection
                is_self = encoder_hidden_states is None
                
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)
                # XXX: explore attention_probs output
                # print(place_unet, attention_probs.size())
                if is_self:
                    # XXX: вместо A делать только Q, K, или V
                    # XXX: получаю текущий слой
                    layer_ix = len(guider_self.output[f"{place_unet}_self"])
                    t1, t2 = guider_self.layers_num[f'{place_unet}_self'][0], guider_self.layers_num[f'{place_unet}_self'][1]
                    if layer_ix >= t1 and layer_ix < t2:
                        guider_self.output[f"{place_unet}_self"].append(attention_probs)
                    else:
                        # guider_self.output[f"{place_unet}_self"].append(torch.zeros_like(attention_probs))
                        guider_self.output[f"{place_unet}_self"].append(torch.tensor(0.0))

                    t1, t2 = guider_self.layers_num[f'{place_unet}_v'][0], guider_self.layers_num[f'{place_unet}_v'][1]
                    if layer_ix >= t1 and layer_ix < t2:
                        guider_self.output[f"{place_unet}_v"].append(value)
                    else:
                        # guider_self.output[f"{place_unet}_v"].append(torch.zeros_like(value))
                        guider_self.output[f"{place_unet}_v"].append(torch.tensor(0.0))
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states
            return patched_forward
        
        def register_attn(module, place_in_unet, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    module.forward = new_forward_info(module, place_in_unet)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, place_in_unet, layers_num, cur_layers_num)
                return cur_layers_num
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down", self_attn_layers_num[0])
            if "mid" in name:
                register_attn(net, "mid", self_attn_layers_num[1])
            if "up" in name:
                register_attn(net, "up", self_attn_layers_num[2])

@opt_registry.add_to_registry('self_attn_qkv_l2')
class SelfAttnQKVL2EnergyGuider(BaseGuider):
    def __init__(
            self, 
            q_scale: float, kv_scale: float,
            q_iter_start: int, q_iter_end: int,
            kv_iter_start: int, kv_iter_end: int,
            save_data_dict: bool, save_data_dir: str,
            layers_num: dict):
        super().__init__()
        self.q_scale = q_scale
        self.kv_scale = kv_scale

        self.q_iter_start = q_iter_start
        self.q_iter_end = q_iter_end
        self.kv_iter_start = kv_iter_start
        self.kv_iter_end = kv_iter_end

        self.save_data_dict = save_data_dict
        self.save_data_dir = save_data_dir

        self.layers_num = layers_num

    patched = True
    forward_hooks = ['cur_inv', 'inv_inv', 'sty_inv']
    def single_output_clear(self):
        return {
            "down_self": [], 'mid_self': [], 'up_self': [],
            "down_q": [], "mid_q": [], "up_q": [],
            "down_k": [], "mid_k": [], "up_k": [],
            "down_v": [], "mid_v": [], "up_v": []
        }

    def calc_energy(self, data_dict):
        result = 0.
        # XXX: save data_dict ot observe it later

        # print(data_dict['self_attn_qkv_l2_inv_inv']['down_q'][4].size())
        # assert False, 'stop'

        if self.save_data_dict:
            source = ['cur_inv', 'inv_inv', 'sty_inv']

            unet_parts = [
                'down_self', 'mid_self', 'up_self',
                'down_q', 'mid_q', 'up_q',
                'down_k', 'mid_k', 'up_k',
                'down_v', 'mid_v', 'up_v'
            ]
            for d in source:
                p = os.path.join(self.save_data_dir, d, str(data_dict['diff_iter']))
                os.makedirs(p)
                for part in unet_parts:
                    for i in range(len(data_dict[f'self_attn_qkv_l2_{d}'][part])):
                        img = visualize_self_attn(data_dict[f'self_attn_qkv_l2_{d}'][part][i])
                        img.save(os.path.join(p, f'{part}_{i}.png'))

        for unet_place, data in data_dict['self_attn_qkv_l2_cur_inv'].items():
            if '_q' in unet_place and self.q_iter_start <= data_dict["diff_iter"] and data_dict["diff_iter"] < self.q_iter_end:
                for elem_idx, elem in enumerate(data):
                    result += self.q_scale * torch.mean(
                        torch.pow(
                            elem - data_dict['self_attn_qkv_l2_inv_inv'][unet_place][elem_idx], 2
                        )
                    )
            elif ('_k' or '_v' in unet_place) and self.kv_iter_start <= data_dict["diff_iter"] and data_dict["diff_iter"] < self.kv_iter_end:
                # XXX: multiply by 0.5 (K, V) -> multiply by kv_scale
                for elem_idx, elem in enumerate(data):
                    st_res = torch.mean(
                        torch.pow(
                            elem - data_dict['self_attn_qkv_l2_sty_inv'][unet_place][elem_idx], 2
                        )
                    )
                    result += st_res * self.kv_scale
        self.single_output_clear()
        return result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        # XXX: `self_attn_layers_num` by default is [(0, 6), (0, 1), (0, 9)] i.e. all
        def new_forward_info(self, place_unet):
            def patched_forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                
                ## Injection
                is_self = encoder_hidden_states is None
                
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)
                # XXX: explore attention_probs output
                # print(place_unet, attention_probs.size())
                if is_self:
                    # XXX: вместо A делать только Q, K, или V
                    # XXX: получаю текущий слой
                    layer_ix = len(guider_self.output[f"{place_unet}_self"])
                    t1, t2 = guider_self.layers_num[f'{place_unet}_self'][0], guider_self.layers_num[f'{place_unet}_self'][1]
                    # if layer_ix >= t1 and layer_ix < t2:
                    #     guider_self.output[f"{place_unet}_self"].append(attention_probs)
                    # else:
                    #     guider_self.output[f"{place_unet}_self"].append(torch.zeros_like(attention_probs))

                    t1, t2 = guider_self.layers_num[f'{place_unet}_q'][0], guider_self.layers_num[f'{place_unet}_q'][1]
                    if layer_ix >= t1 and layer_ix < t2:
                        # print('q')
                        guider_self.output[f"{place_unet}_q"].append(query)
                    else:
                        guider_self.output[f'{place_unet}_q'].append(torch.zeros_like(query))

                    t1, t2 = guider_self.layers_num[f'{place_unet}_k'][0], guider_self.layers_num[f'{place_unet}_k'][1]
                    # print(t1, t2)
                    if layer_ix >= t1 and layer_ix < t2:
                        # print('k')
                        guider_self.output[f"{place_unet}_k"].append(key)
                    else:
                        guider_self.output[f'{place_unet}_k'].append(torch.zeros_like(key))

                    t1, t2 = guider_self.layers_num[f'{place_unet}_v'][0], guider_self.layers_num[f'{place_unet}_v'][1]
                    # print(t1, t2)
                    if layer_ix >= t1 and layer_ix < t2:
                        # print('v')
                        guider_self.output[f"{place_unet}_v"].append(value)
                    else:
                        guider_self.output[f"{place_unet}_v"].append(torch.zeros_like(value))
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states
            return patched_forward
        
        def register_attn(module, place_in_unet, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    module.forward = new_forward_info(module, place_in_unet)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, place_in_unet, layers_num, cur_layers_num)
                return cur_layers_num
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down", self_attn_layers_num[0])
            if "mid" in name:
                register_attn(net, "mid", self_attn_layers_num[1])
            if "up" in name:
                register_attn(net, "up", self_attn_layers_num[2])

@opt_registry.add_to_registry('self_attn_map_l2')
class SelfAttnMapL2EnergyGuider(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']
    def single_output_clear(self):
        return {
            "down_cross": [], "mid_cross": [], "up_cross": [],
            "down_self":  [], "mid_self":  [], "up_self":  []
        }
    
    def calc_energy(self, data_dict):
        result = 0.
        for unet_place, data in data_dict['self_attn_map_l2_cur_inv'].items():
            for elem_idx, elem in enumerate(data):
                result += torch.mean(
                    torch.pow(
                        elem - data_dict['self_attn_map_l2_inv_inv'][unet_place][elem_idx], 2
                    )
                )
        self.single_output_clear()
        return result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self, place_unet):
            def patched_forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                
                ## Injection
                is_self = encoder_hidden_states is None
                
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)
                if is_self:
                    # XXX: вместо A делать только Q, K, или V
                    guider_self.output[f"{place_unet}_self"].append(attention_probs)
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states
            return patched_forward
        
        def register_attn(module, place_in_unet, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    module.forward = new_forward_info(module, place_in_unet)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, place_in_unet, layers_num, cur_layers_num)
                return cur_layers_num
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down", self_attn_layers_num[0])
            if "mid" in name:
                register_attn(net, "mid", self_attn_layers_num[1])
            if "up" in name:
                register_attn(net, "up", self_attn_layers_num[2])

    
@opt_registry.add_to_registry('self_attn_map_l2_appearance')
class SelfAttnMapL2withAppearanceEnergyGuider(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']

    def __init__(
        self, self_attn_gs: float, app_gs: float, new_features: bool=False, 
        total_last_steps: Optional[int] = None, total_first_steps: Optional[int] = None
    ):
        super().__init__()
        
        self.new_features = new_features

        if total_last_steps is not None:
            self.app_gs = last_steps(app_gs, total_last_steps)
            self.self_attn_gs = last_steps(self_attn_gs, total_last_steps)
        elif total_first_steps is not None:
            self.app_gs = first_steps(app_gs, total_first_steps)
            self.self_attn_gs = first_steps(self_attn_gs, total_first_steps)
        else:
            self.app_gs = app_gs
            self.self_attn_gs = self_attn_gs

    def single_output_clear(self):
        return {
            "down_self":  [], 
            "mid_self":  [], 
            "up_self":  [],
            "features": None
        }
    
    def calc_energy(self, data_dict):
        self_attn_result = 0.
        unet_places = ['down_self', 'up_self', 'mid_self']
        for unet_place in unet_places:
            data = data_dict['self_attn_map_l2_appearance_cur_inv'][unet_place]
            for elem_idx, elem in enumerate(data):
                self_attn_result += torch.mean(
                    torch.pow(
                        elem - data_dict['self_attn_map_l2_appearance_inv_inv'][unet_place][elem_idx], 2
                    )
                )
        
        features_orig = data_dict['self_attn_map_l2_appearance_inv_inv']['features']
        features_cur = data_dict['self_attn_map_l2_appearance_cur_inv']['features']
        app_result = torch.mean(torch.abs(features_cur - features_orig))

        self.single_output_clear()

        if type(self.app_gs) == float:
            _app_gs = self.app_gs
        else:
            _app_gs = self.app_gs[data_dict['diff_iter']]

        if type(self.self_attn_gs) == float:
            _self_attn_gs = self.self_attn_gs
        else:
            _self_attn_gs = self.self_attn_gs[data_dict['diff_iter']]

        return _self_attn_gs * self_attn_result + _app_gs * app_result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self, place_unet):
            def patched_forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                
                ## Injection
                is_self = encoder_hidden_states is None
                
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)
                if is_self:
                    guider_self.output[f"{place_unet}_self"].append(attention_probs)
                
                hidden_states = torch.bmm(attention_probs, value)

                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states
            return patched_forward
        
        def register_attn(module, place_in_unet, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    module.forward = new_forward_info(module, place_in_unet)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, place_in_unet, layers_num, cur_layers_num)
                return cur_layers_num
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down", self_attn_layers_num[0])
            if "mid" in name:
                register_attn(net, "mid", self_attn_layers_num[1])
            if "up" in name:
                register_attn(net, "up", self_attn_layers_num[2])
        
        def hook_fn(module, input, output):
            guider_self.output["features"] = output

        if guider_self.new_features:
            model.unet.up_blocks[-1].register_forward_hook(hook_fn)
        else:
            model.unet.conv_norm_out.register_forward_hook(hook_fn)
