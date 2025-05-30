import json
from collections import OrderedDict
from typing import Callable, Dict, Optional

import torch
import numpy as np
import PIL
import gc

from tqdm.auto import trange, tqdm
from diffusion_core.guiders.opt_guiders import opt_registry
from diffusion_core.diffusion_utils import latent2image, image2latent
from diffusion_core.custom_forwards.unet_sd import unet_forward
from diffusion_core.guiders.noise_rescales import noise_rescales
from diffusion_core.inversion import Inversion, NullInversion, NegativePromptInversion
from diffusion_core.utils import toggle_grad, use_grad_checkpointing

import logging
log = logging.getLogger(__name__)

def adain(latent_ctrl: torch.Tensor, latent_sty: torch.Tensor):
    ctrl_mean = latent_ctrl.mean(dim=[0,2,3], keepdim=True)
    ctrl_std = latent_ctrl.std(dim=[0,2,3], keepdim=True)
    sty_mean = latent_sty.mean(dim=[0,2,3], keepdim=True)
    sty_std = latent_sty.std(dim=[0,2,3], keepdim=True)

    return ((latent_ctrl - ctrl_mean)/ctrl_std) * sty_std + sty_mean

class GuidanceEditing:
    def __init__(
            self, model, config, root_path: str = None,
            do_others_rescaling: bool = False, others_rescaling_iter_start: int = 30,
            others_rescaling_iter_end: int = 50, others_rescaling_factor: int = 1
    ):

        self.config = config
        self.model = model
        self.root_path = root_path

        self.do_others_rescaling = do_others_rescaling
        self.others_rescaling_iter_start = others_rescaling_iter_start
        self.others_rescaling_iter_end = others_rescaling_iter_end
        self.others_rescaling_factor = others_rescaling_factor

        toggle_grad(self.model.unet, False)

        if config.get('gradient_checkpointing', False):
            use_grad_checkpointing(mode=True)
        else:
            use_grad_checkpointing(mode=False)

        self.guiders = {
            g_data.name: (opt_registry[g_data.name](**g_data.get('kwargs', {})), g_data.g_scale)
            for g_data in config.guiders
        }

        self._setup_inversion_engine()
        self.latents_stack = []

        self.context = None

        self.noise_rescaler = noise_rescales[config.noise_rescaling_setup.type](
            config.noise_rescaling_setup.init_setup,
            **config.noise_rescaling_setup.get('kwargs', {})
        )

        for guider_name, (guider, _) in self.guiders.items():
            guider.clear_outputs()

        self.self_attn_layers_num = config.get('self_attn_layers_num', [6, 1, 9])
        if type(self.self_attn_layers_num[0]) is int:
            for i in range(len(self.self_attn_layers_num)):
                self.self_attn_layers_num[i] = (0, self.self_attn_layers_num[i])
            

    def _setup_inversion_engine(self):
        if self.config.inversion_type == 'ntinv':
            self.inversion_engine = NullInversion(
                self.model,
                self.model.scheduler.num_inference_steps,
                self.config.guiders[0]['g_scale'],
                forward_guidance_scale=1,
                verbose=self.config.verbose
            )
        elif self.config.inversion_type == 'npinv':
            self.inversion_engine = NegativePromptInversion(
                self.model,
                self.model.scheduler.num_inference_steps,
                self.config.guiders[0]['g_scale'],
                forward_guidance_scale=1,
                verbose=self.config.verbose
            )
        elif self.config.inversion_type == 'dummy':
            self.inversion_engine = Inversion(
                self.model,
                self.model.scheduler.num_inference_steps,
                self.config.guiders[0]['g_scale'],
                forward_guidance_scale=1,
                verbose=self.config.verbose
            )
        else:
            raise ValueError('Incorrect InversionType')

    def __call__(
            self,
            image_gt: PIL.Image.Image,
            inv_prompt: str,
            trg_prompt: str,
            control_image: Optional[PIL.Image.Image] = None,
            verbose: bool = False
    ):
        self.train(
            image_gt,
            inv_prompt,
            trg_prompt,
            control_image,
            verbose
        )

        return self.edit()
    
    def train_stylisation(
            self,
            image_gt: PIL.Image.Image,
            inv_prompt: str,
            trg_prompt: str,
            control_image: Optional[PIL.Image.Image] = None,
            inv_control_image_prompt: str = "",
            verbose: bool = False
    ):
        self.init_prompt_stylisation(inv_prompt, trg_prompt, inv_control_image_prompt)
        self.verbose = verbose
        self.sty_image = control_image
        self.cnt_image = image_gt

        image_gt = np.array(image_gt)
        image_ctrl = np.array(control_image)
        if self.config.start_latent == 'inversion':
            _, self.inv_latents, self.uncond_embeddings = self.inversion_engine(
                image_gt, inv_prompt,
                verbose=self.verbose
            )
            # HACK: inverse style image
            _, self.inv_ctrl_latents, self.uncond_ctrl_embeddings = self.inversion_engine(
                image_ctrl, inv_control_image_prompt,
                verbose=self.verbose
            )
        elif self.config.start_latent == 'random':
            self.inv_latents = self.sample_noised_latents(
                image2latent(image_gt, self.model)
            )
        else:
            raise ValueError('Incorrect start latent type')

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                guider.model_patch(self.model, self_attn_layers_num=self.self_attn_layers_num)
        
        # [1, 4, 64, 64]
        self.start_latent = self.inv_latents[-1].clone()

        # print(f'---> START LATENT: {type(self.start_latent)}; {self.start_latent.size()}')

        # XXX AdaIN(start_latent, ctrl_latent)
        # apply_adain - whether to apply adain to initial latents or not
        # adain_start_ix - first iteration after which adain is applied
        # adain_end_ix - last+1 iteration after which adain is applied
        if 'apply_adain' in self.config and self.config.apply_adain == True:
            start_sty_latent = self.inv_ctrl_latents[-1].clone()
            self.start_latent = adain(self.start_latent, start_sty_latent)
        # if 'apply_adain' in self.config and self.config.apply_adain == True:
        #     start_sty_latent = self.inv_ctrl_latents[-1].clone()

        #     cnt_mean = self.start_latent.mean(dim=[0, 2, 3], keepdim=True)
        #     cnt_std = self.start_latent.std(dim=[0, 2, 3], keepdim=True)
        #     sty_mean = start_sty_latent.mean(dim=[0, 2, 3], keepdim=True)
        #     sty_std = start_sty_latent.std(dim=[0, 2, 3], keepdim=True)
        #     self.start_latent = ((self.start_latent-cnt_mean)/cnt_std)*sty_std + sty_mean

        # log.info(f'start latent size: {self.start_latent.size()}')

        params = {
            'model': self.model,
            'inv_prompt': inv_prompt,
            'trg_prompt': trg_prompt
        }
        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'train'):
                guider.train(params)

        for guider_name, (guider, _) in self.guiders.items():
            guider.clear_outputs()

    def _construct_data_dict_stylisation(
            self, latents,
            diffusion_iter,
            timestep
    ):
        uncond_emb, inv_prompt_emb, trg_prompt_emb, inv_ctrl_prompt_emb = self.context.chunk(4)

        if self.uncond_embeddings is not None:
            uncond_emb = self.uncond_embeddings[diffusion_iter]

        # log.info(f'---> {self.model.scheduler.alphas_cumprod.size()}')

        data_dict = {
            'latent': latents,
            'inv_latent': self.inv_latents[-diffusion_iter - 1],
            'inv_ctrl_latent': self.inv_ctrl_latents[-diffusion_iter - 1],
            'timestep': timestep,
            'model': self.model,
            'uncond_emb': uncond_emb,
            'trg_emb': trg_prompt_emb,
            'inv_emb': inv_prompt_emb,
            'inv_ctrl_emb': inv_ctrl_prompt_emb,
            'diff_iter': diffusion_iter,
            'alpha_t': self.model.scheduler.alphas_cumprod[timestep],
            'sty_img': self.sty_image,
            'cnt_img': self.cnt_image,
            'root_path': self.root_path
        }

        with torch.no_grad():
            uncond_unet = unet_forward(
                self.model,
                data_dict['latent'],
                data_dict['timestep'],
                data_dict['uncond_emb'],
                None
            )

        # XXX: Где то гайдеры привязаны к UNet, поэтому отдельно форвардить их не нужно (кажется что так)
        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                guider.clear_outputs()

        # XXX: здесь инферится латент content изображения
        with torch.no_grad():
            inv_prompt_unet = unet_forward(
                self.model,
                data_dict['inv_latent'],
                data_dict['timestep'],
                data_dict['inv_emb'],
                None
            )

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'inv_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_inv_inv": guider.output})
                guider.clear_outputs()

        # HACK: get unet features from style image
        with torch.no_grad():
            inv_style_unet = unet_forward(
                self.model,
                data_dict['inv_ctrl_latent'],
                data_dict['timestep'],
                data_dict['inv_ctrl_emb'],
                None
            )
        # HACK: get inputs for guiders
        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'sty_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_sty_inv": guider.output})
                guider.clear_outputs()

        data_dict['latent'].requires_grad = True

        src_prompt_unet = unet_forward(
            self.model,
            data_dict['latent'],
            data_dict['timestep'],
            data_dict['inv_emb'],
            None
        )

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'cur_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_inv": guider.output})
                guider.clear_outputs()

        # XXX: different in crossattn kwargs (compared to MaterialFusion)
        trg_prompt_unet = unet_forward(
            self.model,
            data_dict['latent'],
            data_dict['timestep'],
            data_dict['trg_emb'],
            None
        )

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'cur_trg' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_trg": guider.output})
                guider.clear_outputs()

        data_dict.update({
            'uncond_unet': uncond_unet,
            'trg_prompt_unet': trg_prompt_unet,
            'sty_unet': inv_style_unet
        })

        # z0_approx_cur = (data_dict['latent'] - torch.sqrt(1 - data_dict['timestep']) * data_dict['trg_prompt_unet']) / torch.sqrt(data_dict['timestep'])
        # img_approx_cur = latent2image(z0_approx_cur, self.model)[0]

        # z0_approx_sty = (data_dict['inv_ctrl_latent'] - torch.sqrt(1 - data_dict['timestep']) * data_dict['sty_unet']) / torch.sqrt(data_dict['timestep'])
        # img_approx_sty = latent2image(z0_approx_sty, self.model)[0]


        return data_dict

    def edit_stylisation(self):
        self.model.scheduler.set_timesteps(self.model.scheduler.num_inference_steps)
        latents = self.start_latent
        self.latents_stack = []

        for i, timestep in tqdm(
                enumerate(self.model.scheduler.timesteps),
                total=self.model.scheduler.num_inference_steps,
                desc='Editing',
                disable=not self.verbose
        ):
            # 1. Construct dict
            data_dict = self._construct_data_dict_stylisation(latents, i, timestep)
            # log.info(data_dict.keys())
            # log.info(data_dict['timestep'])

            # 2. Calculate guidance
            noise_pred = self._get_noise(data_dict, i)

            # 3. Scheduler step
            latents = self._step(noise_pred, timestep, latents)
            # log.info(f'sty lat:{data_dict["inv_ctrl_latent"].size()}')
            if i >= self.config.adain_start_ix and i < self.config.adain_end_ix:
                log.info(f'{i}')
                latents = adain(latents, data_dict['inv_ctrl_latent'])
            # log.info(latents.size())

        self._model_unpatch(self.model)
        return latent2image(latents, self.model)[0]

    def call_stylisation(
            self,
            image_gt: PIL.Image.Image,
            inv_prompt: str,
            trg_prompt: str,
            control_image: PIL.Image.Image,
            inv_control_prompt: str,
            verbose: bool = False
    ):
        self.train_stylisation(
            image_gt,
            inv_prompt,
            trg_prompt,
            control_image,
            inv_control_prompt,
            verbose
        )

        return self.edit_stylisation()

    def train(
            self,
            image_gt: PIL.Image.Image,
            inv_prompt: str,
            trg_prompt: str,
            control_image: Optional[PIL.Image.Image] = None,
            verbose: bool = False
    ):
        self.init_prompt(inv_prompt, trg_prompt)
        self.verbose = verbose

        image_gt = np.array(image_gt)
        if self.config.start_latent == 'inversion':
            _, self.inv_latents, self.uncond_embeddings = self.inversion_engine(
                image_gt, inv_prompt,
                verbose=self.verbose
            )
        elif self.config.start_latent == 'random':
            self.inv_latents = self.sample_noised_latents(
                image2latent(image_gt, self.model)
            )
        else:
            raise ValueError('Incorrect start latent type')

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                guider.model_patch(self.model, self_attn_layers_num=self.self_attn_layers_num)

        self.start_latent = self.inv_latents[-1].clone()

        params = {
            'model': self.model,
            'inv_prompt': inv_prompt,
            'trg_prompt': trg_prompt
        }
        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'train'):
                guider.train(params)

        for guider_name, (guider, _) in self.guiders.items():
            guider.clear_outputs()

    def _construct_data_dict(
            self, latents,
            diffusion_iter,
            timestep
    ):
        uncond_emb, inv_prompt_emb, trg_prompt_emb = self.context.chunk(3)

        if self.uncond_embeddings is not None:
            uncond_emb = self.uncond_embeddings[diffusion_iter]

        data_dict = {
            'latent': latents,
            'inv_latent': self.inv_latents[-diffusion_iter - 1],
            'timestep': timestep,
            'model': self.model,
            'uncond_emb': uncond_emb,
            'trg_emb': trg_prompt_emb,
            'inv_emb': inv_prompt_emb,
            'diff_iter': diffusion_iter
        }

        with torch.no_grad():
            uncond_unet = unet_forward(
                self.model,
                data_dict['latent'],
                data_dict['timestep'],
                data_dict['uncond_emb'],
                None
            )

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                guider.clear_outputs()

        with torch.no_grad():
            inv_prompt_unet = unet_forward(
                self.model,
                data_dict['inv_latent'],
                data_dict['timestep'],
                data_dict['inv_emb'],
                None
            )

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'inv_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_inv_inv": guider.output})
                guider.clear_outputs()

        data_dict['latent'].requires_grad = True

        src_prompt_unet = unet_forward(
            self.model,
            data_dict['latent'],
            data_dict['timestep'],
            data_dict['inv_emb'],
            None
        )

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'cur_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_inv": guider.output})
                guider.clear_outputs()

        trg_prompt_unet = unet_forward(
            self.model,
            data_dict['latent'],
            data_dict['timestep'],
            data_dict['trg_emb'],
            None
        )

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'cur_trg' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_trg": guider.output})
                guider.clear_outputs()

        data_dict.update({
            'uncond_unet': uncond_unet,
            'trg_prompt_unet': trg_prompt_unet,
        })

        return data_dict

    def _get_noise(self, data_dict, diffusion_iter):
        backward_guiders_sum = 0.
        noises = {
            'uncond': data_dict['uncond_unet'],
        }
        index = torch.where(self.model.scheduler.timesteps == data_dict['timestep'])[0].item()

        # self.noise_rescaler
        for name, (guider, g_scale) in self.guiders.items():
            if guider.grad_guider:
                s = self._get_scale(g_scale, diffusion_iter)
                cur_noise_pred = s * guider(data_dict)
                noises[name] = cur_noise_pred
            else:
                s = self._get_scale(g_scale, diffusion_iter)
                energy = s * guider(data_dict)
                log.info(energy)
                if not torch.allclose(energy, torch.tensor(0.)):
                    backward_guiders_sum += energy
        if hasattr(backward_guiders_sum, 'backward'):
            backward_guiders_sum.backward()
            noises['other'] = data_dict['latent'].grad

        scales = self.noise_rescaler(noises, index)
        if self.do_others_rescaling and diffusion_iter >= self.others_rescaling_iter_start and diffusion_iter < self.others_rescaling_iter_end:
            scales['other'] *= self.others_rescaling_factor * torch.norm(noises['uncond']) / torch.norm(noises['other'])
        # if diffusion_iter >= 30:
        #     scales['other'] *= 2 * torch.norm(noises['uncond']) / torch.norm(noises['other'])
        noise_pred = sum(scales[k] * noises[k] for k in noises)

        log.info(f'Uncond norm: {torch.norm(scales["uncond"] * noises["uncond"])}')
        if 'other' in noises:
            log.info(f'Others norm: {torch.norm(scales["other"] * noises["other"])}')

        for g_name, (guider, _) in self.guiders.items():
            if not guider.grad_guider:
                guider.clear_outputs()
            gc.collect()
            torch.cuda.empty_cache()

        return noise_pred

    @staticmethod
    def _get_scale(g_scale, diffusion_iter):
        if type(g_scale) is float:
            return g_scale
        else:
            return g_scale[diffusion_iter]

    @torch.no_grad()
    def _step(self, noise_pred, t, latents):
        latents = self.model.scheduler.step_backward(noise_pred, t, latents).prev_sample
        self.latents_stack.append(latents.detach())
        return latents

    def edit(self):
        self.model.scheduler.set_timesteps(self.model.scheduler.num_inference_steps)
        latents = self.start_latent
        self.latents_stack = []

        for i, timestep in tqdm(
                enumerate(self.model.scheduler.timesteps),
                total=self.model.scheduler.num_inference_steps,
                desc='Editing',
                disable=not self.verbose
        ):
            # 1. Construct dict
            data_dict = self._construct_data_dict(latents, i, timestep)

            # 2. Calculate guidance
            noise_pred = self._get_noise(data_dict, i)

            # 3. Scheduler step
            latents = self._step(noise_pred, timestep, latents)

        self._model_unpatch(self.model)
        return latent2image(latents, self.model)[0]
    
    @torch.no_grad()
    def init_prompt_stylisation(self, inv_prompt: str, trg_prompt: str, inv_ctrl_prompt: str = ""):
        trg_prompt_embed = self.get_prompt_embed(trg_prompt)
        inv_prompt_embed = self.get_prompt_embed(inv_prompt)
        uncond_embed = self.get_prompt_embed("")
        inv_ctrl_prompt_embed = self.get_prompt_embed(inv_ctrl_prompt)

        self.context = torch.cat([uncond_embed, inv_prompt_embed, trg_prompt_embed, inv_ctrl_prompt_embed])

    @torch.no_grad()
    def init_prompt(self, inv_prompt: str, trg_prompt: str):
        trg_prompt_embed = self.get_prompt_embed(trg_prompt)
        inv_prompt_embed = self.get_prompt_embed(inv_prompt)
        uncond_embed = self.get_prompt_embed("")

        self.context = torch.cat([uncond_embed, inv_prompt_embed, trg_prompt_embed])

    def get_prompt_embed(self, prompt: str):
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(
            text_input.input_ids.to(self.model.device)
        )[0]

        return text_embeddings

    def sample_noised_latents(self, latent):
        all_latent = [latent.clone().detach()]
        latent = latent.clone().detach()
        for i in trange(self.model.scheduler.num_inference_steps, desc='Latent Sampling'):
            timestep = self.model.scheduler.timesteps[-i - 1]
            if i + 1 < len(self.model.scheduler.timesteps):
                next_timestep = self.model.scheduler.timesteps[- i - 2]
            else:
                next_timestep = 999

            alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
            alpha_prod_t_next = self.model.scheduler.alphas_cumprod[next_timestep]

            alpha_slice = alpha_prod_t_next / alpha_prod_t

            latent = torch.sqrt(alpha_slice) * latent + torch.sqrt(1 - alpha_slice) * torch.randn_like(latent)
            all_latent.append(latent)
        return all_latent

    def _model_unpatch(self, model):
        def new_forward_info(self):
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

        def register_attn(module):
            if 'Attention' in module.__class__.__name__:
                module.forward = new_forward_info(module)
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    register_attn(module_)

        def remove_hooks(module):
            if hasattr(module, "_forward_hooks"):
                module._forward_hooks: Dict[int, Callable] = OrderedDict()
            if hasattr(module, 'children'):
                for module_ in module.children():
                    remove_hooks(module_)

        register_attn(model.unet)
        remove_hooks(model.unet)
