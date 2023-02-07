import torch
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import math
import imageio
from PIL import Image
import torchvision
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import datetime
import torch
import sys
import os
from torchvision import datasets
import pickle

# StableDiffusion P2P implementation originally from https://github.com/bloc97/CrossAttentionControl

# Have diffusers with hardcoded double-casting instead of float
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from diffusers import LMSDiscreteScheduler, PNDMScheduler, DDPMScheduler, DDIMScheduler

import random
from tqdm.auto import tqdm
from torch import autocast
from difflib import SequenceMatcher


# has to be done with attn slicing
# unet.set_attention_slice(1)


def do_inversion(pipe,
                 init_image,
                 base_prompt,
                 steps=50,
                 mix_weight=0.93,
                 init_image_strength=0.8,
                 guidance_scale=3,
                 run_baseline=False,
                 height=512,
                 width=512,
                 ):
    # compute latent pair (second one will be original latent if run_baseline=True)
    latents = coupled_stablediffusion(pipe,
                                      base_prompt,
                                      reverse=True,
                                      init_image=init_image,
                                      init_image_strength=init_image_strength,
                                      steps=steps, mix_weight=mix_weight,
                                      guidance_scale=guidance_scale,
                                      run_baseline=run_baseline,
                                      height=height,
                                      width=width,)

    return latents


def EDICT_editing(pipe,
                  init_image,
                  base_prompt,
                  edit_prompt,
                  use_p2p=False,
                  steps=50,
                  mix_weight=0.93,
                  init_image_strength=0.8,
                  guidance_scale=3,
                  run_baseline=False,
                  leapfrog_steps=True, ):
    """
    Main call of our research, performs editing with either EDICT or DDIM

    Args:
        im_path: path to image to run on
        base_prompt: conditional prompt to deterministically noise with
        edit_prompt: desired text conditoining
        steps: ddim steps
        mix_weight: Weight of mixing layers.
            Higher means more consistent generations but divergence in inversion
            Lower means opposite
            This is fairly tuned and can get good results
        init_image_strength: Editing strength. Higher = more dramatic edit.
            Typically [0.6, 0.9] is good range.
            Definitely tunable per-image/maybe best results are at a different value
        guidance_scale: classifier-free guidance scale
            3 I've found is the best for both our method and basic DDIM inversion
            Higher can result in more distorted results
        run_baseline:
            VERY IMPORTANT
            True is EDICT, False is DDIM
    Output:
        PAIR of Images (tuple)
        If run_baseline=True then [0] will be edit and [1] will be original
        If run_baseline=False then they will be two nearly identical edited versions
    """
    # Resize/center crop to 512x512 (Can do higher res. if desired)
    # orig_im = preprocess(init_image, 512, 512)

    # compute latent pair (second one will be original latent if run_baseline=True)
    latents = coupled_stablediffusion(pipe,
        base_prompt,
                                      reverse=True,
                                      init_image=init_image,
                                      init_image_strength=init_image_strength,
                                      steps=steps,
                                      mix_weight=mix_weight,
                                      guidance_scale=guidance_scale,
                                      run_baseline=run_baseline)
    # Denoise intermediate state with new conditioning
    gen = coupled_stablediffusion(pipe,
        edit_prompt if (not use_p2p) else base_prompt,
                                  None if (not use_p2p) else edit_prompt,
                                  fixed_starting_latent=latents,
                                  init_image_strength=init_image_strength,
                                  steps=steps,
                                  mix_weight=mix_weight,
                                  guidance_scale=guidance_scale,
                                  run_baseline=run_baseline)

    return gen


def center_crop(im):
    width, height = im.size  # Get dimensions
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


####################################

#### HELPER FUNCTIONS FOR OUR METHOD #####

def get_alpha_and_beta(t, scheduler):
    # want to run this for both current and previous timestep
    if t.dtype == torch.long:
        alpha = scheduler.alphas_cumprod[t]
        return alpha, 1 - alpha

    if t < 0:
        return scheduler.final_alpha_cumprod, 1 - scheduler.final_alpha_cumprod

    low = t.floor().long()
    high = t.ceil().long()
    rem = t - low

    low_alpha = scheduler.alphas_cumprod[low]
    high_alpha = scheduler.alphas_cumprod[high]
    interpolated_alpha = low_alpha * rem + high_alpha * (1 - rem)
    interpolated_beta = 1 - interpolated_alpha
    return interpolated_alpha, interpolated_beta


# A DDIM forward step function
def forward_step(
        self,
        model_output,
        timestep: int,
        sample,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        return_dict: bool = True,
        use_double=False,
):
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    prev_timestep = timestep - self.config.num_train_timesteps / self.num_inference_steps

    if timestep > self.timesteps.max():
        raise NotImplementedError("Need to double check what the overflow is")

    alpha_prod_t, beta_prod_t = get_alpha_and_beta(timestep, self)
    alpha_prod_t_prev, _ = get_alpha_and_beta(prev_timestep, self)

    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev) ** 0.5)
    first_term = (1. / alpha_quotient) * sample
    second_term = (1. / alpha_quotient) * (beta_prod_t ** 0.5) * model_output
    third_term = ((1 - alpha_prod_t_prev) ** 0.5) * model_output
    return first_term - second_term + third_term


# A DDIM reverse step function, the inverse of above
def reverse_step(
        self,
        model_output,
        timestep: int,
        sample,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        return_dict: bool = True,
        use_double=False,
):
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    prev_timestep = timestep - self.config.num_train_timesteps / self.num_inference_steps

    if timestep > self.timesteps.max():
        raise NotImplementedError
    else:
        alpha_prod_t = self.alphas_cumprod[timestep]

    alpha_prod_t, beta_prod_t = get_alpha_and_beta(timestep, self)
    alpha_prod_t_prev, _ = get_alpha_and_beta(prev_timestep, self)

    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev) ** 0.5)

    first_term = alpha_quotient * sample
    second_term = ((beta_prod_t) ** 0.5) * model_output
    third_term = alpha_quotient * ((1 - alpha_prod_t_prev) ** 0.5) * model_output
    return first_term + second_term - third_term


@torch.no_grad()
def latent_to_image(pipe, latent):
    image = pipe.vae.decode(latent.to(pipe.vae.dtype) / 0.18215).sample
    image = prep_image_for_return(image)
    return image


def prep_image_for_return(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    image = Image.fromarray(image)
    return image


def image_to_latent(pipe, im, width, height):
    if isinstance(im, torch.Tensor):
        # assume it's the latent
        # used to avoid clipping new generation before inversion
        init_latent = im.to(pipe.device)
    else:
        # Resize and transpose for numpy b h w c -> torch b c h w
        im = im.resize((width, height), resample=Image.LANCZOS)
        im = np.array(im).astype(np.float32) / 255.0 * 2.0 - 1.0
        # check if black and white
        if len(im.shape) < 3:
            im = np.stack([im for _ in range(3)], axis=2)  # putting at end b/c channels

        im = torch.from_numpy(im[np.newaxis, ...].transpose(0, 3, 1, 2)).to(torch.float16)

        # If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
        if im.shape[1] > 3:
            im = im[:, :3] * im[:, 3:] + (1 - im[:, 3:])

        # Move image to GPU
        im = im.to(pipe.device)
        # Encode image
        init_latent = pipe.vae.encode(im).latent_dist.sample() * 0.18215
        return init_latent


#############################

##### MAIN EDICT FUNCTION #######
# Use EDICT_editing to perform calls

@torch.no_grad()
def coupled_stablediffusion(pipe,
                            prompt="",
                            prompt_edit=None,
                            null_prompt='',
                            guidance_scale=7.0,
                            steps=50,
                            width=512,
                            height=512,
                            init_image=None,
                            init_image_strength=1.0,
                            run_baseline=False,
                            use_lms=False,
                            leapfrog_steps=True,
                            reverse=False,
                            return_latents=False,
                            fixed_starting_latent=None,
                            beta_schedule='scaled_linear',
                            mix_weight=0.93):
    assert not use_lms, "Can't invert LMS the same as DDIM"
    if run_baseline: leapfrog_steps = False
    # Change size to multiple of 64 to prevent size mismatches inside model
    width = width - width % 64
    height = height - height % 64

    # Preprocess image if it exists (img2img)
    if init_image is not None:
        assert reverse  # want to be performing deterministic noising
        # can take either pair (output of generative process) or single image
        if isinstance(init_image, list):
            if isinstance(init_image[0], torch.Tensor):
                init_latent = [t.clone() for t in init_image]
            else:
                init_latent = [image_to_latent(pipe, im, width, height) for im in init_image]
        else:
            init_latent = image_to_latent(pipe, init_image, width, height)
        # this is t_start for forward, t_end for reverse
        t_limit = steps - int(steps * init_image_strength)
    else:
        assert not reverse, 'Need image to reverse from'
        init_latent = torch.zeros((1, pipe.unet.in_channels, height // 8, width // 8), device=pipe.device)
        t_limit = 0

    if reverse:
        latent = init_latent
    else:
        # Generate random normal noise
        noise = torch.randn(init_latent.shape, device=pipe.device, dtype=torch.float16)
        if fixed_starting_latent is None:
            latent = noise
        else:
            if isinstance(fixed_starting_latent, list):
                latent = [l.clone() for l in fixed_starting_latent]
            else:
                latent = fixed_starting_latent.clone()
            t_limit = steps - int(steps * init_image_strength)
    if isinstance(latent, list):  # initializing from pair of images
        latent_pair = latent
    else:  # initializing from noise
        latent_pair = [latent.clone(), latent.clone()]

    # if steps == 0:
    #     if init_image is not None:
    #         return image_to_latent(init_image)
    #     else:
    #         image = vae.decode(latent.to(vae.dtype) / 0.18215).sample
    #         return prep_image_for_return(image)

    # Set inference timesteps to scheduler
    schedulers = []
    for i in range(2):
        # num_raw_timesteps = max(1000, steps)
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                  beta_schedule=beta_schedule,
                                  num_train_timesteps=1000,
                                  clip_sample=False,
                                  set_alpha_to_one=False)
        scheduler.set_timesteps(steps)
        schedulers.append(scheduler)

    # CLIP Text Embeddings
    tokens_unconditional = pipe.tokenizer(null_prompt, padding="max_length",
                                          max_length=pipe.tokenizer.model_max_length,
                                          truncation=True, return_tensors="pt",
                                          return_overflowing_tokens=True)
    embedding_unconditional = pipe.text_encoder(tokens_unconditional.input_ids.to(pipe.device)).last_hidden_state

    tokens_conditional = pipe.tokenizer(prompt, padding="max_length",
                                        max_length=pipe.tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt",
                                        return_overflowing_tokens=True)
    embedding_conditional = pipe.text_encoder(tokens_conditional.input_ids.to(pipe.device)).last_hidden_state

    timesteps = schedulers[0].timesteps[t_limit:]
    if reverse: timesteps = timesteps.flip(0)

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
        t_scale = t / schedulers[0].num_train_timesteps

        if (reverse) and (not run_baseline):
            # Reverse mixing layer
            new_latents = [l.clone() for l in latent_pair]
            new_latents[1] = (new_latents[1].clone() - (1 - mix_weight) * new_latents[0].clone()) / mix_weight
            new_latents[0] = (new_latents[0].clone() - (1 - mix_weight) * new_latents[1].clone()) / mix_weight
            latent_pair = new_latents

        # alternate EDICT steps
        for latent_i in range(2):
            if run_baseline and latent_i == 1: continue  # just have one sequence for baseline
            # this modifies latent_pair[i] while using
            # latent_pair[(i+1)%2]
            if reverse and (not run_baseline):
                if leapfrog_steps:
                    # what i would be from going other way
                    orig_i = len(timesteps) - (i + 1)
                    offset = (orig_i + 1) % 2
                    latent_i = (latent_i + offset) % 2
                else:
                    # Do 1 then 0
                    latent_i = (latent_i + 1) % 2
            else:
                if leapfrog_steps:
                    offset = i % 2
                    latent_i = (latent_i + offset) % 2

            latent_j = ((latent_i + 1) % 2) if not run_baseline else latent_i

            latent_model_input = latent_pair[latent_j]
            latent_base = latent_pair[latent_i]

            # Predict the conditional noise residual and save the cross-attention layer activations
            noise_pred_cond = pipe.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample

            if guidance_scale > 1:
                # Predict the unconditional noise residual
                noise_pred_uncond = pipe.unet(latent_model_input, t,
                                              encoder_hidden_states=embedding_unconditional).sample


                # Perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            step_call = reverse_step if reverse else forward_step
            new_latent = step_call(schedulers[latent_i],
                                   noise_pred,
                                   t,
                                   latent_base)  # .prev_sample
            new_latent = new_latent.to(latent_base.dtype)

            latent_pair[latent_i] = new_latent

        if (not reverse) and (not run_baseline):
            # Mixing layer (contraction) during generative process
            new_latents = [l.clone() for l in latent_pair]
            new_latents[0] = (mix_weight * new_latents[0] + (1 - mix_weight) * new_latents[1]).clone()
            new_latents[1] = ((1 - mix_weight) * new_latents[0] + (mix_weight) * new_latents[1]).clone()
            latent_pair = new_latents

    # scale and decode the image latents with vae, can return latents instead of images
    if reverse or return_latents:
        results = [latent_pair]
        return results if len(results) > 1 else results[0]

    # decode latents to images
    images = []
    for latent_i in range(2):
        latent = latent_pair[latent_i] / 0.18215
        image = pipe.vae.decode(latent.to(pipe.vae.dtype)).sample
        images.append(image)

    # Return images
    return_arr = []
    for image in images:
        image = prep_image_for_return(image)
        return_arr.append(image)
    results = [return_arr]
    return results if len(results) > 1 else results[0]
