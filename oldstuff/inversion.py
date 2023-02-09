import torch
from PIL import Image
import numpy as np
from typing import Union


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], model):
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
    return prev_sample


def next_step(model, model_output, timestep, sample):
    timestep, next_timestep = min(
        timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_prod_t_next = model.scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(model, latents, t, context):
    noise_pred = model.unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


# def get_noise_pred(model, latents, t, context, is_forward=True):
#     latents_input = torch.cat([latents] * 2)
#     guidance_scale = 1 if is_forward else GUIDANCE_SCALE
#     noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
#     noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
#     noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
#     return noise_pred


@torch.no_grad()
def latent2image(model, latents, return_type='np'):
    latents = 1 / 0.18215 * latents.detach()
    image = model.vae.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
    return image


@torch.no_grad()
def image2latent(model, image):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).to(model.vae.dtype) / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(model.device)
            latents = model.vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
    return latents


@torch.no_grad()
def init_prompt(model, prompt: str, cfg):
    text_input = model.tokenizer(
        [prompt],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    if cfg:
        uncond_input = model.tokenizer(
            [""], padding="max_length", max_length=model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
        context = torch.cat([uncond_embeddings, text_embeddings])
    else:
        context = text_embeddings

    return context


@torch.no_grad()
def ddim_loop(latent, context, model, steps):
    # uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    model.scheduler.set_timesteps(steps)
    for i in range(steps):
        t = model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(model, latent, t, context)
        latent = next_step(model, noise_pred, t, latent)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipe, image, prompt, steps, cfg=False):
    latent = image2latent(pipe, image)
    context = init_prompt(pipe, prompt, cfg)
    ddim_latents = ddim_loop(latent, context, pipe, steps)
    return ddim_latents