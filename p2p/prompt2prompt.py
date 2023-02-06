from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as F
import numpy as np
import abc
import p2p.ptp_utils as ptp_utils
import p2p.seq_aligner as seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image

# class EthanBlend:
#
#     def get_mask(self, x_t, maps, alpha, use_pool):
#         k = 1
#         maps = (maps * alpha).sum(-1).mean(1)
#         if use_pool:
#             maps = F.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
#         mask = F.interpolate(maps, size=(x_t.shape[2:]))
#         # TODO maybe use gaussian smoothing here?
#         if True:
#             pass
#         mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
#         mask = mask.gt(self.th[1 - int(use_pool)])
#         mask = mask[:1] + mask
#         return mask
#
#     def __call__(self, x_t, attention_store, MAX_NUM_WORDS=77):
#         self.counter += 1
#         if self.counter > self.start_blend:
#
#             maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
#             maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
#             maps = torch.cat(maps, dim=1)
#             mask = self.get_mask(x_t, maps, self.alpha_layers, True)
#             if self.subtract_layers is not None:
#                 maps_sub = ~self.get_mask(x_t, maps, self.subtract_layers, False)
#                 mask = mask * maps_sub
#             mask = mask.to(x_t.dtype)
#             x_t = x_t[:1] + mask * (x_t - x_t[:1])
#         return x_t
#
#     # th is threshold for mask
#     def __init__(self, prompts: List[str], words: [List[List[str]]], tokenizer, NUM_DDIM_STEPS, subtract_words=None,
#                  start_blend=0.2, th=(.3, .3), MAX_NUM_WORDS=77, device=None, dtype=None):
#         alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
#         for i, (prompt, words_) in enumerate(zip(prompts, words)):
#             if type(words_) is str:
#                 words_ = [words_]
#             for word in words_:
#                 ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
#                 alpha_layers[i, :, :, :, :, ind] = 1
#
#         if subtract_words is not None:
#             subtract_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
#             for i, (prompt, words_) in enumerate(zip(prompts, subtract_words)):
#                 if type(words_) is str:
#                     words_ = [words_]
#                 for word in words_:
#                     ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
#                     subtract_layers[i, :, :, :, :, ind] = 1
#             self.subtract_layers = subtract_layers.to(device).to(dtype)
#         else:
#             self.subtract_layers = None
#         self.alpha_layers = alpha_layers.to(device).to(dtype)
#         self.start_blend = int(start_blend * NUM_DDIM_STEPS)
#         self.counter = 0
#         self.th = th

class LocalBlend:

    def get_mask(self, x_t, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = F.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = F.interpolate(maps, size=(x_t.shape[2:]))
        # TODO maybe use gaussian smoothing here?
        if True:
            pass
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1 - int(use_pool)])
        mask = mask[:1] + mask
        return mask

    def __call__(self, x_t, attention_store, MAX_NUM_WORDS=77):
        self.counter += 1
        if self.counter > self.start_blend:

            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(x_t, maps, self.alpha_layers, True)
            if self.subtract_layers is not None:
                maps_sub = ~self.get_mask(x_t, maps, self.subtract_layers, False)
                mask = mask * maps_sub
            mask = mask.to(x_t.dtype)
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    # th is threshold for mask
    def __init__(self, prompts: List[str], words: [List[List[str]]], tokenizer, NUM_DDIM_STEPS, subtract_words=None,
                 start_blend=0.2, th=(.3, .3), MAX_NUM_WORDS=77, device=None, dtype=None):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if subtract_words is not None:
            subtract_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, subtract_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    subtract_layers[i, :, :, :, :, ind] = 1
            self.subtract_layers = subtract_layers.to(device).to(dtype)
        else:
            self.subtract_layers = None
        self.alpha_layers = alpha_layers.to(device).to(dtype)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0
        self.th = th


class EmptyControl:

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class SpatialReplace(EmptyControl):

    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float, NUM_DDIM_STEPS):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)


class AttentionControl(abc.ABC):

    def step_callback(self, i, t, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        # TODO better way of reseting, if we choose to do diff number of steps this will fuck up
        if self.cur_step == self.num_steps:
            self.cur_step = 0
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.num_uncond_att_layers = 0
        self.total_att_layers = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    # i and t not used, this is just so we can plug it into the callback method of standard SD pipeline
    def step_callback(self, i, t, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        #TODO consider swapping uncond self attn too?
        if att_replace.shape[2] <= self.threshold_res ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace

    def set_uncond_layers(self, num):
        self.num_uncond_att_layers = num
        self.num_att_layers = self.total_att_layers - num

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    #
    # def forward(self, attn, is_cross: bool, place_in_unet: str):
    #     super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
    #     if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
    #         h = attn.shape[0] // (self.batch_size)
    #         attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
    #         attn_base, attn_replace = attn[0], attn[1:]
    #         if is_cross:
    #             alpha_words = self.cross_replace_alpha[self.cur_step]
    #             attn_replace_new = self.replace_cross_attention(attn_base, attn_replace) * alpha_words + (
    #                         1 - alpha_words) * attn_replace
    #             attn[1:] = attn_replace_new
    #         else:
    #             attn[1:] = self.replace_self_attention(attn_base, attn_replace, place_in_unet)
    #         attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
    #     return attn

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]) or (self.num_cross_replace[0] <= self.cur_step < self.num_cross_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_replace = attn[0], attn[1:]
            if is_cross and (self.num_cross_replace[0] <= self.cur_step < self.num_cross_replace[1]):
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_replace_new = self.replace_cross_attention(attn_base, attn_replace) * alpha_words + (
                            1 - alpha_words) * attn_replace
                attn[1:] = attn_replace_new
            elif (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
                attn[1:] = self.replace_self_attention(attn_base, attn_replace, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int, tokenizer,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend], device=None, dtype=None, threshold_res=32):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps,
                                                                            tokenizer).to(device).to(dtype)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        if type(cross_replace_steps) is float:
            cross_replace_steps = 0, self_replace_steps
        self.num_cross_replace = int(num_steps * cross_replace_steps[0]), int(num_steps * cross_replace_steps[1])
        self.local_blend = local_blend
        self.threshold_res = threshold_res
        self.num_steps = num_steps


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        #TODO i dont entirely understand this indexing here
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, tokenizer, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, device=None, dtype=None, threshold_res=32):
        super(AttentionRefine, self).__init__(prompts, num_steps, tokenizer, cross_replace_steps, self_replace_steps,
                                              local_blend, device=device, dtype=dtype, threshold_res=threshold_res)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1]).to(device).to(dtype)


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, tokenizer, cross_replace_steps: float, self_replace_steps: float,
                 equalizer, local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None,
                 device=None, dtype=None):
        super(AttentionReweight, self).__init__(prompts, num_steps, tokenizer, cross_replace_steps, self_replace_steps,
                                                local_blend, device=device, dtype=dtype)
        self.equalizer = equalizer.to(device).to(dtype)
        self.prev_controller = controller


def make_controller(prompts, tokenizer, NUM_DDIM_STEPS, cross_replace_steps: Dict[str, float],
                    self_replace_steps: float, blend_words=None, substruct_words=None, start_blend=0.2, th=(.3, .3),
                    device=None, dtype=None, equalizer=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words, tokenizer, NUM_DDIM_STEPS, subtract_words=substruct_words,
                        start_blend=start_blend, th=th, device=device, dtype=dtype)
    # if is_replace_controller:
    #     controller = AttentionReplace(prompts, NUM_DDIM_STEPS, tokenizer, cross_replace_steps=cross_replace_steps,
    #                                   self_replace_steps=self_replace_steps, local_blend=lb, device=device, dtype=dtype)
    # else:
    controller = AttentionRefine(prompts, NUM_DDIM_STEPS, tokenizer, cross_replace_steps=cross_replace_steps,
                                 self_replace_steps=self_replace_steps, local_blend=lb, device=device, dtype=dtype)
    if equalizer is not None:
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, tokenizer, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=equalizer, local_blend=lb,
                                       controller=controller, device=device, dtype=dtype)
    return controller


class AttentionJustReweight:

    def set_uncond_layers(self, num):
        self.num_uncond_att_layers = num
        self.num_att_layers = self.total_att_layers - num

    def __call__(self, attn, is_cross: bool, place_in_unet: str):

        # reverse the range of steps, we want it so that beginning steps follow original prompt, later are weighted
        if self.cur_att_layer >= self.num_uncond_att_layers and is_cross and self.cur_step > self.cross_replace_steps:
            h = attn.shape[0] // (self.batch_size)
            cond_attn = attn[h // 2:]

            cond_attn = cond_attn.reshape(self.batch_size, h//2, *attn.shape[1:])
            old_mean = cond_attn.mean()
            cond_attn = cond_attn[:, :, :, :] * self.equalizer[:, None, None, :]
            new_mean = cond_attn.mean()
            if self.normalize: #TODO maybe norm by the max, but only for the probs, to match softmax behavior
                cond_attn = cond_attn / (new_mean/old_mean)
            cond_attn = cond_attn.reshape(self.batch_size * (h//2), *attn.shape[1:])

            attn[h // 2:] = cond_attn

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
        if self.cur_step == self.num_steps:
            self.cur_step = 0
        return attn

    def __init__(self, batch_size, num_steps: int, cross_replace_steps: float, equalizer, device=None, dtype=None, normalize=False):
        self.equalizer = equalizer.repeat(batch_size, 1).to(device).to(dtype)
        self.cross_replace_steps = int(cross_replace_steps * num_steps)
        self.num_steps = num_steps
        self.cur_att_layer = 0
        self.cur_step = 0
        self.num_att_layers = 0
        self.num_uncond_att_layers = 0
        self.total_att_layers = 0
        self.batch_size = batch_size
        self.normalize = normalize

# class AttentionReplace(AttentionControlEdit):
#
#     def replace_cross_attention(self, attn_base, att_replace):
#         return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
#
#     def __init__(self, prompts, num_steps: int, tokenizer, cross_replace_steps: float, self_replace_steps: float,
#                  local_blend: Optional[LocalBlend] = None, device=None, dtype=None):
#         super(AttentionReplace, self).__init__(prompts, num_steps, tokenizer, cross_replace_steps, self_replace_steps, local_blend, device=device, dtype=dtype)
#         self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device).to(dtype)


# def aggregate_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
#     out = []
#     attention_maps = attention_store.get_average_attention()
#     num_pixels = res ** 2
#     for location in from_where:
#         for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
#             if item.shape[1] == num_pixels:
#                 cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
#                 out.append(cross_maps)
#     out = torch.cat(out, dim=0)
#     out = out.sum(0) / out.shape[0]
#     return out.cpu()

# def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
#     tokens = tokenizer.encode(prompts[select])
#     decoder = tokenizer.decode
#     attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
#     images = []
#     for i in range(len(tokens)):
#         image = attention_maps[:, :, i]
#         image = 255 * image / image.max()
#         image = image.unsqueeze(-1).expand(*image.shape, 3)
#         image = image.numpy().astype(np.uint8)
#         image = np.array(Image.fromarray(image).resize((256, 256)))
#         image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
#         images.append(image)
#     ptp_utils.view_images(np.stack(images, axis=0))
#
#
# def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
#                              max_com=10, select: int = 0):
#     attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape(
#         (res ** 2, res ** 2))
#     u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
#     images = []
#     for i in range(max_com):
#         image = vh[i].reshape(res, res)
#         image = image - image.min()
#         image = 255 * image / image.max()
#         image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
#         image = Image.fromarray(image).resize((256, 256))
#         image = np.array(image)
#         images.append(image)
#     ptp_utils.view_images(np.concatenate(images, axis=1))


