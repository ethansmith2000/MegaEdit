# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, Dict
from IPython.display import display
import torch.nn.functional as F


def get_schedule(num_steps, start, end, start_buffer=0, start_buffer_value=None):
    if start_buffer_value is None:
        start_buffer_value = start
    if num_steps <= start_buffer:
        diff = start_buffer - num_steps
        num_steps = num_steps + diff + 1
    else:
        diff = 0
    schedule = np.linspace(start, end, num_steps - start_buffer).tolist()
    schedule = [start_buffer_value] * (start_buffer - diff) + schedule
    return schedule


def register_attention_control(model, controller, res_skip_layers=3):
    def ca_forward(self, place_in_unet, controller):

        class EditCrossAttnProcessor:
            def __call__(
                    self,
                    attn,
                    hidden_states,
                    encoder_hidden_states=None,
                    attention_mask=None,
            ):
                batch_size, sequence_length, _ = hidden_states.shape
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                query = attn.to_q(hidden_states)

                is_cross = True  # assume cross until it isn't, kinda a scrappy method
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                    is_cross = False
                elif attn.cross_attention_norm:
                    encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

                attention_probs = attn.get_attention_scores(query, key, attention_mask)
                attention_probs = controller(attention_probs, is_cross, place_in_unet)

                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = attn.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                return hidden_states

        class EditSlicedAttnProcessor:
            def __init__(self, slice_size):
                self.slice_size = slice_size

            def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
                batch_size, sequence_length, _ = hidden_states.shape

                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                query = attn.to_q(hidden_states)
                dim = query.shape[-1]
                query = attn.head_to_batch_dim(query)

                is_cross = True  # assume cross until it isn't, kinda a scrappy method
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                    is_cross = False
                elif attn.cross_attention_norm:
                    encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

                batch_size_attention = query.shape[0]
                hidden_states = torch.zeros(
                    (batch_size_attention, sequence_length, dim // attn.heads), device=query.device, dtype=query.dtype
                )

                for i in range(hidden_states.shape[0] // self.slice_size):
                    start_idx = i * self.slice_size
                    end_idx = (i + 1) * self.slice_size

                    query_slice = query[start_idx:end_idx]
                    key_slice = key[start_idx:end_idx]
                    attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

                    attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
                    attn_slice = controller(attn_slice, is_cross, place_in_unet)

                    attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

                    hidden_states[start_idx:end_idx] = attn_slice

                hidden_states = attn.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                return hidden_states

        if self.processor.__class__.__name__ == 'CrossAttnProcessor' or self.processor.__class__.__name__ == 'EditCrossAttnProcessor':
            self.processor = EditCrossAttnProcessor()
        elif self.processor.__class__.__name__ == 'SlicedAttnProcessor' or self.processor.__class__.__name__ == 'EditSlicedAttnProcessor':
            self.processor = EditSlicedAttnProcessor(self.processor.slice_size)

    def res_forward(self):

        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            hidden_states = controller(hidden_states, None, None)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            # output_tensor = controller(output_tensor)

            return output_tensor

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0
            self.num_res_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            ca_forward(net_, place_in_unet, controller)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    def find_res(module, count=[]):
        if module.__class__.__name__ == 'ResnetBlock2D':
            module.forward = res_forward(module)
            if len(count) > res_skip_layers:
                count.append(1)
            else:
                count.append(0)

        elif hasattr(module, 'children'):
            for x in module.children():
                find_res(x, count)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

    res_count = find_res(model.unet.up_blocks)
    controller.num_conv_layers = sum(res_count)


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor] = None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)