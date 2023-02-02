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


def register_res_control(model, controller, skip_layers=3):
    def res_forward(self):
        # to_out = self.to_out
        # if type(to_out) is torch.nn.modules.container.ModuleList:
        #     to_out = self.to_out[0]
        # else:
        #     to_out = self.to_out

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

            hidden_states = controller(hidden_states)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            # output_tensor = controller(output_tensor)

            return output_tensor

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_res_layers = 0

    if controller is None:
        controller = DummyController()

    def find_res(module, count=[]):
        if module.__class__.__name__ == 'ResnetBlock2D':
            module.forward = res_forward(module)
            if len(count) > skip_layers:
                count.append(1)
            else:
                count.append(0)

        elif hasattr(module, 'children'):
            for x in module.children():
                find_res(x, count)
        return count

    res_count = find_res(model.unet.up_blocks)

    controller.num_res_layers = sum(res_count)


class FeatControl:

    def __call__(self, feat):
        cond = self.cur_step >= self.num_self_replace if not self.before else self.cur_step < self.num_self_replace
        cond2 = self.cur_layer > (12 - self.num_res_layers)
        if cond and cond2:
            mask = torch.tensor([1, 0, 1, 0], dtype=bool)
            feat[~mask] = feat[mask]

        self.cur_layer += 1
        if self.cur_layer == 12:
            self.cur_layer = 0
            self.cur_step += 1
        if self.cur_step == self.num_steps:
            self.cur_step = 0
        return feat

    def __init__(self, num_steps: int, self_replace_steps, before=True):
        self.num_self_replace = int(self_replace_steps * num_steps)
        self.num_steps = num_steps
        self.local_blend = None
        self.cur_layer = 0
        self.num_res_layers = 0
        self.cur_step = 0
        self.before = before

