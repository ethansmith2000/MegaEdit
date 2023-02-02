import torch.nn.functional as F
import numpy as np
import abc
import p2p.ptp_utils as ptp_utils
import p2p.seq_aligner as seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
import torch


class AttentionControlEdit:

    def replace_self_attention(self, attn_base, attn_replace, place_in_unet):
        if True:  # attn_replace.shape[2] <= 32 ** 2:
            # attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return attn_replace

    def __init__(self, batch_size, num_steps: int, self_replace_steps, before):
        self.batch_size = batch_size
        self.num_self_replace = int(self_replace_steps * num_steps)
        self.local_blend = None
        self.num_steps = num_steps
        self.LOW_RESOURCE = False
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.before = before

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # h = attn.shape[0]
        # attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        cond = self.cur_step >= self.num_self_replace if not self.before else self.cur_step < self.num_self_replace
        if cond:
            # h = attn.shape[0] // (self.batch_size * 2)
            # batch_size, seq_len, dim = attn.shape
            # attn = attn.reshape(batch_size // h, h, seq_len, dim)
            # attn = attn.permute(0, 2, 1, 3).reshape(batch_size // h, seq_len, dim * h)

            # TODO this is wildly inefficient
            mask = [1] * 8 + [0] * 8 + [1] * 8 + [0] * 8
            mask = torch.tensor(mask, dtype=bool)
            attn_base, attn_replace = attn[mask], attn[~mask]
            if is_cross:
                # alpha_words = self.cross_replace_alpha[self.cur_step]
                # attn_replace_new = self.replace_cross_attention(attn_base, attn_replace) * alpha_words + (1 - alpha_words) * attn_replace
                # attn[1:] = attn_replace_new
                pass
            else:
                attn[~mask] = self.replace_self_attention(attn_base, attn_replace, place_in_unet)

            # batch_size, seq_len, dim = attn.shape
            # attn = attn.reshape(batch_size, seq_len, h, dim // h)
            # attn = attn.permute(0, 2, 1, 3).reshape(batch_size * h, seq_len, dim // h)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # self.between_steps()
        if self.cur_step == self.num_steps:
            self.cur_step = 0
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class AttentionReplace(AttentionControlEdit):

    def __init__(self, batch_size, num_steps: int, self_replace_steps: float, before=True):
        super(AttentionReplace, self).__init__(batch_size, num_steps, self_replace_steps, before)


# ----

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def _attention(query, key, value, is_cross, attention_mask=None):
            if self.upcast_attention:
                query = query.float()
                key = key.float()

            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query,
                key.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            if self.upcast_softmax:
                attention_scores = attention_scores.float()

            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            # cast back to the original dtype
            attention_probs = attention_probs.to(value.dtype)

            # compute attention output
            hidden_states = torch.bmm(attention_probs, value)

            # reshape hidden_states
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            return hidden_states

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, _ = hidden_states.shape

            if encoder_hidden_states is not None:
                is_cross = True
            else:
                is_cross = False

            encoder_hidden_states = encoder_hidden_states

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)
            dim = query.shape[-1]
            query = self.reshape_heads_to_batch_dim(query)

            if self.added_kv_proj_dim is not None:
                key = self.to_k(hidden_states)
                value = self.to_v(hidden_states)
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                key = self.reshape_heads_to_batch_dim(key)
                value = self.reshape_heads_to_batch_dim(value)
                encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
                encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

                key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
                value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
            else:
                encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                key = self.reshape_heads_to_batch_dim(key)
                value = self.reshape_heads_to_batch_dim(value)

            if attention_mask is not None:
                if attention_mask.shape[-1] != query.shape[1]:
                    target_length = query.shape[1]
                    attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                    attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

            # attention, what we cannot get enough of
            if self._use_memory_efficient_attention_xformers:
                hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
            else:
                if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                    hidden_states = self._attention(query, key, value, is_cross, attention_mask)
                else:
                    hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)

            # dropout
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states

        return _attention, forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_._attention, net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
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
