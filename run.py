import torch
import os
from parse_prompt import get_weighted_text_embeddings
import diffusers
import transformers
from p2p_pipeline import P2pPipeline
from p2p.edict import do_inversion
from p2p.ptp_utils import register_attention_control, get_schedule
from p2p.prompt2prompt import make_controller
from p2p.prompt2prompt import LocalBlend
from p2p.prompt2prompt import get_equalizer
from PIL import Image


class MegaEdit:
    def __init__(self, model_path_or_repo, device="cuda", dtype=torch.float16, attn_slices="0"):
        self.pipe = P2pPipeline.from_pretrained(model_path_or_repo).to(device).to(dtype)
        self.pipe.safety_checker = None # get that pesky thing out of here!
        self.inversion_prompt = None
        self.height = 512
        self.width = 512
        self.latents = None
        self.device = device
        self.dtype = dtype
        if attn_slices != "0":
            self.pipe.set_attention_slice(attn_slices)

    def invert_image(self, path_to_image, prompt, steps=50, width=512, height=512):
        # these params do well, we can improve if we do begin inversion at half noise level (x_T/2) for example rather than x0, but i've botched it for now
        init_image = Image.open(path_to_image).convert("RGB").resize((width, height))
        latents = do_inversion(self.pipe, init_image, prompt,
                               height=height, width=width,
                               end_noise=1.0,
                               begin_noise=0.0,
                               steps=50,
                               mix_weight=1.0, guidance_scale=1.0)
        self.latents = latents[0].repeat(2, 1, 1, 1)
        self.width = width
        self.height = height
        self.inversion_prompt = prompt
        self.steps = steps

    def run_edit(self,
                 prompt, # str
                 local_edit_word=None, #str
                 invert_local_edit=False,
                 neg_prompt="bad quality, low resolution, jpg artifacts",
                 cross_replace_steps=0.5,  # 0.0 - 0.5 is good
                 self_replace_steps=0.65,  # 0.25-0.65
                 conv_replace_steps=0.55,  # 0.25-0.6, typically like this one lower than self replace
                 ):
        if self.latents is None:
            raise Exception("You must invert an image before running an edit!")

        prompts = [self.inversion_prompt, prompt]

        # OPTIONAL the first word you put will be the area of the image that gets edited, 2nd doesn't matter lol
        # if you do invert mask it will edit everywhere but the area you specify.

        # in the future, rather than doing
        # this goofy method of weighting words, we can just borrow the parse_prompt code
        i, k, eq = get_weighted_text_embeddings(self.pipe, prompts[1])
        eof = torch.where(k == 49407)[1][0].item()  # find first end of phrase tokens

        parsed_prompt = self.pipe.tokenizer.batch_decode(k[:, 1:eof])[0]

        prompts[1] = parsed_prompt

        # these allow us to inject blend of original features with proposed features
        # value of 1 is purely the original features, 0 is purely new
        # buffer is number of steps for which we hold a constant value specified by start buffer value
        # from there it linearly interpolates from start to end value
        # cross i wouldn't mess with a ton
        cross_schedule = {
            "start": 1.0,
            "end": 0.7,
            "start_buffer": 0,
            "start_buffer_value": 1.0,
        }

        # can try lowering start/start_buffer_value a tad,
        # but biggest thing to experiment with is the end value for which 0.2-0.7 are interesting
        self_schedule = {
            "start": 1.0,
            "end": 0.45,
            "start_buffer": int(self.steps * 0.2),
            "start_buffer_value": 1.0,
        }
        conv_schedule = {
            "start": 1.0,
            "end": 0.45,
            "start_buffer": int(self.steps * 0.2),
            "start_buffer_value": 1.0,
        }

        # all of the _____replace_steps are percent into run when we stop injecting
        attn_controller = make_controller(prompts,
                                          self.pipe.tokenizer, self.steps,
                                          cross_replace_steps=cross_replace_steps,
                                          self_replace_steps=self_replace_steps,
                                          device=self.device,
                                          dtype=self.dtype,
                                          threshold_res=0, # 0, 1, or 2 at 0 attn injection is done at all layers, at 1 the largest resolution layer is skipped, at 2 only the smallest attn layers will do it
                                          # the layers with the highest attn size is image size /8, layers below specified value will be skipped
                                          conv_replace_steps=conv_replace_steps,
                                          equalizer=eq,
                                          conv_mix_schedule=conv_schedule,
                                          cross_attn_mix_schedule=cross_schedule,
                                          self_attn_mix_schedule=self_schedule,
                                          blend_words=local_edit_word,
                                          image_size=(self.width, self.height),
                                          smooth_steps=0.5,
                                          invert_mask=invert_local_edit,
                                          )
        register_attention_control(self.pipe, attn_controller,res_skip_layers=2)
        # res skip layers determines how many upblock conv layers receive injection, res_skip_layers=2 will skip first 3 as per paper

        # neg_prompt = ""
        img = self.pipe(prompts, latents=self.latents,
                   negative_prompt=[neg_prompt] * 2, guidance_scale=(1, 8),
                   # we have to use a special pipeline to allow for us to use diff guidance scales for each image
                   width=self.width,
                   height=self.height,
                   num_inference_steps=self.steps,
                   callback=attn_controller.step_callback).images  # the call back is so that we can do local blend, basically locally edit image

        return img # returns a list of 2 images, the first is the reconstruction, the second is the edited image