
#ATTN REWEIGHTING - in isolation
from parse_prompt import get_prompts_with_weights, pad_tokens_and_weights
prompt = ["a plate of [[[[[eggs]]]]] and (((((bacon)))))"]
tokens, weights = get_prompts_with_weights(pipe, prompt, 77)
prompt = pipe.tokenizer.batch_decode(tokens)
tokens, weights = pad_tokens_and_weights(tokens, weights, 77, pipe.tokenizer.bos_token_id, pipe.tokenizer.eos_token_id)
weights = torch.tensor(weights).to("cuda")

# first number is batch size, 2nd is num steps, 3rd is percent of run when it begins to reweight, so 30% into run
attn_controller = AttentionJustReweight(1, 50, 0.3, weights, normalize=False)
register_attention_control(pipe, attn_controller)


#full self + cross attn controller - refining + reweighting
# includes self + cross
prompts = ["an ice warrior in a frozen world, fantasy", "a mystical lava mage from a world of fire"]
# prompts, words to isolate the edit to, tokenizer, num steps) - may need to adjust some other params?
lb = LocalBlend(prompts, ((('warrior',), ("warrior",))), pipe.tokenizer, 50)

# REFINE - i dont know how to explain this one but basically general use case
# prompts, steps, tokenizer, cross_replace ratio (% into run when it STOPS - opposite to what reweight does), self_replace ratio
attn_refine=AttentionRefine(prompts, 50, pipe.tokenizer, 0.4, 0.4, local_blend=None ,device="cuda", dtype=torch.float16, threshold_res=32)

# you can also use the make_controller function, using the reweighting-refine hybrid requires it to be done this way


# RESNET CONTROLLER
# steps, ratio (% into run when it begins)
res_controller = FeatControl(50, 0.3)
# skip_layers is the number of resnet layers to skip before starting to inject features
register_res_control(pipe, res_controller, skip_layers=3)


# you can test inversion either with regular inversion or edict inversion
# this one returns full list of latents, so we can use latents[-1] to get the init noise
# this image is preprocessed by this
image = load_512(image_path)
latents = ddim_inversion(pipe, image, prompt, steps):


#edict version is this - THIS IS THE BETTER VERSION
# init_image is just a regular PIL image
latents = do_inversion(init_image, base_prompt, init_image_strength=1.0, mix_weight=1.0)

# I am also testing to see if it might be better to invert only to 0.95, but then this would require to run
#img2img pipeline at same strength



#_________________ full example
import torch
import diffusers
import transformers

from p2p.EDICT import do_inversion
from p2p.feat_share import register_res_control, FeatControl
from p2p.ptp_utils import register_attention_control
from p2p.prompt2prompt import AttentionRefine


pipe = diffusers.StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda").to(torch.float16)

attn_controller = AttentionRefine(prompts, 50, pipe.tokenizer,
                                  cross_replace_steps=0.2, #
                                  self_replace_steps=0.3,
                                 device="cuda",
                                 dtype=torch.float16)
register_attention_control(pipe, attn_controller)
#register_attention_control(pipe, None)

res_controller = FeatControl(50, 0.35)
register_res_control(pipe, res_controller, skip_layers=3)
#register_res_control(pipe, None)


# torch.manual_seed(31)
# latents = torch.randn(1, 4, 64, 64).expand(2, 4, 64, 64).to("cuda").to(torch.float16)

latents = do_inversion(init_image, base_prompt, init_image_strength=1.0, mix_weight=1.0)

latents = latents * pipe.scheduler.init_noise_sigma
neg_prompt = "bad quality, low resolution, jpg artifacts"
neg_prompt = ""
img = pipe(prompts, latents=latents[0].repeat(2, 1, 1, 1), negative_prompt=[neg_prompt] * 2, guidance_scale=3).images







