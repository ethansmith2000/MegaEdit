
#ATTN REWEIGHTING - in isolation
from parse_prompt import get_prompts_with_weights, pad_tokens_and_weights
prompt = ["a plate of [[[[[eggs]]]]] and (((((bacon)))))"]
tokens, weights = get_prompts_with_weights(pipe, prompt, 77)
prompt = pipe.tokenizer.batch_decode(tokens)
tokens, weights = pad_tokens_and_weights(tokens, weights, 77, pipe.tokenizer.bos_token_id, pipe.tokenizer.eos_token_id)
weights = torch.tensor(weights).to("cuda")

# first number is batch size, 2nd is num steps, 3rd is percent of run when it begins to reweight, so 30% into run
attn_controller = AttentionJustReweight(1, 50, 0.3, weights, normalize=True)
register_attention_control(pipe, attn_controller)


#full self + cross attn controller - refining + reweighting
# includes self + cross
prompts = ["an ice warrior in a frozen world, fantasy", "a mystical lava mage from a world of fire"]
# prompts, words to isolate the edit to, tokenizer, num steps) - may need to adjust some other params?
lb = LocalBlend(prompts, ((('warrior',), ("warrior",))), pipe.tokenizer, 50)

# REFINE - i dont know how to explain this one but basically general use case
# prompts, steps, tokenizer, cross_replace ratio (% into run when it begins), self_replace ratio
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

#edict version is this
# init_image is just a regular PIL image
latents = do_inversion(init_image, base_prompt, init_image_strength=1.0):







