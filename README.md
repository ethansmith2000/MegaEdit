# MegaEdit

A collection of works on inversion and diffusion image editing via feature/attention injection.

![barack2 (5) copy](https://user-images.githubusercontent.com/98723285/224556125-bb031128-d9d7-49e3-82b8-dcd6dfa0093b.png)
![land](https://user-images.githubusercontent.com/98723285/224556442-f0357d36-60df-4d79-8303-fd4dcfd3c164.jpeg)

colab demo: https://drive.google.com/file/d/1qBPkSVr2zf-KyQgdeke4fBUFLc8cWnWu/view?usp=sharing

*special thanks to the team at Leonardo.ai for helping me put this together*

NOTE: this is not compatible with Xformers, but it does support sliced attention if you are experiencing memory issues

This repo was originally based off of prompt2prompt but contains a number of improvements and implementations of other papers + some of my own stuff

This includes:
  - injection of convolution features (https://arxiv.org/abs/2211.12572)
  - originally used EDICT for inversion but now the args passed make it just do standard DDIM inversion (https://github.com/salesforce/EDICT)

My own addons include:
  - injecting an interpolation of original and proposed features, which is on a schedule. This allows us to have influence from the original features much further into the generation without fully taking over the generation. This gradual approach may confer similar benefits to (https://github.com/pix2pixzero/pix2pix-zero)
  - split guidance scale. This allows to do inversion without classifier free guidance for stability, but do editing at a different guidance scale
  - Gaussian Smoothed attention. the original intention behind this was to allow attention to cover more ground before amplifying it. Instead, I am noticing less erratic details and less of a photobashed look. See the examples below.
  - (WIP) An attempt at gradient-free attend and excite by locally amplifying attention in a region of the image. This isn't optimal as original method optimizes latents, but hope that giving special care to certain tokens can help give a simiilar effect without adding too much time/VRAM
  - Some other QoL improvements for easy deployment and demystifying some of the parameters

Usage:
1. set up torch environment of choice
2. git clone this repo
3. pip install -r requirements.txt
4. run the notebook!

Smoothing example:
<img width="1310" alt="Screen Shot 2023-02-21 at 5 47 16 PM" src="https://user-images.githubusercontent.com/98723285/224555971-6e964c6a-327f-4b40-a507-ff4e97d6685b.png">


Other editing examples

<img width="640" alt="david edits" src="https://user-images.githubusercontent.com/98723285/224555364-d7505bb9-c918-4c96-bade-313f4ad073ca.png">
<img width="640" alt="david refines" src="https://user-images.githubusercontent.com/98723285/224555383-2e0a67bf-f0b4-4cec-b06c-2ff859e15c24.png">

<img width="1278" alt="superhero" src="https://user-images.githubusercontent.com/98723285/224556519-61ce8daa-8202-4541-a545-5faa30cb95c7.png">

![download (25)](https://user-images.githubusercontent.com/98723285/224556727-a75ccab9-885a-4d4d-b828-7367123c45bb.png)

<img width="1030" alt="link edits" src="https://user-images.githubusercontent.com/98723285/224556758-b603cbed-d78b-4540-b82b-7dd824ef5b3f.png">

<img width="1030" alt="Me" src="https://user-images.githubusercontent.com/98723285/224556960-ddc8ff50-f81e-4796-92f7-b6fe156c5965.png">

Usefulness of attention reweighing, an alternative to how automatic1111 does it which is at the text encoder level, and better solution for when SD isn't listening to your prompt.

<img width="1248" alt="Screen Shot 2023-02-03 at 2 24 24 PM" src="https://user-images.githubusercontent.com/98723285/224557263-b4276004-ce41-44b6-95ef-d81607f0cfe5.png">



