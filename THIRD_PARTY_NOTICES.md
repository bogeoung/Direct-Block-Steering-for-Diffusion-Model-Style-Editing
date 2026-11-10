# Third-party notices

No third-party source code, model weights, checkpoints, or generated data are
included in this repository. The DBsteer implementation interoperates with the
following separately obtained dependencies.

| Dependency | Purpose | License / source |
| --- | --- | --- |
| SDXL-Unbox | Hook-enabled SDXL pipeline used at runtime | MIT; https://github.com/surkovv/sdxl-unbox |
| Stable Diffusion XL Base 1.0 | Diffusion model weights | https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 |
| SDXL fp16-fix VAE | VAE used for fp16 generation | https://huggingface.co/madebyollin/sdxl-vae-fp16-fix |
| Diffusers / PyTorch | Model loading and execution | Respective upstream licenses |

StyleAligned, sparse autoencoder code/checkpoints, CLIP, DINOv2, LPIPS, CSD,
and their wrappers are intentionally absent because this export contains no
baseline comparison or metric-evaluation implementation.
