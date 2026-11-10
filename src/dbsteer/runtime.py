"""Runtime helpers for raw block-update capture and DBsteer injection.

The hook-enabled pipeline is an external runtime dependency. This file contains
only the project-specific capture and injection logic.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

BLOCKS = {
    "down.2.1": "unet.down_blocks.2.attentions.1",
    "mid.0": "unet.mid_block.attentions.0",
    "up.0.0": "unet.up_blocks.0.attentions.0",
    "up.0.1": "unet.up_blocks.0.attentions.1",
}
SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
FP16_VAE = "madebyollin/sdxl-vae-fp16-fix"


def _hooked_pipeline_class():
    root = os.environ.get("SDXL_UNBOX_PATH")
    if not root:
        raise RuntimeError("Set SDXL_UNBOX_PATH to a separately installed SDXL-Unbox checkout.")
    root = str(Path(root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from SDLens.hooked_sd_pipeline import HookedStableDiffusionXLPipeline
    return HookedStableDiffusionXLPipeline


def load_pipeline(device: str = "cuda", dtype: torch.dtype = torch.float16):
    """Load SDXL with the external hook-enabled pipeline implementation."""
    from diffusers import AutoencoderKL

    pipeline_cls = _hooked_pipeline_class()
    vae = AutoencoderKL.from_pretrained(FP16_VAE, torch_dtype=dtype)
    model = pipeline_cls.from_pretrained(
        SDXL_BASE, vae=vae, torch_dtype=dtype, use_safetensors=True, variant="fp16"
    )
    model.pipe.to(device)
    model.pipe.set_progress_bar_config(disable=True)
    return model


class RawUpdateCapture:
    """Capture spatial means of a block's raw update, output minus input.

    With classifier-free guidance, the conditional half of the batch is used.
    The captured sequence has shape ``[batch, denoising_steps, channels]``.
    """
    def __init__(self, cfg: bool = True):
        self.cfg = cfg
        self.updates: list[torch.Tensor] = []

    @torch.no_grad()
    def __call__(self, module, inputs, output):
        update = output[0] - inputs[0]
        if self.cfg:
            update = update.chunk(2)[1]
        self.updates.append(update.mean(dim=(2, 3)).float().cpu())
        return output

    def stacked(self) -> torch.Tensor:
        return torch.stack(self.updates, dim=1)


class DBSteerHook:
    """Inject one fixed channel vector at every spatial position and step."""
    def __init__(self, direction: torch.Tensor, alpha: float, cfg: bool = True):
        self.direction = direction
        self.alpha = alpha
        self.cfg = cfg

    @torch.no_grad()
    def __call__(self, module, inputs, output):
        hidden = output[0]
        addition = (self.alpha * self.direction).to(hidden.dtype).view(1, -1, 1, 1)
        if self.cfg:
            unconditional, conditional = hidden.chunk(2)
            hidden = torch.cat((unconditional - addition, conditional + addition), dim=0)
        else:
            hidden = hidden + addition
        return (hidden,) + tuple(output[1:])


def generate(model, prompts, seeds, hooks=None, steps: int = 30, cfg_scale: float = 7.0,
             size: int = 1024):
    generators = [torch.Generator(device="cpu").manual_seed(int(seed)) for seed in seeds]
    kwargs = {
        "prompt": list(prompts),
        "num_inference_steps": steps,
        "guidance_scale": cfg_scale,
        "height": size,
        "width": size,
        "generator": generators,
    }
    if hooks:
        return model.run_with_hooks(
            position_hook_dict={BLOCKS[role]: hook for role, hook in hooks.items()}, **kwargs
        ).images
    return model.pipe(**kwargs).images
