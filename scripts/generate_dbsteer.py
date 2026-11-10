"""Generate held-out images with a single DBsteer direction."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dbsteer.prompts import EVALUATION_CONTENTS, NEUTRAL_TEMPLATE
from dbsteer.runtime import DBSteerHook, generate, load_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", required=True)
    parser.add_argument("--direction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--save-base", action="store_true")
    args = parser.parse_args()

    styled = torch.load(args.direction_dir / f"{args.style}.pt", map_location="cpu")
    neutral = torch.load(args.direction_dir / "photo.pt", map_location="cpu")
    if styled["block"] != neutral["block"]:
        raise ValueError("Styled and neutral statistics must come from the same block.")
    block = styled["block"]
    direction = (styled["raw_updates"].float() - neutral["raw_updates"].float()).mean((0, 1)).cuda()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_pipeline()
    for start in range(0, len(EVALUATION_CONTENTS), args.batch_size):
        contents = EVALUATION_CONTENTS[start:start + args.batch_size]
        seeds = [args.seed + index for index in range(start, start + len(contents))]
        prompts = [NEUTRAL_TEMPLATE.format(content=content) for content in contents]

        if args.save_base:
            base_images = generate(model, prompts, seeds, steps=args.steps)
            for index, image in enumerate(base_images, start=start):
                image.save(args.output_dir / f"base_{index:02d}.png")

        images = generate(model, prompts, seeds,
                          hooks={block: DBSteerHook(direction, args.alpha)}, steps=args.steps)
        for index, image in enumerate(images, start=start):
            image.save(args.output_dir / f"dbsteer_{args.style}_{index:02d}.png")

    torch.save({"direction": direction.cpu(), "block": block, "style": args.style},
               args.output_dir / f"direction_{args.style}.pt")


if __name__ == "__main__":
    main()
