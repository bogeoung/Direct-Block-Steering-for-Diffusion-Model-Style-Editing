"""Estimate raw block-update statistics for a neutral/style prompt pair."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dbsteer.prompts import EXTRACTION_CONTENTS, NEUTRAL_TEMPLATE, STYLE_TEMPLATES
from dbsteer.runtime import BLOCKS, RawUpdateCapture, generate, load_pipeline


def prompt_for(condition: str, content: str) -> str:
    if condition == "photo":
        return NEUTRAL_TEMPLATE.format(content=content)
    return STYLE_TEMPLATES[condition].format(content=content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", required=True,
                        help="Use photo plus one or more style names.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--block", choices=BLOCKS, default="up.0.1")
    args = parser.parse_args()

    unknown = set(args.conditions) - {"photo", *STYLE_TEMPLATES}
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_pipeline()

    for condition in args.conditions:
        batches = []
        for start in range(0, len(EXTRACTION_CONTENTS), args.batch_size):
            indices = range(start, min(start + args.batch_size, len(EXTRACTION_CONTENTS)))
            capture = RawUpdateCapture()
            prompts = [prompt_for(condition, EXTRACTION_CONTENTS[index]) for index in indices]
            generate(model, prompts, [1000 + index for index in indices],
                     hooks={args.block: capture}, steps=args.steps)
            batches.append(capture.stacked().half())

        torch.save({
            "raw_updates": torch.cat(batches, dim=0),
            "block": args.block,
            "condition": condition,
            "prompts": [prompt_for(condition, content) for content in EXTRACTION_CONTENTS],
        }, args.output_dir / f"{condition}.pt")


if __name__ == "__main__":
    main()
