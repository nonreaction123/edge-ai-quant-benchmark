"""Create a lightweight demo GIF artifact for the portfolio repo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    font = ImageFont.load_default()
    rng = np.random.default_rng(7)

    for idx, quant in enumerate(["FP32", "Dynamic float16", "INT8"]):
        image = Image.new("RGB", (520, 260), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([20, 30, 500, 225], outline="black", width=2)
        draw.text((40, 45), "Edge AI inference demo", fill="black", font=font)
        draw.text((40, 80), f"Model variant: {quant}", fill="black", font=font)
        draw.text((40, 115), f"Predicted class: defect_type_{idx}", fill="black", font=font)
        draw.text((40, 150), f"Confidence: {0.86 + idx * 0.03:.2f}", fill="black", font=font)

        # Draw a synthetic sensor/image feature strip
        values = rng.random(18)
        for i, v in enumerate(values):
            x0 = 40 + i * 24
            y0 = 210 - int(v * 50)
            draw.rectangle([x0, y0, x0 + 14, 210], fill="gray")

        frames.append(image)

    frames[0].save(out, save_all=True, append_images=frames[1:], duration=850, loop=0)
    print(f"Saved demo GIF: {out}")


if __name__ == "__main__":
    main()
