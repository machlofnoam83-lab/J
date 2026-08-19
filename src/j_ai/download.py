from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the JAI base model ahead of training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--minimum-free-gb", type=float, default=8.0)
    args = parser.parse_args()

    cache_home = Path.home() / ".cache" / "huggingface"
    free_gb = shutil.disk_usage(cache_home.parent).free / (1024**3)
    if free_gb < args.minimum_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB are free. At least {args.minimum_free_gb:.1f} GB are required."
        )

    from .model import ensure_model_downloaded

    location = ensure_model_downloaded(args.model)
    print(f"ההורדה הסתיימה בהצלחה. קבצי המודל נמצאים ב־{location}")


if __name__ == "__main__":
    main()
