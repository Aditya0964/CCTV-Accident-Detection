"""Download the public YOLO11x checkpoint used by the accident benchmark."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    target = hf_hub_download(
        repo_id="Enos-123/traffic-accident-detection-yolo11x",
        filename="weights/epoch61.pt",
        local_dir=Path("outputs/model_cache/enos_yolo11x"),
    )
    print(f"YOLO11x checkpoint ready: {target}")
    print("The ViT and R3D weights download automatically on first use.")


if __name__ == "__main__":
    main()
