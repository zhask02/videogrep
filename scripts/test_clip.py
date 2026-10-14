"""Smoke-test pretrained CLIP on one local frame and one text query."""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel

MODEL_ID = "openai/clip-vit-base-patch32"
FRAME_DIR = Path("data/cache/frames")


def main() -> None:
    frames = sorted(FRAME_DIR.glob("*/frame_*.jpg"))
    if not frames:
        raise SystemExit(
            "No extracted frames found. Run scripts/extract_frames.py first."
        )

    image_path = frames[0]
    query = "a person in a video"

    print(f"Loading model: {MODEL_ID}")
    print("The first run may download model files.")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=[query],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    # inference_mode disables gradient tracking because we are not training.
    with torch.inference_mode():
        outputs = model(**inputs)

        # Normalize vectors so their dot product equals cosine similarity.
        image_embedding = F.normalize(outputs.image_embeds, p=2, dim=-1)
        text_embedding = F.normalize(outputs.text_embeds, p=2, dim=-1)
        similarity = (image_embedding @ text_embedding.T).item()

    print("\nCLIP smoke test passed")
    print(f"Frame: {image_path}")
    print(f"Image embedding shape: {tuple(image_embedding.shape)}")
    print(f"Text embedding shape:  {tuple(text_embedding.shape)}")
    print(f"Cosine similarity:     {similarity:.4f}")
    print(f"Device:                {next(model.parameters()).device}")


if __name__ == "__main__":
    main()
