# Download SCAM dataset and generate image folders with optimized images for the project page

from datasets import load_dataset, Image as HfImage
from pathlib import Path
from PIL import Image

dataset = load_dataset("BLISS-e-V/SCAM", split="train").cast_column("image", HfImage(decode=False))

for elem in dataset:
    ds_type = elem["type"]
    input_path = elem["image"]['path']
    name = Path(input_path).stem
    img = Image.open(input_path).convert("RGB").resize((256, 256))
    output_path = Path(f"data_images/{ds_type}/{name}.webp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="webp", quality=50)
