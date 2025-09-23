# Download SCAM dataset and generate image folders with optimized images for the project page

from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

dataset = load_dataset("BLISS-e-V/SCAM", split="train")

for elem in tqdm(dataset):
    ds_type = elem["type"]
    img = elem["image"].convert("RGB").resize((384, 384))
    output_path = Path(f"data_images/{ds_type}/{elem['id']}.webp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="webp", quality=70)
