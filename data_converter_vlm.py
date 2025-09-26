"""
SCAM Analysis Data Converter

Processes the combined results CSV file and directly generates:
- vlm_models_properties.json: Model properties in JSON format
- vlm_similarity_metadata.json: Metadata about similarity scores
- vlm_similarity_data.bin: Binary file with similarity scores
- vlm_similarity_index.json: Index file with image metadata, grouped by base image

Usage: python data_converter_vlm.py INPUTFILE
"""

import pandas as pd
import numpy as np
import json
import struct
import argparse
import sys
from time import time
from pathlib import Path


def process_input_csv(input_file: Path):
    """Process the input CSV and return dataframes for models and similarity data."""
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found.")
        return None, None

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    # filter datasets
    datasets_to_use = ["SCAM", "NoSCAM", "SynthSCAM"]
    df_ignored = df[~df["dataset"].isin(datasets_to_use)]
    if len(df_ignored) > 0:
        print(f"Ignoring {len(df_ignored)} rows with datasets not in {datasets_to_use}")
        print(f"Ignored datasets: {df_ignored['dataset'].unique()}")
    df = df[df["dataset"].isin(datasets_to_use)]

    # Define model properties columns
    model_columns = [
        "model",
        "image_size",
        "mparams",
        "gflops",
        "pretraining_data",
        "pretraining_dataset_size_estimate",
        "pretraining_dataset_size_estimate_numeric",
    ]

    # Extract unique model configurations
    models_df = df[model_columns].drop_duplicates().reset_index(drop=True)
    print(f"Found {len(models_df)} unique model configurations.")

    # Extract image-specific columns and metrics
    image_base_columns = [
        "dataset",
        "object_label",
        "attack_word",
        "postit_area_pct",
    ]
    metric_columns = [
        "object_similarities",
        "attack_similarities",
        "object_prob",
        "attack_prob",
    ]

    # Extract image_id from image_path - use the filename as the image_id
    df["image_id"] = df["image_path"].apply(lambda x: Path(x).name if isinstance(x, str) else None)
    # Turn NoScam and SynthSCAM all into SCAM, beacuse uniqueness will come from next step
    df["image_id"] = df["image_id"].str.replace("NoSCAM", "SCAM").str.replace("SynthSCAM", "SCAM")

    # Create unique image identifier based on image_id and dataset
    df["unique_id"] = df["image_id"] + "|" + df["dataset"].astype(str)
    unique_images = (
        df[["unique_id", "image_id"] + image_base_columns].drop_duplicates().reset_index(drop=True)
    )
    unique_models = df["model"].unique()

    print(f"Processing {len(unique_images)} images across {len(unique_models)} models...")

    # Build column data for each model and metric
    new_columns = {}
    for model in unique_models:
        model_data = df[df["model"] == model]
        for metric in metric_columns:
            metric_dict = dict(zip(model_data["unique_id"], model_data[metric]))
            new_columns[f"{model}_{metric}"] = unique_images["unique_id"].map(metric_dict)

    # Create final DataFrame for similarity data
    similarity_df = pd.concat(
        [unique_images.drop(columns=["unique_id"]), pd.DataFrame(new_columns)], axis=1
    )

    print("Data processing completed successfully!")
    return models_df, similarity_df


def generate_model_json_files(models_df: pd.DataFrame, models_json: Path):
    """Generate JSON files for model properties."""
    # Handle NaN values properly
    models_df = models_df.replace({np.nan: None})

    # Convert to list of dictionaries and save
    models_list = models_df.sort_values(by="model").to_dict(orient="records")

    with open(models_json, "w") as f:
        json.dump(models_list, f, indent=2)

    print(f"Saved {len(models_list)} models to JSON files")
    return True


def generate_similarity_files(
    similarity_df: pd.DataFrame, metadata_json: Path, binary_file: Path, index_json: Path
):
    """Generate binary and JSON files for similarity data."""
    # Get all columns containing 'similarities'
    similarity_columns = [col for col in similarity_df.columns if "similarities" in col]

    print(f"Processing {len(similarity_df)} rows with {len(similarity_columns)} similarity cols")

    # Create and save metadata
    metadata = {
        "columns": similarity_columns,
        "models": list(sorted(set(["_".join(col.split("_")[:-2]) for col in similarity_columns]))),
        "num_rows": len(similarity_df),
        "num_columns": len(similarity_columns),
    }
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)

    # Create binary file
    with open(binary_file, "wb") as f:
        # Write header: number of rows and columns as 32-bit integers
        f.write(struct.pack("<II", len(similarity_df), len(similarity_columns)))

        # Process and write similarity values
        similarity_values = similarity_df[similarity_columns].values.astype(np.float32)
        similarity_values = np.nan_to_num(similarity_values, nan=0.0)
        similarity_values.tofile(f)

    # Group images by their base image_id and organize variants together
    image_groups = {}

    # First pass: organize data by image_id
    for i, row in similarity_df.iterrows():
        image_id = row["image_id"] if not pd.isna(row["image_id"]) else None
        dataset = row["dataset"] if not pd.isna(row["dataset"]) else None

        if not image_id:
            continue

        if image_id not in image_groups:
            # Initialize with common data that should be the same across variants
            image_groups[image_id] = {
                "image_id": image_id,
                "object_label": (row["object_label"] if not pd.isna(row["object_label"]) else None),
                "attack_word": (row["attack_word"] if not pd.isna(row["attack_word"]) else None),
                "variants": {},
            }

        # posit area
        postit_area_pct = row["postit_area_pct"] if not pd.isna(row["postit_area_pct"]) else None
        if "postit_area_pct" not in image_groups[image_id]:
            image_groups[image_id]["postit_area_pct"] = postit_area_pct
        else:
            if image_groups[image_id]["postit_area_pct"] != postit_area_pct:
                print(
                    f"Warning: postit_area_pct mismatch for {image_id} - {postit_area_pct} != {image_groups[image_id]['postit_area_pct']}"
                )

        # Store row index for this variant
        image_groups[image_id]["variants"][dataset] = {
            "row_index": i,
        }

    # Convert to list for output
    index_data = list(image_groups.values())

    with open(index_json, "w") as f:
        json.dump(index_data, f)
    print(f"Metadata JSON: {metadata_json.stat().st_size / 1024:.1f} KB")
    print(f"Binary data: {binary_file.stat().st_size / (1024*1024):.2f} MB")
    print(f"Index data: {index_json.stat().st_size / (1024*1024):.2f} MB")

    return True


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="SCAM Analysis Data Converter")
    parser.add_argument("input", help="Path to the combined results CSV file")
    args = parser.parse_args()

    # Set file paths for output
    input_file = Path(args.input)
    output_dir = Path("data/")
    models_json = output_dir / "vlm_models_properties.json"
    metadata_json = output_dir / "vlm_similarity_metadata.json"
    binary_file = output_dir / "vlm_similarity_data.bin"
    index_json = output_dir / "vlm_similarity_index.json"

    start_time = time()

    print("=== SCAM Analysis Data Converter ===")

    # Step 1: Process the input CSV
    print("\n--- Processing input CSV ---")
    models_df, similarity_df = process_input_csv(input_file)
    if models_df is None or similarity_df is None:
        sys.exit(1)

    # Step 2: Generate model properties JSON files
    print("\n--- Generating model properties JSON files ---")
    generate_model_json_files(models_df, models_json)

    # Step 3: Generate similarity files
    print("\n--- Generating similarity files ---")
    generate_similarity_files(similarity_df, metadata_json, binary_file, index_json)

    # Report completion
    elapsed = time() - start_time
    print(f"\n=== Conversion complete in {elapsed:.2f} seconds ===")
    print("Generated files:")
    print(f"- {models_json}")
    print(f"- {metadata_json}")
    print(f"- {binary_file}")
    print(f"- {index_json}")


if __name__ == "__main__":
    main()
