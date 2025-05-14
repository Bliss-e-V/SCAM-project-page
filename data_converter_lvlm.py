"""
SCAM Analysis Data Converter

Processes the combined results CSV file and directly generates:
- lvlm_models_properties.json: Model properties in JSON format
- lvlm_similarity_metadata.json: Metadata about similarity scores
- lvlm_similarity_data.bin: Binary file with similarity scores
- lvlm_similarity_index.json: Index file with image metadata, grouped by base image

Usage: python data_converter_lvlm.py INPUTFILE
"""

import pandas as pd
import numpy as np
import json
import struct
import argparse
import sys
from time import time
from pathlib import Path


def process_input_df(
    df: pd.DataFrame, filename_mapping_path, prompt_ids_to_use=[1, 5]
):
    """Process the input dataframe and return dataframes for models and similarity data."""
    # filter datasets
    datasets_to_use = ["SCAM", "NoSCAM", "SynthSCAM"]
    df_ignored = df[~df["dataset"].isin(datasets_to_use)]
    if len(df_ignored) > 0:
        print(
            f"Ignoring {len(df_ignored)} rows with datasets not in {datasets_to_use}"
        )
        print(f"Ignored datasets: {df_ignored['dataset'].unique()}")
    df = df[df["dataset"].isin(datasets_to_use)]

    # filter by prompts
    if len(prompt_ids_to_use) > 0:
        df = df[df["prompt_id"].isin(prompt_ids_to_use)]
        df["prompt_id"] = df["prompt_id"].map(
            {pid: idx for idx, pid in enumerate(prompt_ids_to_use)}
        )

    # map filenames
    if filename_mapping_path:
        tmp = pd.read_csv(filename_mapping_path)
        filename_mapping = dict(zip(tmp["old_filename"], tmp["new_filename"]))
        print(f"Mapping {len(filename_mapping)} filenames")
        df_copy = df.copy()
        df["image_path"] = df["image_path"].map(
            lambda x: filename_mapping.get(x.split("/")[-1], None)
        )
        if df["image_path"].isna().sum() > 0:
            print(
                f"Failed to map filenames in {df['image_path'].isna().sum()}/{len(df)} ({df['image_path'].isna().sum()/len(df):.2%}) entries"
            )
            print(
                f"Example failures: {df_copy[df['image_path'].isna()]['image_path'].head().tolist()}"
            )

    # Define model properties columns - simplified to just model name
    model_columns = ["model"]

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

    # Create derived similarity columns based on processed_answer
    # (0 == 100%, -(2**20) == 0%)
    df["object_similarities"] = df["processed_answer"].apply(
        lambda x: 0 if x == "object_wins" else -(2**20)
    )
    df["attack_similarities"] = df["processed_answer"].apply(
        lambda x: -(2**20) if x == "object_wins" else 0
    )

    metric_columns = [
        "object_similarities",
        "attack_similarities",
    ]

    # Extract image_id from image_path - use the filename as the image_id
    df["image_id"] = df["image_path"].apply(
        lambda x: Path(x).name if isinstance(x, str) else None
    )

    # Create unique image identifier based on image_id, dataset, and prompt_id
    df["unique_id"] = (
        df["image_id"]
        + "|"
        + df["dataset"].astype(str)
        + "|"
        + df["prompt_id"].astype(str)
    )
    unique_images = (
        df[["unique_id", "image_id", "prompt_id"] + image_base_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    unique_models = df["model"].unique()

    print(
        f"Processing {len(unique_images)} images across {len(unique_models)} models..."
    )

    # Build column data for each model and metric
    new_columns = {}
    for model in unique_models:
        model_data = df[df["model"] == model]
        for metric in metric_columns:
            metric_dict = dict(zip(model_data["unique_id"], model_data[metric]))
            new_columns[f"{model}_{metric}"] = unique_images["unique_id"].map(
                metric_dict
            )

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
    models_list = models_df.to_dict(orient="records")

    with open(models_json, "w") as f:
        json.dump(models_list, f, indent=2)

    print(f"Saved {len(models_list)} models to JSON files")
    return True


def generate_similarity_files(
    similarity_df: pd.DataFrame,
    metadata_json: Path,
    binary_file: Path,
    index_json: Path,
):
    """Generate binary and JSON files for similarity data."""
    # Get all columns containing 'similarities' (computed previously)
    similarity_columns = [
        col for col in similarity_df.columns if "similarities" in col
    ]

    print(
        f"Processing {len(similarity_df)} rows with {len(similarity_columns)} similarity cols"
    )

    # Create and save metadata
    metadata = {
        "columns": similarity_columns,
        "models": ["_".join(col.split("_")[:-2]) for col in similarity_columns][::2],
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
        # similarity_values = np.nan_to_num(similarity_values, nan=0.0)
        similarity_values.tofile(f)

    # Group images by their base image_id and organize variants together
    image_groups = {}

    # First pass: organize data by image_id
    for i, row in similarity_df.iterrows():
        image_id = row["image_id"] if not pd.isna(row["image_id"]) else None
        dataset = row["dataset"] if not pd.isna(row["dataset"]) else None
        prompt_id = row["prompt_id"] if not pd.isna(row["prompt_id"]) else None

        if not image_id:
            continue

        if image_id not in image_groups:
            # Initialize with common data that should be the same across variants
            image_groups[image_id] = {
                "image_id": image_id,
                "object_label": (
                    row["object_label"] if not pd.isna(row["object_label"]) else None
                ),
                "attack_word": (
                    row["attack_word"] if not pd.isna(row["attack_word"]) else None
                ),
                "variants": {},
            }

        # posit area
        postit_area_pct = (
            row["postit_area_pct"] if not pd.isna(row["postit_area_pct"]) else None
        )
        if "postit_area_pct" not in image_groups[image_id]:
            image_groups[image_id]["postit_area_pct"] = postit_area_pct
        else:
            if image_groups[image_id]["postit_area_pct"] != postit_area_pct:
                print(
                    f"Warning: postit_area_pct mismatch for {image_id} - {postit_area_pct} != {image_groups[image_id]['postit_area_pct']}"
                )

        # Store row index, prompt_id, and postit area for this variant
        variant_key = f"{dataset}_{prompt_id}" if prompt_id is not None else dataset
        image_groups[image_id]["variants"][variant_key] = {
            "row_index": i,
            "prompt_id": prompt_id,
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
    parser.add_argument(
        "filename_mapping", help="Path to the filename mapping CSV file", default=None
    )
    args = parser.parse_args()

    # Set file paths for output
    input_file = Path(args.input)
    output_dir = Path("data/")
    models_json = output_dir / "lvlm_models_properties.json"
    metadata_json = output_dir / "lvlm_similarity_metadata.json"
    binary_file = output_dir / "lvlm_similarity_data.bin"
    index_json = output_dir / "lvlm_similarity_index.json"

    start_time = time()

    print("=== SCAM Analysis Data Converter ===")

    # Step 0: Load input CSV and sort models
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found.")
        return
    print(f"Reading {input_file}...")
    input_df = pd.read_csv(input_file)

    # Adjust the sorting of models
    models = [
        "llava-llama3-1.1:8b",
        "llava-1.5:7b-CLIPA",
        "llava-1.5:7b-openai-reprod",
        "llava-1.6:7b",
        "llava-1.6:13b",
        "llava-1.6:34b",
        "gemma3:4b",
        "gemma3:12b",
        "gemma3:27b",
        "llama3.2-vision:90b",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
        "llama4:scout",
    ]

    # Print additional and missing models
    input_models = set(input_df["model"].unique())
    expected_models = set(models)
    additional_models = input_models - expected_models
    missing_models = expected_models - input_models
    if additional_models:
        print(f"Additional models in input: {sorted(additional_models)}")
    if missing_models:
        print(f"Missing models from input: {sorted(missing_models)}")

    # Sort the DataFrame by the specified model order
    input_df["model"] = pd.Categorical(
        input_df["model"], categories=models, ordered=True
    )
    input_df = input_df.sort_values("model")

    # Step 1: Process the input CSV
    print("\n--- Processing input CSV ---")
    models_df, similarity_df = process_input_df(input_df, args.filename_mapping)
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
