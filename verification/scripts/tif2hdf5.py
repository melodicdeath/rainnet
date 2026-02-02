import argparse
import sys
import os
import yaml
import numpy as np
import pandas as pd
import h5py
import cv2
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

# Add project root to path so we can import things if needed
sys.path.append(os.getcwd())

from pysteps.io.importers import import_fmi_geotiff


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def read_geotiff(filename):
    # Mimic FMIComposite.py read_geotiff
    try:
        data, _, _ = import_fmi_geotiff(filename)
        data = np.nan_to_num(data, nan=-32.0)
        return data
    except Exception as e:
        logging.warning(f"Failed to read {filename}: {e}")
        return None


def process_file(im, bbox, input_image_size):
    # 1. Crop
    if bbox is not None:
        # FMIComposite.py logic: im[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        # Config bbox: [x1, x2, y1, y2] used as slice(bbox[0], bbox[1]) for dim0
        im = im[bbox[0] : bbox[1], bbox[2] : bbox[3]]

    # 2. Resize
    # FMIComposite logic: Block reduce or resize.
    # We use cv2.resize as per optimized logic.
    # cv2.resize(src, dsize=(width, height)) -> dsize is (cols, rows) -> (dim1, dim0)
    # input_image_size is [dim0, dim1] usually?
    # FMIComposite line 215: block_x = im.shape[0] / input_image_size[0]
    # So input_image_size is [target_dim0, target_dim1] i.e. [Height, Width]

    target_h, target_w = input_image_size[0], input_image_size[1]

    if im.shape[0] != target_h or im.shape[1] != target_w:
        # cv2.resize takes (width, height)
        im = cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_AREA)

    return im


def main():
    parser = argparse.ArgumentParser(description="Convert RainNet data to HDF5")
    parser.add_argument("config", help="Path to config yaml file")
    parser.add_argument(
        "--output", help="Output HDF5 filename", default="rainnet_data.hdf5"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers (not used yet)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Extract Datelists
    date_list_fmt = cfg["date_list"]
    unique_dates = set()

    for split in ["train", "valid", "test"]:
        dl_path = date_list_fmt.format(split=split)
        if os.path.exists(dl_path):
            print(f"Reading datelist: {dl_path}")
            df = pd.read_csv(dl_path, header=None, parse_dates=[0])
            dates = df.iloc[:, 0].tolist()
            unique_dates.update(dates)
        else:
            print(f"Warning: Datelist {dl_path} not found.")

    dates = sorted(list(unique_dates))
    print(f"Total unique dates to process: {len(dates)}")

    # Setup HDF5
    output_path = args.output
    print(f"Creating HDF5: {output_path}")

    # Params
    bbox = cfg.get("bbox", None)
    input_image_size = cfg.get("input_image_size", [64, 64])

    path_fmt = cfg["path"]
    filename_fmt = cfg["filename"]

    with h5py.File(output_path, "w") as f:
        # We can store some metadata if we want, but FMIComposite just reads by key

        for date in tqdm(dates):
            # Construct filename
            # Note: config file parsing of {year} etc is usually done by .format()
            # But the logic in FMIComposite uses path / filename with format args

            # FMIComposite line 162
            file_dir = path_fmt.format(
                year=date.year,
                month=date.month,
                day=date.day,
                hour=date.hour,
                minute=date.minute,
                second=date.second,
            )

            file_name = filename_fmt.format(
                year=date.year,
                month=date.month,
                day=date.day,
                hour=date.hour,
                minute=date.minute,
                second=date.second,
            )

            full_path = Path(file_dir) / Path(file_name)

            if not full_path.exists():
                # Try simple path if formatting failed or was absolute
                # Sometimes path is just "data" and filename is complex
                pass

            ts_key = date.strftime("%Y-%m-%d %H:%M:%S")

            if ts_key in f:
                continue

            im = read_geotiff(str(full_path))

            if im is None:
                continue

            im_processed = process_file(im, bbox, input_image_size)

            # Save to HDF5
            # Use float16 to save space if appropriate, but standard is float32 or keep original?
            # FMIComposite uses float, so float32 is safe.
            f.create_dataset(
                ts_key, data=im_processed.astype(np.float32), compression="gzip"
            )

    print("Done! verification:")
    with h5py.File(output_path, "r") as f:
        print(f"Keys in HDF5: {len(f.keys())}")
        if len(f.keys()) > 0:
            k = list(f.keys())[0]
            print(f"Sample shape ({k}): {f[k].shape}")


if __name__ == "__main__":
    main()
