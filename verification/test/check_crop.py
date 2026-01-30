import os
import skimage.io
import numpy as np
import matplotlib.pyplot as plt

# File path
file_path = "data/202201010010_FIN-DBZ-3067-250M.tif"

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

# Read the image
print(f"Reading image from {file_path}...")
try:
    im = skimage.io.imread(file_path)
    print(f"Original image shape: {im.shape}")
    print(f"Original image dtype: {im.dtype}")
    print(f"Original image min/max: {np.nanmin(im)} / {np.nanmax(im)}")
except Exception as e:
    print(f"Failed to read image: {e}")
    exit(1)

# User's Bounding Box (provided)
bbox = [545, 4537, 1345, 6545]

# User Logic: slice(bbox[0], bbox[1]) for dim0
bbox_x_slice = slice(bbox[0], bbox[1])
# User Logic: slice(bbox[2], bbox[3]) for dim1
bbox_y_slice = slice(bbox[2], bbox[3])

print("\n--- Analysis of User Logic ---")
print(f"Bbox: {bbox}")
print("User logic: im[bbox[0]:bbox[1], bbox[2]:bbox[3]]")
print(f"Slice 1 (839:4471) applied to Dim0")
print(f"Slice 2 (1314:6403) applied to Dim1")

# Apply user logic (handling potential truncation)
try:
    if im.ndim == 2:
        im_user = im[bbox_x_slice, bbox_y_slice]
    else:
        im_user = im[bbox_x_slice, bbox_y_slice, ...]

    print(f"User Logic Output Shape: {im_user.shape}")

    # Save user result
    # Normalize for visualization
    im_vis_user = im_user.astype(float)
    im_vis_user = np.nan_to_num(im_vis_user, nan=0)
    if np.ptp(im_vis_user) > 0:
        im_vis_user = (
            (im_vis_user - im_vis_user.min())
            / (im_vis_user.max() - im_vis_user.min())
            * 255
        ).astype(np.uint8)
    else:
        im_vis_user = im_vis_user.astype(np.uint8)

    skimage.io.imsave("crop_user_logic.png", im_vis_user)
    print("Saved 'crop_user_logic.png'")

except Exception as e:
    print(f"User logic failed: {e}")

# Swapped Logic (Correcting x/y order)
# Assuming bbox corresponds to [xmin, xmax, ymin, ymax]
# And im corresponds to [y, x] (rows, cols)
# Then correct usage is im[y_slice, x_slice]
# y_slice is 1314:6403 (bbox[2]:bbox[3])
# x_slice is 839:4471 (bbox[0]:bbox[1])

print("\n--- Analysis of Corrected/Swapped Logic ---")
print("Likely intent: swap x and y dimensions (im is usually [y,x])")
print(f"Applying y_slice {bbox[2]}:{bbox[3]} to Dim0 (Rows, max {im.shape[0]})")
print(f"Applying x_slice {bbox[0]}:{bbox[1]} to Dim1 (Cols, max {im.shape[1]})")

# Corrected slices
corrected_y_slice = slice(bbox[2], bbox[3])
corrected_x_slice = slice(bbox[0], bbox[1])

# Check bounds fit
fits = bbox[3] <= im.shape[0] and bbox[1] <= im.shape[1]
if fits:
    print("Coordinates fit within image bounds perfectly.")
else:
    print("WARNING: Coordinates may still be out of bounds even after swapping!")

try:
    if im.ndim == 2:
        im_corrected = im[corrected_y_slice, corrected_x_slice]
    else:
        im_corrected = im[corrected_y_slice, corrected_x_slice, ...]

    print(f"Corrected Logic Output Shape: {im_corrected.shape}")

    # Save corrected result
    im_vis_corr = im_corrected.astype(float)
    im_vis_corr = np.nan_to_num(im_vis_corr, nan=0)
    if np.ptp(im_vis_corr) > 0:
        im_vis_corr = (
            (im_vis_corr - im_vis_corr.min())
            / (im_vis_corr.max() - im_vis_corr.min())
            * 255
        ).astype(np.uint8)
    else:
        im_vis_corr = im_vis_corr.astype(np.uint8)

    skimage.io.imsave("crop_corrected_logic.png", im_vis_corr)
    print("Saved 'crop_corrected_logic.png'")

except Exception as e:
    print(f"Corrected logic failed: {e}")

print("\nCheck the generated PNG files to confirm the area.")
