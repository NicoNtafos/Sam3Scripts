import os
import numpy as np
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_DIR = "input"
OUTPUT_DIR = "output"

# 1. The original source photograph (lives in 'input')
ORIGINAL_IMAGE_FILENAME = "DSCF5720.JPG"

# 2. Your hand-edited overlay file (lives in 'output')
EDITED_OVERLAY_FILENAME = "DSCF5720_overlay_manual.jpg"

# 3. Suffix for the new output mask file (saved in 'output') - appended to original file name
OUTPUT_SUFFIX = "_mask_manual.png"
# ==========================================

base_name, _ = os.path.splitext(ORIGINAL_IMAGE_FILENAME)

# Build full paths based on directories
original_path = os.path.join(INPUT_DIR, ORIGINAL_IMAGE_FILENAME)
overlay_path = os.path.join(OUTPUT_DIR, EDITED_OVERLAY_FILENAME)

output_mask_filename = f"{base_name}{OUTPUT_SUFFIX}"
mask_path = os.path.join(OUTPUT_DIR, output_mask_filename)

print(f"Loading original image from '{original_path}'...")
base_img = np.array(Image.open(original_path).convert("RGB")).astype(np.float32)

print(f"Loading edited overlay from '{overlay_path}'...")
edited_overlay = np.array(Image.open(overlay_path).convert("RGB")).astype(np.float32)

# Find pixels that differ from the original background
# (meaning you painted over them to add or remove parts of the mask)
print("Extracting mask modifications...")
diff_from_base = np.abs(edited_overlay - base_img).mean(axis=-1)
new_mask = (diff_from_base > 12.0).astype(np.uint8)

# Save the new binary mask with the touch-up suffix to the output folder
mask_img = Image.fromarray((new_mask * 255).astype(np.uint8))
mask_img.save(mask_path)

print(f"Success! New touch-up mask written to '{mask_path}'.")