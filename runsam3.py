import os
import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor
from scipy.ndimage import label, binary_fill_holes, gaussian_filter, distance_transform_edt

# ==========================================
# CONFIGURATION
# ==========================================
IMAGE_FILENAME = "DSCF5720.JPG"

# Tailored prompts capturing clothing, details, etc.
PROMPTS = [
    "placeholder prompt"
]

THRESHOLD = 0.22 # Balanced sensitivity - default 0.22
DILATION_SIZE = 0  # Edge expansion - default 0 (was 5 for early images)
SIGMA = 10  # Gaussian smoothing - default 10

# Maximum pixel area for holes to fill.
MAX_HOLE_SIZE = 5000000  # default is 5000, covers relatively small gaps. 500 for absolute minimal

# Maximum number of separate blobs/components to keep (sorted by size)
# Set to 1 for a single primary subject, 2+ to keep multiple disconnected people/objects
MAX_COMPONENTS_TO_KEEP = 1 # default is 1

# --- SELECTIVE EDGE FILLING SIDES ---
# Specify which sides of the crop window should act as walls to seal edge gaps.
# Options: Choose any combination from "top", "bottom", "left", "right" (or leave empty [] to disable)
EDGE_FILL_SIDES = []

# --- MULTI-PERSON SIDE RESTRICTION (PRE-CROP) ---
# Options: "left", "right", or None (process whole image)
SIDE_RESTRICTION = None

# If None, it automatically splits right down the middle (W // 2).
# Otherwise, you can specify an exact X pixel coordinate.
CUSTOM_SPLIT_X = None
# ==========================================

# Ensure input and output directories exist
INPUT_DIR = "input"
OUTPUT_DIR = "output"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_PATH = os.path.join(INPUT_DIR, IMAGE_FILENAME)
base_name, _ = os.path.splitext(IMAGE_FILENAME)

# Enable RTX 4090 Tensor Core accelerations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("Loading SAM 3 model via transformers...")
model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16, device_map="cuda")
processor = Sam3Processor.from_pretrained("facebook/sam3")

print(f"Loading image from: {IMAGE_PATH}")
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Could not find image at '{IMAGE_PATH}'. Please ensure it is placed inside the '{INPUT_DIR}' directory.")

full = Image.open(IMAGE_PATH).convert("RGB")
full = ImageOps.exif_transpose(full)  # Automatically fix camera rotation
W, H = full.size
print(f"Original Image dimensions: {W}x{H}")

# --- 1. PRE-CROP THE INPUT IMAGE BASED ON SIDE RESTRICTION ---
split_x = CUSTOM_SPLIT_X if CUSTOM_SPLIT_X is not None else W // 2

if SIDE_RESTRICTION == "left":
    print(f"Pre-cropping to LEFT side (X: 0 to {split_x})...")
    working_img = full.crop((0, 0, split_x, H))
    offset_x = 0
elif SIDE_RESTRICTION == "right":
    print(f"Pre-cropping to RIGHT side (X: {split_x} to {W})...")
    working_img = full.crop((split_x, 0, W, H))
    offset_x = split_x
else:
    print("Processing full image (no side restriction)...")
    working_img = full
    offset_x = 0

W_sub, H_sub = working_img.size


def get_mask(prompt, thresh):
    inputs = processor(images=working_img, text=prompt, return_tensors="pt")
    inputs = {k: v.to("cuda", dtype=torch.bfloat16) if v.dtype == torch.float32 else v.to("cuda") for k, v in
              inputs.items()}

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(**inputs)

    res = processor.post_process_instance_segmentation(outputs, threshold=thresh, target_sizes=[(H_sub, W_sub)])[0]

    merged_float = np.zeros((H_sub, W_sub), dtype=np.float32)
    if "masks" in res and len(res["masks"]) > 0:
        for m in res["masks"]:
            merged_float = np.maximum(merged_float, m.cpu().numpy().astype(np.float32))

    return merged_float


print("Running multi-prompt segmentation on cropped region...")
mask_accumulated = np.zeros((H_sub, W_sub), dtype=np.float32)
for p in PROMPTS:
    m = get_mask(p, THRESHOLD)
    mask_accumulated = np.maximum(mask_accumulated, m)

# ================= SMOOTHING ===============
print("Smoothing probability map to eliminate high-resolution stair-steps...")
mask_accumulated = gaussian_filter(mask_accumulated, sigma=SIGMA)

# Automatic background cleanup: keep only the largest connected component
# binary_mask = (mask_accumulated > 0.2).astype(np.int32)
# labeled_mask, num_features = label(binary_mask)
#
# if num_features > 1:
#     print(f"Detected {num_features} segments. Filtering out background clutter...")
#     component_sizes = np.bincount(labeled_mask.ravel())
#     component_sizes[0] = 0
#     largest_component_label = component_sizes.argmax()
#     binary_mask = (labeled_mask == largest_component_label).astype(np.uint8)
# else:
#     binary_mask = (binary_mask > 0).astype(np.uint8)

# ================= BACKGROUND CLEANUP & COMPONENT FILTERING =================
print(f"Filtering background clutter (keeping top {MAX_COMPONENTS_TO_KEEP} largest components)...")
binary_mask = (mask_accumulated > 0.2).astype(np.int32)
labeled_mask, num_features = label(binary_mask)

if num_features > 0:
    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0  # Ignore background index 0

    # Sort component indices by size in descending order (largest first)
    sorted_indices = np.argsort(component_sizes)[::-1]

    # Take up to MAX_COMPONENTS_TO_KEEP valid components
    top_indices = sorted_indices[:MAX_COMPONENTS_TO_KEEP]
    valid_labels = top_indices[component_sizes[top_indices] > 0]  # Filter out empty slots if fewer than N exist

    print(f"Detected {num_features} total segments. Keeping the top {len(valid_labels)} largest component(s).")
    binary_mask = np.isin(labeled_mask, valid_labels).astype(np.uint8)
else:
    binary_mask = np.zeros_like(binary_mask, dtype=np.uint8)

# =============== SELECTIVE EDGE HOLE FILLING ====================
print(f"Filling internal holes (under {MAX_HOLE_SIZE} pixels) [Target Edges: {EDGE_FILL_SIDES}]...")

pad_top = 1 if "top" in EDGE_FILL_SIDES else 0
pad_bottom = 1 if "bottom" in EDGE_FILL_SIDES else 0
pad_left = 1 if "left" in EDGE_FILL_SIDES else 0
pad_right = 1 if "right" in EDGE_FILL_SIDES else 0

if any([pad_top, pad_bottom, pad_left, pad_right]):
    pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
    padded_mask = np.pad(binary_mask, pad_width=pad_width, mode='constant', constant_values=1)
    fully_filled_padded = binary_fill_holes(padded_mask)

    t_end = -pad_bottom if pad_bottom > 0 else None
    l_end = -pad_right if pad_right > 0 else None
    fully_filled = fully_filled_padded[pad_top:t_end, pad_left:l_end]
else:
    fully_filled = binary_fill_holes(binary_mask)

all_holes = fully_filled & (~binary_mask.astype(bool))
labeled_holes, num_holes = label(all_holes)

if num_holes > 0:
    hole_sizes = np.bincount(labeled_holes.ravel())
    hole_sizes[0] = 0  # Ignore background

    small_holes_mask = np.isin(labeled_holes, np.where((hole_sizes > 0) & (hole_sizes < MAX_HOLE_SIZE))[0])
    binary_mask = np.logical_or(binary_mask, small_holes_mask).astype(np.uint8)

# --- DILATION ---
if DILATION_SIZE > 0:
    print(f"Applying smooth distance-transform dilation (radius={DILATION_SIZE})...")
    dist_to_object = distance_transform_edt(1 - binary_mask)
    dilated_mask = np.logical_or(binary_mask, dist_to_object <= DILATION_SIZE).astype(np.uint8)
else:
    dilated_mask = binary_mask

# --- 2. MAP SUB-MASK BACK ONTO FULL-SIZE (W, H) CANVAS ---
full_mask_canvas = np.zeros((H, W), dtype=np.uint8)
full_mask_canvas[:, offset_x:offset_x + W_sub] = dilated_mask

# Save grayscale mask into output folder
mask_img = Image.fromarray((full_mask_canvas * 255).astype(np.uint8))
mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask.png")
mask_img.save(mask_path)

# Generate final color overlay image into output folder
print("Generating color overlay...")
base_img = np.array(full).astype(np.float32)
mask_float = np.array(mask_img).astype(np.float32) / 255.0
mask_3ch = np.stack([mask_float] * 3, axis=-1)

colored_tint = np.array([255, 60, 90], dtype=np.float32)  # Vibrant pinkish-red
blended = base_img * 0.4 + colored_tint * 0.6
final_overlay = base_img * (1 - mask_3ch) + blended * mask_3ch

overlay_path = os.path.join(OUTPUT_DIR, f"{base_name}_overlay.jpg")
Image.fromarray(np.clip(final_overlay, 0, 255).astype(np.uint8)).save(overlay_path, quality=95)

print(f"Successfully processed! Results saved to '{OUTPUT_DIR}/':")
print(f" - Mask: {mask_path}")
print(f" - Overlay: {overlay_path}")