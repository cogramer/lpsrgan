import cv2
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# Folder containing the GT and SR images
folder_path = Path(r"D:/ThucTapDienToanGroup/for_comparison/plate-100000")

gt_filename = "hr-001.png"
output_filename = "results.txt"


def pad_to_ratio(image, target_height_to_width=(2, 3), color=(127, 127, 127)):
    """
    Pad an image symmetrically to the target height-to-width ratio.

    OpenCV uses BGR color order. Since all three values are 127,
    this produces a gray padding color.
    """
    height, width = image.shape[:2]

    target_h, target_w = target_height_to_width
    target_ratio = target_h / target_w
    current_ratio = height / width

    if current_ratio < target_ratio:
        # Image is too wide, so add padding to the top and bottom
        new_height = round(width * target_ratio)
        padding_total = new_height - height

        pad_top = padding_total // 2
        pad_bottom = padding_total - pad_top
        pad_left = 0
        pad_right = 0

    elif current_ratio > target_ratio:
        # Image is too tall, so add padding to the left and right
        new_width = round(height / target_ratio)
        padding_total = new_width - width

        pad_left = padding_total // 2
        pad_right = padding_total - pad_left
        pad_top = 0
        pad_bottom = 0

    else:
        return image

    return cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )


# Locate the GT image
gt_path = folder_path / gt_filename

if not gt_path.exists():
    raise FileNotFoundError(f"Could not find GT image: {gt_path}")

gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)

if gt is None:
    raise FileNotFoundError(f"Could not load GT image: {gt_path}")

# Pad GT once to a 2:3 height-to-width ratio
gt_padded = pad_to_ratio(
    gt,
    target_height_to_width=(2, 3),
    color=(127, 127, 127)
)

print("Original GT shape:", gt.shape)
print("Padded GT shape:", gt_padded.shape)


# Supported image extensions
image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Find all image files except the GT image
sr_paths = sorted(
    path for path in folder_path.iterdir()
    if (
        path.is_file()
        and path.suffix.lower() in image_extensions
        and path.name != gt_filename
    )
)

if not sr_paths:
    raise FileNotFoundError(
        f"No SR images found in folder: {folder_path}"
    )


results = []

for sr_path in sr_paths:
    sr = cv2.imread(str(sr_path), cv2.IMREAD_COLOR)

    if sr is None:
        print(f"Skipping unreadable image: {sr_path.name}")
        continue

    # Resize the padded GT to the SR image dimensions
    gt_resized = cv2.resize(
        gt_padded,
        (sr.shape[1], sr.shape[0]),  # width, height
        interpolation=cv2.INTER_AREA
    )

    # Calculate PSNR
    psnr = peak_signal_noise_ratio(
        gt_resized,
        sr,
        data_range=255
    )

    # Calculate SSIM
    ssim = structural_similarity(
        gt_resized,
        sr,
        data_range=255,
        channel_axis=2
    )

    result = (
        f"name: {sr_path.name}\n"
        f"PSNR: {psnr:.6f}\n"
        f"SSIM: {ssim:.6f}\n"
    )

    results.append(result)

    print(f"Processed: {sr_path.name}")
    print(f"PSNR: {psnr:.6f}")
    print(f"SSIM: {ssim:.6f}")


# Write all results to the final text file
output_path = folder_path / output_filename

with open(output_path, "w", encoding="utf-8") as file:
    file.write("\n".join(results))

print(f"\nResults written to: {output_path}")
