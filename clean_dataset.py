import cv2
import os
import shutil
import numpy as np

# --- Configuration ---
BLUR_THRESHOLD = 35.0    # Adjust for dataset (lower -> more blurry)
DARKNESS_THRESHOLD = 50  # Adjust for dataset (0-255 scale)

SOURCE_DIR = "dataset"               # Main dataset folder (train/val inside)
BAD_QUALITY_DIR = "dataset_bad_quality"  # Destination for bad images


def check_image_quality(image_path):
    """Checks a single image for blur and darkness."""
    image = cv2.imread(image_path)
    if image is None:
        return False, "Unreadable/Corrupt"

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Blur check
    focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Brightness check
    brightness = np.mean(gray)

    if focus_measure < BLUR_THRESHOLD:
        return False, f"Blurry (Score: {focus_measure:.2f})"

    if brightness < DARKNESS_THRESHOLD:
        return False, f"Too Dark (Brightness: {brightness:.2f})"

    return True, "Good Quality"


print("🚀 Starting dataset cleaning...")
os.makedirs(BAD_QUALITY_DIR, exist_ok=True)

# Walk through the dataset
for dirpath, dirnames, filenames in os.walk(SOURCE_DIR):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(dirpath, filename)

            is_good, reason = check_image_quality(image_path)

            if not is_good:
                print(f"⚠️ Flagged: {image_path} - Reason: {reason}")

                # Create unique filename (prefix with folder name)
                prefix = os.path.basename(dirpath)
                bad_name = f"{prefix}_{filename}"
                destination_path = os.path.join(BAD_QUALITY_DIR, bad_name)

                # Move bad image
                shutil.move(image_path, destination_path)

print("\n✅ Cleaning complete. Bad images moved to 'dataset_bad_quality'.")
