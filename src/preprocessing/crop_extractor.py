import os
import cv2
import pandas as pd
from tqdm import tqdm


MIN_WIDTH = 32
MIN_HEIGHT = 64
MIN_VISIBILITY = 0.3


def clip_bbox(x, y, w, h, img_width, img_height):

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(img_width, x + w)
    y2 = min(img_height, y + h)

    new_w = x2 - x1
    new_h = y2 - y1

    return x1, y1, new_w, new_h


def is_valid_bbox(w, h, visibility):

    if w < MIN_WIDTH:
        return False

    if h < MIN_HEIGHT:
        return False

    if visibility < MIN_VISIBILITY:
        return False

    return True


def extract_crop(image, x, y, w, h):

    crop = image[y:y+h, x:x+w]

    return crop


def process_annotations(metadata_path, output_root):

    df = pd.read_parquet(metadata_path)

    print(f"\nLoaded annotations: {len(df)}")

    saved_count = 0
    skipped_count = 0

    grouped = df.groupby(["sequence_id", "id"])

    for (sequence_id, object_id), group_df in tqdm(grouped):

        save_dir = os.path.join(
            output_root,
            sequence_id,
            f"id_{object_id:04d}"
        )

        os.makedirs(save_dir, exist_ok=True)

        group_df = group_df.sort_values("frame")

        for _, row in group_df.iterrows():

            image_path = row["image_path"]

            frame_id = int(row["frame"])

            x = int(row["x"])
            y = int(row["y"])
            w = int(row["w"])
            h = int(row["h"])

            visibility = float(row["visibility"])

            if not os.path.exists(image_path):
                skipped_count += 1
                continue

            image = cv2.imread(image_path)

            if image is None:
                skipped_count += 1
                continue

            img_height, img_width = image.shape[:2]

            x, y, w, h = clip_bbox(
                x,
                y,
                w,
                h,
                img_width,
                img_height
            )

            if not is_valid_bbox(w, h, visibility):
                skipped_count += 1
                continue

            crop = extract_crop(
                image,
                x,
                y,
                w,
                h
            )

            if crop.size == 0:
                skipped_count += 1
                continue

            crop_filename = f"frame_{frame_id:06d}.jpg"

            crop_save_path = os.path.join(
                save_dir,
                crop_filename
            )

            cv2.imwrite(crop_save_path, crop)

            saved_count += 1

    print("\nCrop extraction complete.")
    print(f"Saved crops: {saved_count}")
    print(f"Skipped crops: {skipped_count}")


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    metadata_path = os.path.join(
        BASE_DIR,
        "..",
        "..",
        "data",
        "metadata",
        "annotations.parquet"
    )

    output_root = os.path.join(
        BASE_DIR,
        "..",
        "..",
        "data",
        "crops"
    )

    os.makedirs(output_root, exist_ok=True)

    process_annotations(
        metadata_path,
        output_root
    )