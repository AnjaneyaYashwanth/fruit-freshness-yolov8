import os
import shutil
import random

random.seed(42)

# Define paths
base_path = "data"
raw_path = os.path.join(base_path, "raw")

train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "val")
test_path = os.path.join(base_path, "test")

# Class mapping
class_mapping = {
    "fresh": ["Day1_GreenishYellow", "Day2_Yellow"],
    "ripe": ["Day3_Ripe"],
    "overripe": ["Day4_Overripe", "Day5_Overripe"]
}

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

def create_dirs():
    for split in ["train", "val", "test"]:
        for cls in class_mapping.keys():
            os.makedirs(os.path.join(base_path, split, cls), exist_ok=True)

def split_images():
    for cls, folders in class_mapping.items():
        all_images = []

        for folder in folders:
            folder_path = os.path.join(raw_path, folder)
            images = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]
            all_images.extend(images)

        random.shuffle(all_images)

        total = len(all_images)
        train_end = int(train_ratio * total)
        val_end = int((train_ratio + val_ratio) * total)

        splits = {
            "train": all_images[:train_end],
            "val": all_images[train_end:val_end],
            "test": all_images[val_end:]
        }

        for split_name, images in splits.items():
            for img_path in images:
                dest = os.path.join(base_path, split_name, cls, os.path.basename(img_path))
                shutil.copy(img_path, dest)

        print(f"{cls}: {total} images split successfully")

if __name__ == "__main__":
    create_dirs()
    split_images()
    print("Dataset splitting completed.")