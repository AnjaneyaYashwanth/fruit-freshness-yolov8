import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

radius = 1
n_points = 8 * radius

base_path = "data"
splits = ["train", "val", "test"]
classes = ["fresh", "ripe", "overripe"]

def process_images():
    for split in splits:
        for cls in classes:
            input_dir = os.path.join(base_path, split, cls)

            rgb_out_dir = os.path.join(base_path, "processed", split, cls, "rgb")
            edge_out_dir = os.path.join(base_path, "processed", split, cls, "edge")
            lbp_out_dir = os.path.join(base_path, "processed", split, cls, "lbp")

            images = os.listdir(input_dir)

            for img_name in images:
                img_path = os.path.join(input_dir, img_name)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                # Resize to YOLO size
                image = cv2.resize(image, (640, 640))

                # Save RGB
                cv2.imwrite(os.path.join(rgb_out_dir, img_name), image)

                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Edge detection
                edges = cv2.Canny(gray, 100, 200)
                cv2.imwrite(os.path.join(edge_out_dir, img_name), edges)

                # LBP
                lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
                lbp = np.uint8((lbp / lbp.max()) * 255)
                cv2.imwrite(os.path.join(lbp_out_dir, img_name), lbp)

            print(f"{split} - {cls} processed.")

    print("All preprocessing completed successfully.")

if __name__ == "__main__":
    process_images()