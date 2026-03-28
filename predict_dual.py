import torch
import cv2
import numpy as np
import warnings
from skimage.feature import local_binary_pattern

warnings.filterwarnings("ignore")

from models.dual_stream_model import DualStreamModel

# ----------------------------
# Settings
# ----------------------------
MODEL_PATH = "best_dual_model.pth"
IMAGE_PATH = "data/test/ripe/IMG_20241101_185111089.jpg"

classes = ["fresh", "ripe", "overripe"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Model
# ----------------------------
model = DualStreamModel(num_classes=3, dropout=0.285)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

# ----------------------------
# Load Image
# ----------------------------
rgb = cv2.imread(IMAGE_PATH)

if rgb is None:
    raise ValueError("❌ Image not found. Check path!")

rgb = cv2.resize(rgb, (640, 640))

# ----------------------------
# Convert to grayscale (IMPORTANT)
# ----------------------------
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

# ----------------------------
# EDGE (same as training)
# ----------------------------
edge = cv2.Canny(gray, 100, 200)

# ----------------------------
# LBP (same as training)
# ----------------------------
radius = 1
n_points = 8 * radius

lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
lbp = np.uint8((lbp / lbp.max()) * 255)

# ----------------------------
# Normalize
# ----------------------------
rgb = rgb / 255.0
edge = edge / 255.0
lbp = lbp / 255.0

# ----------------------------
# Convert to Tensor
# ----------------------------
rgb = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
edge = torch.tensor(edge, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
lbp = torch.tensor(lbp, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

rgb, edge, lbp = rgb.to(device), edge.to(device), lbp.to(device)

# ----------------------------
# Prediction
# ----------------------------
with torch.no_grad():
    outputs = model(rgb, edge, lbp)

    probs = torch.softmax(outputs, dim=1)
    confidence, pred = torch.max(probs, 1)

# ----------------------------
# Freshness Score
# ----------------------------
fresh_p = probs[0][0].item()
ripe_p = probs[0][1].item()
overripe_p = probs[0][2].item()

score = (
    fresh_p * 9 +
    ripe_p * 6 +
    overripe_p * 2
)

# ----------------------------
# Output
# ----------------------------
print("\n==============================")
print("🍌 FOOD FRESHNESS RESULT")
print("==============================")

print(f"Prediction  : {classes[pred.item()].upper()}")
print(f"Confidence  : {confidence.item():.2f}")

print("\nClass Probabilities:")
print(f"Fresh     : {fresh_p:.2f}")
print(f"Ripe      : {ripe_p:.2f}")
print(f"Overripe  : {overripe_p:.2f}")

print(f"\nFreshness Score (0–10): {score:.2f}")
print("==============================\n")