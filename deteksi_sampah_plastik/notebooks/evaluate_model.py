from ultralytics import YOLO
from pathlib import Path
import shutil

# Paths
MODEL_PATH = "models/best.pt"
DATA_YAML = "data/waste-detection/data.yaml"
RESULTS_DIR = Path("results/evaluation")

# Create output directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load trained model
print("Loading trained model...")
model = YOLO(MODEL_PATH)

print("Starting test-set evaluation...")

# Evaluate on TEST set
metrics = model.val(
    data=DATA_YAML,
    split="test",
    imgsz=640,
    batch=4,
    conf=0.25,
    project="results",
    name="evaluation",
    exist_ok=True,
    plots=True,
    device="cpu",
)

# Print metrics
print("\n" + "=" * 50)
print("MODEL EVALUATION RESULTS")
print("=" * 50)

print(f"Precision:    {metrics.box.mp:.4f}")
print(f"Recall:       {metrics.box.mr:.4f}")
print(f"mAP@50:       {metrics.box.map50:.4f}")
print(f"mAP@50-95:    {metrics.box.map:.4f}")

print("=" * 50)
print("Evaluation completed successfully!")
print(f"Results saved in: {RESULTS_DIR}")