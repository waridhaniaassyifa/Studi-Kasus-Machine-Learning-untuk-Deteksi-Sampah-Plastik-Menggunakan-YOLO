from ultralytics import YOLO
from pathlib import Path


# =============================
# PROJECT PATH
# =============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
SOURCE_PATH = PROJECT_ROOT / "data" / "waste-detection" / "test" / "images"

OUTPUT_DIR = PROJECT_ROOT / "results" / "predictions"
OUTPUT_NAME = "test_predictions"


# =============================
# LOAD MODEL
# =============================

print("Loading trained YOLO model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully!")
print("Running predictions...")


# =============================
# RUN PREDICTION
# =============================

results = model.predict(
    source=str(SOURCE_PATH),
    conf=0.25,
    save=True,
    project=str(OUTPUT_DIR.resolve()),
    name=OUTPUT_NAME,
    exist_ok=True
)


# =============================
# SUMMARY
# =============================

FINAL_OUTPUT = OUTPUT_DIR / OUTPUT_NAME

print("\n" + "=" * 50)
print("PREDICTION COMPLETED")
print("=" * 50)

print(f"Images processed: {len(results)}")
print(f"Predictions saved to: {FINAL_OUTPUT}")