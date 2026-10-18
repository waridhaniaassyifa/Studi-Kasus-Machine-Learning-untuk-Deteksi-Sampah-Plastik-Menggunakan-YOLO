
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
IMAGE_PATH = BASE_DIR / "gambar.jpg"
OUTPUT_DIR = BASE_DIR / "hasil"
OUTPUT_DIR.mkdir(exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Bobot model tidak ditemukan: {MODEL_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Gambar tidak ditemukan: {IMAGE_PATH}"
    )

print("Memuat model deteksi sampah...")
model = YOLO(str(MODEL_PATH))

print("Mendeteksi sampah plastik...")
results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    imgsz=640,
    save=False,
    verbose=False
)

result = results[0]
output_path = OUTPUT_DIR / "hasil_deteksi.jpg"
result.save(filename=str(output_path))

print("\n=== HASIL DETEKSI ===")

if result.boxes is None or len(result.boxes) == 0:
    print("Belum ada objek yang terdeteksi.")
else:
    for i, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls.item())
        label = model.names[class_id]
        confidence = float(box.conf.item())

        print(
            f"{i}. {label} "
            f"| Confidence: {confidence:.2%}"
        )

    print(f"Total objek: {len(result.boxes)}")

print(f"Hasil gambar: {output_path}")
