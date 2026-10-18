import gradio as gr
from ultralytics import YOLO
from pathlib import Path


# ---------------------------------------
# Paths
# ---------------------------------------

MODEL_PATH = Path("models/best.pt")


# ---------------------------------------
# Load trained model
# ---------------------------------------

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")


# ---------------------------------------
# Detection function
# ---------------------------------------

def detect_waste(image):

    if image is None:
        return None

    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    # Image with bounding boxes
    annotated_image = result.plot()

    return annotated_image


# ---------------------------------------
# Gradio Interface
# ---------------------------------------

with gr.Blocks(title="Waste Detection YOLO") as app:

    gr.Markdown(
        """
        # ♻️ Waste Detection System

        Upload an image and the trained YOLO model will detect
        different types of waste objects.
        """
    )

    with gr.Row():

        input_image = gr.Image(
            type="numpy",
            label="Upload Waste Image"
        )

        output_image = gr.Image(
            label="Detection Result"
        )

    detect_button = gr.Button("🔍 Detect Waste")

    detect_button.click(
        fn=detect_waste,
        inputs=input_image,
        outputs=output_image
    )

    gr.Markdown(
        """
        ### Detected Classes

        - Garbage Bag
        - Glass
        - Paper Bag
        - Pet Bottle
        - Plastic Bag
        - Can
        """
    )


# ---------------------------------------
# Launch application
# ---------------------------------------

if __name__ == "__main__":
    app.launch()