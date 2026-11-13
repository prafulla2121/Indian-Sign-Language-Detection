import streamlit as st
import cv2
import csv
import numpy as np
from PIL import Image
from ultralytics import YOLO
import yaml
import os

st.set_page_config(page_title="ISL Vision - Indian Sign Language Detection", layout="centered")

MODEL_PATH = "runs/detect/train/weights/best.pt"
YAML_PATH = "data.yaml"
RESULTS_CSV = "runs/detect/train/results.csv"
SAMPLES = {
    "Hello": "samples/hello.jpg",
    "Good": "samples/good.jpg",
    "What": "samples/what.jpg",
}

def load_classes(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']

@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise

@st.cache_data
def load_metrics(csv_path):
    # Read the real numbers from the training log instead of hard-coding them.
    with open(csv_path, newline='') as f:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]
    last = rows[-1]
    peak = max(rows, key=lambda r: float(r['metrics/mAP50(B)']))
    return last, peak

def detect_sign_language_from_frame(frame, model, class_names, conf_threshold):
    results = model(frame, conf=conf_threshold, verbose=False)[0]
    detected_signs = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        sign = class_names[cls_id]
        detected_signs.append((sign, conf))

        # Draw box and label
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
        cv2.putText(frame, f"{sign} {conf:.2f}", (xyxy[0], max(xyxy[1] - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame, detected_signs

def show_detection(image, model, class_names, conf_threshold):
    try:
        # convert("RGB") so PNGs with transparency do not break the colour conversion.
        image_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        with st.spinner("Detecting sign..."):
            processed_frame, signs = detect_sign_language_from_frame(image_cv, model, class_names, conf_threshold)

        if signs:
            st.success("Detected: " + ", ".join(f"{s} ({c:.2f})" for s, c in signs))
        else:
            st.info("No sign detected above the confidence threshold. Try lowering it, or use better light.")
        # Natural size: samples are only ~350 px wide and look blurry when stretched.
        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), width="content")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

def main():
    st.title("ISL Vision")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return

    if not os.path.exists(YAML_PATH):
        st.error(f"YAML file not found at {YAML_PATH}")
        return

    try:
        class_names = load_classes(YAML_PATH)
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load resources: {str(e)}")
        return

    st.write(f"Detects {len(class_names)} common Indian Sign Language signs in a photo, using a small YOLOv8 model.")

    conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.5, 0.05)

    sample_tab, upload_tab, camera_tab = st.tabs(["Try a sample", "Upload image", "Camera snapshot"])

    with sample_tab:
        st.caption("Frames from the validation set. The coloured box already on each frame is the human label.")
        cols = st.columns(len(SAMPLES))
        for col, (name, path) in zip(cols, SAMPLES.items()):
            col.image(path, width="stretch")
            if col.button(f"Try \"{name}\"", key=f"sample_{name}", width="stretch"):
                st.session_state["sample"] = path
        if "sample" in st.session_state:
            show_detection(Image.open(st.session_state["sample"]), model, class_names, conf_threshold)

    with upload_tab:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            show_detection(Image.open(uploaded_file), model, class_names, conf_threshold)

    with camera_tab:
        # Browser camera snapshot: the old cv2.VideoCapture mode only works on the machine running the server.
        photo = st.camera_input("Hold one sign in front of the camera, then take a photo")
        if photo is not None:
            show_detection(Image.open(photo), model, class_names, conf_threshold)

    st.subheader("Signs the model knows")
    st.write(", ".join(class_names))

    st.subheader("Model card")
    last, peak = load_metrics(RESULTS_CSV)
    st.markdown(
        f"""
- **Model:** YOLOv8n (nano), fine-tuned on the public Roboflow
  [Indian Sign Language dataset](https://universe.roboflow.com/school-fzizy/indian-sign-language-fkx1f) (CC BY 4.0, 6,770 images).
- **Classes:** {len(class_names)} signs. **Training:** {last['epoch']} epochs, image size 350, CPU.
- **Validation mAP@50:** {float(last['metrics/mAP50(B)']):.3f} at the last epoch
  (peak {float(peak['metrics/mAP50(B)']):.3f} at epoch {peak['epoch']}).
  Stricter mAP@50-95: {float(last['metrics/mAP50-95(B)']):.3f}.
- **Limitations:** small dataset with a handful of signers in similar indoor rooms, so it may fail on
  new people, busy backgrounds or poor light. It reads single photos, not moving signs or sentences.
"""
    )

    with st.expander("Training curves and confusion matrix"):
        st.image("runs/detect/train/results.png", caption="Loss and metrics per epoch", width="stretch")
        st.image("runs/detect/train/confusion_matrix_normalized.png", caption="Normalised confusion matrix (validation)",
                 width="stretch")

if __name__ == "__main__":
    main()
