import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Sports Player Detection", page_icon="🏟️")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

st.title("🏟️ Sports Player Detection")
st.write("Upload a sports image. A pre-trained YOLO11 model finds the players and the ball.")

conf = st.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
file = st.file_uploader("Upload a sports image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    r = model(img, classes=[0, 32], conf=conf, verbose=False)[0]
    st.image(r.plot()[:, :, ::-1], caption="Detections", use_container_width=True)
    players = sum(int(b.cls[0]) == 0 for b in r.boxes)
    balls = sum(int(b.cls[0]) == 32 for b in r.boxes)
    c1, c2 = st.columns(2)
    c1.metric("Players detected", players)
    c2.metric("Balls detected", balls)

if os.path.exists("output.mp4"):
    st.subheader("Demo: video tracking")
    st.video("output.mp4")

st.caption("Model: Ultralytics YOLO11n, pre-trained on COCO.")