"""Streamlit demo: upload a photograph, get a predicted quality score, a level and
three image measurements. Uses the same preprocessing as training."""
import numpy as np
import streamlit as st
from PIL import Image

from src.predict import assess, load_assets

st.set_page_config(page_title="Image Quality Assessment", page_icon="📷")

LEVEL_COLOURS = {"Poor": "#c0392b", "Average": "#b9770e", "Good": "#1e8449"}

st.title("Image Quality Assessment")
st.write(
    "A MobileNetV2 regression model trained on KonIQ-10k predicts the mean opinion "
    "score a group of human raters would give a photograph. No reference image is needed."
)

try:
    assets = load_assets()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a photograph to score it.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
result = assess(np.asarray(image), assets=assets)

st.image(image, caption=uploaded.name, width="stretch")

left, right = st.columns(2)
left.metric("Quality score", f"{result['score_100']} / 100")
right.markdown(
    f"**Quality level**<br><span style='font-size:2rem;color:"
    f"{LEVEL_COLOURS[result['level']]}'>{result['level']}</span>",
    unsafe_allow_html=True,
)
st.progress(result["score_100"] / 100)
st.caption(f"Predicted MOS {result['mos']} on the KonIQ-10k scale.")

st.subheader("Image measurements")
st.caption(
    "These are measured directly from the image with OpenCV and are shown as extra "
    "information. The model does not use them to produce the score."
)
st.table(
    {
        "Measurement": ["Sharpness (variance of Laplacian)", "Brightness (mean intensity)",
                        "Contrast (standard deviation of intensity)"],
        "Value": [f'{result["diagnostics"][name]["value"]:.1f}'
                  for name in ("sharpness", "brightness", "contrast")],
        "Band": [result["diagnostics"][name]["band"]
                 for name in ("sharpness", "brightness", "contrast")],
    }
)
