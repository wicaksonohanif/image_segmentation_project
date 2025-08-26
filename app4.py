import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
from io import BytesIO
import plotly.express as px
from streamlit_plotly_events import plotly_events
from segment_anything import sam_model_registry, SamPredictor

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor, device

predictor, device = load_model()

@st.cache_data
def preprocess_image(image_np):
    max_dim = 1024
    h, w = image_np.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

    predictor.set_image(image_np)
    return image_np

st.image("assets/banner_seg.png", use_column_width=True)
st.title("🖼️ Image Segmentation Application (ViT B Model)")
st.write("Upload image to segment.")

# Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    image_np = preprocess_image(image_np)

    st.subheader("Clikk on the image to get coordinates")
    fig = px.imshow(image_np)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=True, zeroline=True),
        yaxis=dict(showgrid=True, zeroline=True, scaleanchor="x")
    )
    fig.update_traces(hovertemplate='X: %{x}, Y: %{y}<extra></extra>')
    clicked = plotly_events(fig, click_event=True, select_event=False, key="click")

    if clicked:
        st.info(f"Coordinate: X={int(clicked[0]['x'])}, Y={int(clicked[0]['y'])}")

    method = st.radio("Choose prompt method:", ["Point", "Box"])
    result = None


    if method == "Point":
        st.markdown("### Insert point coordinates (use the image above for reference)")
        x = st.number_input("X coordinate", min_value=0, max_value=image_np.shape[1]-1, value=image_np.shape[1]//2)
        y = st.number_input("Y coordinate", min_value=0, max_value=image_np.shape[0]-1, value=image_np.shape[0]//2)

        if st.button("Segment with Point"):
            input_point = np.array([[x, y]])
            input_label = np.array([1])  # 1 = foreground
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            mask = masks[np.argmax(scores)]

            result = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
            result[~mask] = (255, 255, 255, 0)

            st.image(result, caption="Segmented Image", use_column_width=True)

    elif method == "Box":
        st.markdown("### Insert box coordinates (use the image above for reference)")
        xmin = st.number_input("X min", min_value=0, max_value=image_np.shape[1]-1, value=10)
        ymin = st.number_input("Y min", min_value=0, max_value=image_np.shape[0]-1, value=10)
        xmax = st.number_input("X max", min_value=0, max_value=image_np.shape[1]-1, value=image_np.shape[1]//2)
        ymax = st.number_input("Y max", min_value=0, max_value=image_np.shape[0]-1, value=image_np.shape[0]//2)

        if st.button("Segment with Box"):
            input_box = np.array([xmin, ymin, xmax, ymax])
            masks, scores, _ = predictor.predict(
                box=input_box[None, :],
                multimask_output=True,
            )
            mask = masks[np.argmax(scores)]

            result = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
            result[~mask] = (255, 255, 255, 0)

            st.image(result, caption="Segmented Image", use_column_width=True)

    if result is not None:
        result_pil = Image.fromarray(result)
        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="💾 Download Result PNG",
            data=byte_im,
            file_name="segmented.png",
            mime="image/png",
        )
