
import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go

from utils import load_model, preprocess_image, predict
from class_names import CLASS_INFO


st.set_page_config(
    page_title="CropX — AI Crop Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading CropX model...")
def get_model():
    """Load model once and cache in memory across reruns."""
    return load_model()


def build_probability_chart(result: dict) -> go.Figure:
    """Create a horizontal bar chart of the top-10 prediction probabilities."""
    probabilities = result["probabilities"]
    top_indices = np.argsort(probabilities)[::-1][:10]

    labels = []
    values = []
    for idx in reversed(top_indices):
        idx = int(idx)
        crop, disease = CLASS_INFO[idx]
        labels.append(f"{crop} — {disease}")
        values.append(float(probabilities[idx]) * 100)

    colors = ["#16a34a" if v == max(values) else "#94a3b8" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, cornerradius=6),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title="Top-10 Prediction Probabilities",
        xaxis=dict(title="Confidence (%)", range=[0, max(values) * 1.25]),
        yaxis=dict(automargin=True),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=40, t=50, b=30),
        height=420,
        bargap=0.25,
    )
    return fig


def main():
   
    st.title("🌿 CropX")
    st.caption("AI-Powered Crop Disease Classification — Upload a leaf image for instant diagnosis")
    st.divider()

   
    try:
        model = get_model()
    except (FileNotFoundError, RuntimeError) as e:
        st.error(f"❌ Model Loading Error: {e}")
        st.stop()

   
    st.subheader("📤 Upload a Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a leaf image for disease classification",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG.",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception:
            st.error("❌ Could not read the uploaded file. Please upload a valid image.")
            st.stop()

        
Image+Button (left) | Results (right) ──
        col_left, col_right = st.columns(2)

        with col_left:
            st.image(image, caption=uploaded_file.name, width=280)
            predict_clicked = st.button(
                "🔬 Analyze Leaf", type="primary", use_container_width=True
            )

        if predict_clicked:
            with st.spinner("🧠 Analyzing leaf image..."):
                try:
                    img_array = preprocess_image(image)
                    result = predict(model, img_array)
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    st.stop()

           
            with col_right:
                st.subheader("📊 Diagnosis Results")

                with st.container(border=True):
                    st.metric(label="🌾 Crop", value=result["crop"])

                with st.container(border=True):
                    st.metric(label="🦠 Disease", value=result["disease"])

                with st.container(border=True):
                    st.metric(label="📈 Confidence", value=f"{result['confidence']:.1f}%")

                with st.container(border=True):
                    is_healthy = result["disease"].lower() == "healthy"
                    if is_healthy:
                        st.success(f"✅ Healthy — {result['crop']} leaf looks good!")
                    else:
                        st.error(f"⚠️ {result['disease']} detected in {result['crop']}")

                st.caption(f"⚡ Inference Time: {result['inference_time_ms']:.1f} ms")

         
            st.divider()
            fig = build_probability_chart(result)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Upload a leaf image to get started.")


if __name__ == "__main__":
    main()
