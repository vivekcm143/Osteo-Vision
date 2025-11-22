import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import gdown


st.set_page_config(
    page_title="OSTEO VISION",
    page_icon="🦵",
    layout="wide"
)


# Model configuration
MODEL_URL = "https://drive.google.com/uc?id=1hQ-H_GhruF1_Nkalle4DpHw4N6CKxCvD"
MODEL_PATH = "models/osteo_vision_model.h5"


# Download model if not present
@st.cache_resource
def download_model():
    """Download the model from Google Drive if not already present."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        st.info("Downloading model from Google Drive... This may take a few minutes.")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            st.stop()
    return MODEL_PATH


# Load the model
@st.cache_resource
def load_model():
    """Load the trained model."""
    model_path = download_model()
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        st.stop()


model = load_model()


# Configure Grad-CAM model
@st.cache_resource
def create_grad_cam_model(_model):
    """Create Grad-CAM model for explainability."""
    try:
        grad_model = tf.keras.models.Model(
            inputs=_model.input,
            outputs=[_model.get_layer("global_average_pooling2d").output, _model.output]
        )
        return grad_model
    except Exception as e:
        st.error(f"Failed to configure Grad-CAM: {e}")
        st.stop()


grad_model = create_grad_cam_model(model)


# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])


st.title("🦵 OSTEO VISION")
st.write("**AI-Powered Knee Osteoarthritis Detection using ResNet152V2**")


# Model configuration
target_size = (224, 224)
class_names = ["KL-GRADE 0", "KL-GRADE 1", "KL-GRADE 2", "KL-GRADE 3", "KL-GRADE 4"]


def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    """Generate Grad-CAM heatmap for model explainability."""
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on the original image."""
    heatmap = np.uint8(255 * heatmap)
    jet = get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)


if uploaded_file:
    try:
        img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet_v2.preprocess_input(img_array)

        with st.spinner("🔍 Analyzing the image..."):
            predictions = model.predict(img_array)[0]
            predicted_class = class_names[np.argmax(predictions)]
            prediction_probabilities = 100 * predictions

        st.subheader("✅ Prediction Result")
        st.metric(
            label="Predicted Severity Level",
            value=predicted_class,
            delta=f"Confidence: {np.max(prediction_probabilities):.2f}%"
        )

        heatmap = make_gradcam_heatmap(grad_model, img_array)
        heatmap_overlay = overlay_heatmap_on_image(
            tf.keras.preprocessing.image.img_to_array(img),
            heatmap
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📸 Input Image")
            st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)
        with col2:
            st.subheader("🔥 Explainability with Grad-CAM")
            st.image(heatmap_overlay, caption="Grad-CAM Heatmap", use_column_width=True)

        st.subheader("📊 Prediction Confidence Levels")
        fig, ax = plt.subplots(figsize=(8, 4))

        bars = ax.barh(class_names, prediction_probabilities, color='skyblue')
        ax.set_xlim([0, 100])
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Prediction Confidence by KL Grade")

        for bar, prob in zip(bars, prediction_probabilities):
            ax.text(
                prob + 2,
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.2f}%",
                va='center',
                ha='left',
                fontsize=10
            )

        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error processing the image: {e}")

else:
    st.info("👈 Please upload a knee X-ray image from the sidebar to begin analysis.")
    
    # Add information section
    with st.expander("ℹ️ About Kellgren-Lawrence Grading System"):
        st.write("""
        **KL-GRADE 0**: Normal - No radiographic features of osteoarthritis
        
        **KL-GRADE 1**: Doubtful - Minute osteophyte, doubtful significance
        
        **KL-GRADE 2**: Minimal - Definite osteophyte, unimpaired joint space
        
        **KL-GRADE 3**: Moderate - Moderate diminution of joint space
        
        **KL-GRADE 4**: Severe - Joint space greatly impaired with sclerosis of subchondral bone
        """)
