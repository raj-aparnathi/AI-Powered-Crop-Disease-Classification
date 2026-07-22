"""
CropX - Utility Functions
===========================
Core functions for model loading, image preprocessing, and prediction.
Uses the exact same preprocessing pipeline as MobileNetV3 training.
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

from class_names import CLASS_NAMES, CLASS_INFO, NUM_CLASSES

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
MODEL_PATH = Path(__file__).parent / "model" / "CropX_MobileNetV3_Best.keras"


# ──────────────────────────────────────────────────────────────
# Model Loading (cached at module level)
# ──────────────────────────────────────────────────────────────
def load_model() -> tf.keras.Model:
    """
    Load the trained CropX MobileNetV3 Keras model.

    Returns:
        Compiled Keras model ready for inference.

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If model loading fails for any reason.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Please ensure the model is placed in the 'model/' directory."
        )
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e


# ──────────────────────────────────────────────────────────────
# Image Preprocessing
# ──────────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess an uploaded image exactly as done during MobileNetV3 training.

    Pipeline:
        1. Convert to RGB (handle RGBA/grayscale)
        2. Resize to 224x224
        3. Convert to float32 numpy array
        4. Apply MobileNetV3 preprocess_input (scales to [-1, 1])
        5. Add batch dimension

    Args:
        image: PIL Image object from the uploaded file.

    Returns:
        Preprocessed numpy array of shape (1, 224, 224, 3).
    """
    # Ensure RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to training dimensions
    image = image.resize(IMG_SIZE, Image.LANCZOS)

    # Convert to float32 array
    img_array = np.array(image, dtype=np.float32)

    # Apply official MobileNetV3 preprocessing (scales pixels to [-1, 1])
    img_array = preprocess_input(img_array)

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ──────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────
def predict(model: tf.keras.Model, img_array: np.ndarray) -> dict:
    """
    Run inference on a preprocessed image and return structured results.

    Args:
        model: Loaded Keras model.
        img_array: Preprocessed image array of shape (1, 224, 224, 3).

    Returns:
        Dictionary containing:
            - predicted_class: Raw class label string.
            - crop: Cleaned crop name.
            - disease: Cleaned disease name.
            - confidence: Top-1 confidence as a percentage.
            - probabilities: Full probability array (56 classes).
            - top3: List of top-3 predictions, each a dict with
                    {class_name, crop, disease, confidence}.
            - inference_time_ms: Inference time in milliseconds.
    """
    # Timed inference
    start_time = time.perf_counter()
    predictions = model.predict(img_array, verbose=0)
    end_time = time.perf_counter()

    inference_time_ms = (end_time - start_time) * 1000
    probabilities = predictions[0]

    # Top-1 prediction
    top_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[top_idx]) * 100
    predicted_class = CLASS_NAMES[top_idx]
    crop, disease = CLASS_INFO[top_idx]

    # Top-3 predictions
    top3_indices = np.argsort(probabilities)[::-1][:3]
    top3 = []
    for idx in top3_indices:
        idx = int(idx)
        c, d = CLASS_INFO[idx]
        top3.append({
            "class_name": CLASS_NAMES[idx],
            "crop": c,
            "disease": d,
            "confidence": float(probabilities[idx]) * 100,
        })

    return {
        "predicted_class": predicted_class,
        "crop": crop,
        "disease": disease,
        "confidence": confidence,
        "probabilities": probabilities,
        "top3": top3,
        "inference_time_ms": inference_time_ms,
    }
