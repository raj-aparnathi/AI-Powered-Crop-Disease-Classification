import time
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

from class_names import CLASS_NAMES, CLASS_INFO, NUM_CLASSES

IMG_SIZE = (224, 224)
MODEL_PATH = Path(__file__).parent / "model" / "CropX_MobileNetV3_Best.keras"



def load_model() -> tf.keras.Model:
    
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



def preprocess_image(image: Image.Image) -> np.ndarray:
    
    if image.mode != "RGB":
        image = image.convert("RGB")

  
    image = image.resize(IMG_SIZE, Image.LANCZOS)

    
    img_array = np.array(image, dtype=np.float32)

    preprocessing (scales pixels to [-1, 1])
    img_array = preprocess_input(img_array)

   
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict(model: tf.keras.Model, img_array: np.ndarray) -> dict:
   
    start_time = time.perf_counter()
    predictions = model.predict(img_array, verbose=0)
    end_time = time.perf_counter()

    inference_time_ms = (end_time - start_time) * 1000
    probabilities = predictions[0]

  
    top_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[top_idx]) * 100
    predicted_class = CLASS_NAMES[top_idx]
    crop, disease = CLASS_INFO[top_idx]

    
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
