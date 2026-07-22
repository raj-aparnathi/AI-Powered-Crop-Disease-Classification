"""
CropX - Class Name Definitions
================================
56 crop disease classes extracted from the training dataset.
Ordered alphabetically to match tf.keras.utils.image_dataset_from_directory.

Each class maps to a (Crop Name, Disease Name) tuple for structured display.
"""

# Ordered class labels matching the model's output indices
CLASS_NAMES = [
    "Cotton___Aphids",
    "Cotton___Army_worm",
    "Cotton___Bacterial_Blight",
    "Cotton___Healthy",
    "Cotton___Powdery_Mildew",
    "Cotton___Target_spot",
    "Groundnut___early_leaf_spot",
    "Groundnut___healthy leaf",
    "Groundnut___late leaf spot",
    "Groundnut___nutrition deficiency",
    "Groundnut___rust",
    "Maize___Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Maize___Corn_(maize)___Common_rust_",
    "Maize___Corn_(maize)___Northern_Leaf_Blight",
    "Maize___Corn_(maize)___healthy",
    "Onion___Alternaria_D",
    "Onion___Botrytis Leaf Blight",
    "Onion___Bulb_blight-D",
    "Onion___Caterpillar-P",
    "Onion___Downy mildew",
    "Onion___Fusarium-D",
    "Onion___Healthy leaves",
    "Onion___Iris yellow virus_augment",
    "Onion___Purple blotch",
    "Onion___Rust",
    "Onion___Virosis-D",
    "Onion___Xanthomonas Leaf Blight",
    "Onion___onion1",
    "Onion___stemphylium Leaf Blight",
    "Potato___Potato___Early_blight",
    "Potato___Potato___Late_blight",
    "Potato___Potato___healthy",
    "Tomato___Tomato___Bacterial_spot",
    "Tomato___Tomato___Early_blight",
    "Tomato___Tomato___Late_blight",
    "Tomato___Tomato___Leaf_Mold",
    "Tomato___Tomato___Septoria_leaf_spot",
    "Tomato___Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Tomato___Target_Spot",
    "Tomato___Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato___healthy",
    "Wheat___Aphid",
    "Wheat___Black Rust",
    "Wheat___Blast",
    "Wheat___Brown Rust",
    "Wheat___Common Root Rot",
    "Wheat___Fusarium Head Blight",
    "Wheat___Healthy",
    "Wheat___Leaf Blight",
    "Wheat___Mildew",
    "Wheat___Mite",
    "Wheat___Smut",
    "Wheat___Stem fly",
    "Wheat___Tan spot",
    "Wheat___Yellow Rust",
    "Wheat___septoria",
]

NUM_CLASSES = len(CLASS_NAMES)


def parse_class_label(class_name: str) -> tuple[str, str]:
    """
    Parse a raw class label into (crop_name, disease_name).

    Handles triple-underscore formats like 'Tomato___Tomato___Early_blight'
    and double-underscore formats like 'Cotton___Aphids'.

    Args:
        class_name: Raw class label string from the model.

    Returns:
        Tuple of (crop_name, disease_name) with cleaned formatting.
    """
    parts = class_name.split("___")

    # Extract crop name (always the first part)
    crop = parts[0].strip()

    # Extract disease name (last meaningful part)
    if len(parts) >= 3:
        # Format: Crop___SubCrop___Disease (e.g., Tomato___Tomato___Early_blight)
        disease_raw = parts[-1].strip()
    elif len(parts) == 2:
        disease_raw = parts[1].strip()
    else:
        disease_raw = "Unknown"

    # Clean up underscores and formatting
    disease = disease_raw.replace("_", " ").strip()

    # Handle "healthy" variants consistently
    if disease.lower() in ("healthy", "healthy leaves", "healthy leaf"):
        disease = "Healthy"

    return crop, disease


# Precomputed mapping: index -> (crop, disease)
CLASS_INFO = {i: parse_class_label(name) for i, name in enumerate(CLASS_NAMES)}
