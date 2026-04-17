import cv2
import numpy as np

def load_and_preprocess_image(image_path, target_size=(299, 299)):
    """
    Loads an image from disk and preprocesses for Xception.
    Returns float32 array shape (1, H, W, 3).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32)

    # Xception expects inputs scaled between -1 and 1
    img = (img / 127.5) - 1.0
    img = np.expand_dims(img, axis=0)
    return img