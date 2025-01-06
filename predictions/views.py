from django.shortcuts import render
from keras.models import load_model
from keras.saving import register_keras_serializable
from PIL import Image
import numpy as np
import os
from django.conf import settings
import tensorflow as tf
import cv2

# Custom layer serialization
@register_keras_serializable()
def l1_distance(inputs):
    x, y = inputs
    return tf.abs(x - y)

# Load the pre-trained model
model_path = os.path.join(
    settings.BASE_DIR, 'model', 'signature_fraud.keras'
)  # Use dynamic path for portability
model = load_model(model_path, custom_objects={'l1_distance': l1_distance})

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess the image for model prediction.
    Converts image to grayscale, resizes, normalizes, and expands dimensions.
    """
    # Load the image as grayscale using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to load image at {image_path}")

    # Resize to target size
    image = cv2.resize(image, target_size)

    # Normalize pixel values to [0, 1]
    image = image / 255.0

    # Add batch and channel dimensions
    image = np.expand_dims(image, axis=(0, -1))
    return image

def upload_images(request):
    """
    Handles image upload, preprocessing, and prediction.
    """
    if request.method == 'POST':
        # Validate uploaded files
        if 'image1' not in request.FILES or 'image2' not in request.FILES:
            return render(request, 'upload.html', {'error': 'Please upload both images.'})

        file1 = request.FILES['image1']
        file2 = request.FILES['image2']

        # Create upload directory if it doesn't exist
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        # Save the uploaded files
        file1_path = os.path.join(upload_dir, file1.name)
        file2_path = os.path.join(upload_dir, file2.name)

        try:
            with open(file1_path, 'wb+') as destination:
                for chunk in file1.chunks():
                    destination.write(chunk)

            with open(file2_path, 'wb+') as destination:
                for chunk in file2.chunks():
                    destination.write(chunk)
        except Exception as e:
            return render(request, 'upload.html', {'error': f"Error saving files: {str(e)}"})

        # Preprocess images
        try:
            image1 = preprocess_image(file1_path)
            image2 = preprocess_image(file2_path)
        except Exception as e:
            return render(request, 'upload.html', {'error': f"Error processing images: {str(e)}"})

        # Generate prediction
        try:
            prediction = model.predict([image1, image2])[0][0]
            fraud_status = "Fraudulent" if prediction < 0.5 else "Genuine"
        except Exception as e:
            return render(request, 'upload.html', {'error': f"Error during prediction: {str(e)}"})

        return render(request, 'result.html', {
            'prediction': fraud_status,
        'probability': round(prediction, 2),
        'image1': f"/media/uploads/{file1.name}",
        'image2': f"/media/uploads/{file2.name}"
    })


    # Render upload page for GET requests
    return render(request, 'upload.html')
