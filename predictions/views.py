from django.shortcuts import render
from keras.models import load_model
from keras.saving import register_keras_serializable
from PIL import Image
import numpy as np
import os
from django.conf import settings
import tensorflow as tf
import cv2
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import io
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



@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def predict_signature_fraud(request):
    #import pdb; pdb.set_trace()
    """
    API view to handle signature fraud detection.
    Accepts two images and returns prediction and genuineness status.
    """
    if 'image1' not in request.FILES or 'image2' not in request.FILES:
        return Response({'error': 'Please upload both images.'}, status=status.HTTP_400_BAD_REQUEST)

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
        return Response({'error': f"Error saving files: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Preprocess images
    try:
        image1 = preprocess_image(file1_path)
        image2 = preprocess_image(file2_path)
    except Exception as e:
        return Response({'error': f"Error processing images: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

    # Generate prediction
    try:
        prediction = model.predict([image1, image2])[0][0]
        fraud_status = "Fraudulent" if prediction < 0.5 else "Genuine"
    except Exception as e:
        return Response({'error': f"Error during prediction: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Return the response as JSON
    return Response({
        'fraud_status': fraud_status,
        'probability': round(prediction, 2)
    }, status=status.HTTP_200_OK)


@api_view(['POST'])
def upload_images_api(request):
    if 'image1' not in request.FILES or 'image2' not in request.FILES:
        return Response({'error': 'Please upload both images.'}, status=status.HTTP_400_BAD_REQUEST)

    # Get uploaded files
    file1 = request.FILES['image1']
    file2 = request.FILES['image2']
    image1 = preprocess_image(file1)
    image2 = preprocess_image(file2)
    prediction = model.predict([image1, image2])[0][0]
    fraud_status = "Fraudulent" if prediction < 0.5 else "Genuine"
    return Response({
        'fraud_status': fraud_status,
        'probability': round(prediction, 2),
    }, status=status.HTTP_200_OK)

from PIL import Image
import numpy as np
import io

def preprocess_image2(image_file, target_size=(128, 128)):
    from PIL import Image
    import numpy as np
    import io

    # Open the image using Pillow
    image = Image.open(io.BytesIO(image_file.read())).convert("L")  # Grayscale

    # Resize the image
    image = image.resize(target_size)

    # Convert to numpy array
    image = np.array(image)

    # Normalize pixel values
    image = image / 255.0

    # Add channel dimension
    image = np.expand_dims(image, axis=-1)

    return image



def signature(request):
    if request.method == "POST":
        image_path1 = request.FILES['image1']
        image_path2 = request.FILES['image2']

        # Preprocess the images
        image1 = preprocess_image2(image_path1)
        image2 = preprocess_image2(image_path2)
        image1 = np.expand_dims(image1, axis=0)  # Shape: (1, 128, 128, 1)
        image2 = np.expand_dims(image2, axis=0) 

        # Predict
        prediction = model.predict([np.expand_dims(image1, axis=0), np.expand_dims(image2, axis=0)])
        
        threshold = 0.5
        is_fraud = prediction[0][0] < threshold
        decision = "Fraudulent" if is_fraud else "Genuine"
        print(f"Prediction: {decision} (Prediction: {prediction[0][0]:.2f})")

        return render(request, 'result.html', {'prediction': decision, 'probability': round(prediction[0][0], 2)})

    return render(request, 'upload.html')
