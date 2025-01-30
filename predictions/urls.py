from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_images, name='upload_images'),
    path('api/predict/', views.predict_signature_fraud, name='predict_signature_fraud'),
    path('api/signature/', views.upload_images_api, name='predict_signature_fraud'),
    path('signature/', views.signature, name='predict_signature_fraud'),

]



