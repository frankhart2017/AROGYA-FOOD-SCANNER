from django.shortcuts import render
from django.http import JsonResponse
from arogya_model_app.predict_image import predict

# Create your views here.
def index(requests):
    img = requests.GET['img']
    prediction = predict(img)
    return JsonResponse({'prediction': prediction})
