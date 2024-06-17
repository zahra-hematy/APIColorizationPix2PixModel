

from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
# from Predict.Prediction import pred
from .views import *
from rest_framework import serializers
from rest_framework import generics  
from .models import *
from drf_extra_fields.fields import Base64ImageField
import base64, uuid
from django.core.files.base import ContentFile
from rest_framework import serializers


# Custom image field - handles base 64 encoded images
class Base64ImageField(serializers.ImageField):
    def to_internal_value(self, data):
        if isinstance(data, str) and data.startswith('data:image'):
            # base64 encoded image - decode
            format, imgstr = data.split(';base64,') # format ~= data:image/X,
            ext = format.split('/')[-1] # guess file extension
            id = uuid.uuid4()
            data = ContentFile(base64.b64decode(imgstr), name = id.urn[9:] + '.' + ext)
        return super(Base64ImageField, self).to_internal_value(data)

class MyPhotoSerializer(serializers.ModelSerializer):
    image=Base64ImageField(required=False) 
    class Meta:
        model = Image
        fields = ['id',  'image']
    
class PredictSerializer(serializers.Serializer):
    image = serializers.ImageField()
