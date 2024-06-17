
from rest_framework.views import APIView
from rest_framework.response import Response
from Prediction.p.pred import pred
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from .serializer import MyPhotoSerializer
from rest_framework import generics
from rest_framework import status
from .models import Image
from PIL import Image 
import base64
import cv2
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework.renderers import TemplateHTMLRenderer, JSONRenderer

def to_image(numpy_img):
    img = Image.fromarray(numpy_img, 'RGB')
    return img
import base64
from io import BytesIO
def to_data_uri(pil_img):
    data = BytesIO()
    pil_img.save(data, "JPEG") # pick your format
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8') 
def home(request):
    print('hi')

class PredictAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser) # This allows parsing of file uploads
    renderer_classes = [TemplateHTMLRenderer, JSONRenderer]
    serializer_class = MyPhotoSerializer
    def get(self, request, *args, **kwargs):
        serializer = MyPhotoSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, template_name='colorized_image.html')
    def post(self, request, *args, **kwargs):
        serializer = MyPhotoSerializer(data=request.data)
        if serializer.is_valid():
            image = request.FILES.get('image')
            print(type(image))
            pred_obj = pred(image)
            colorized_image = pred_obj.predict_colorization()
            # Convert NumPy array to base64-encoded image
            _, buffer = cv2.imencode('.jpg', colorized_image)
            colorized_image_base64 = base64.b64encode(buffer).decode()
            
            return Response({'colorized_image_base64': colorized_image_base64}, template_name='colorized_image.html')


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')
