from django.db import models
from django.contrib.auth import get_user_model
User = get_user_model()

# Create your models here.

    

from django.db import models
import base64
from django.core.files.base import ContentFile
def get_covert_path(obj, fn):
    ex = os.path.splitext(fn)[1]
    uid = uuid.uuid5(uuid.NAMESPACE_URL,f"store-book-{obj.pk}" )
    path =datetime.now().strftime(f"get_image/%Y/%m/%d/{uid}{ex}")
    return path


class Image(models.Model):
    image = models.ImageField(upload_to='photos', verbose_name='تصویر')
    # path = models.CharField(max_length=255)
    # user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="imageup", verbose_name='ارسال کننده')

