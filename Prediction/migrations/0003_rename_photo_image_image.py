# Generated by Django 5.0.4 on 2024-05-01 07:52

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('Prediction', '0002_alter_image_photo'),
    ]

    operations = [
        migrations.RenameField(
            model_name='image',
            old_name='photo',
            new_name='image',
        ),
    ]
