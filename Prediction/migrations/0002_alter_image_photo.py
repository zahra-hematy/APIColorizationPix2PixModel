# Generated by Django 5.0.4 on 2024-04-30 16:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Prediction', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='photo',
            field=models.ImageField(upload_to='photos', verbose_name='تصویر'),
        ),
    ]
