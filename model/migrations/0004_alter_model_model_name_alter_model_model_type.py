# Generated by Django 4.1.7 on 2023-03-23 07:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("model", "0003_model_model_picture"),
    ]

    operations = [
        migrations.AlterField(
            model_name="model",
            name="model_name",
            field=models.CharField(default="默认模型名称", max_length=100),
        ),
        migrations.AlterField(
            model_name="model",
            name="model_type",
            field=models.CharField(default="默认模型类型", max_length=50),
        ),
    ]
