# Generated by Django 4.2 on 2023-04-08 07:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("model", "0008_model_model_riskstatus"),
    ]

    operations = [
        migrations.AddField(
            model_name="model",
            name="model_risklist",
            field=models.CharField(default="暂无概率数据", max_length=100),
        ),
    ]
