# Generated by Django 4.2 on 2023-04-08 07:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("model", "0006_alter_model_update_method"),
    ]

    operations = [
        migrations.AddField(
            model_name="model",
            name="model_riskmain",
            field=models.CharField(default="暂无风险", max_length=100),
        ),
        migrations.AddField(
            model_name="model",
            name="model_riskmethods",
            field=models.CharField(default="暂无相应方法", max_length=2000),
        ),
        migrations.AddField(
            model_name="model",
            name="model_riskscore",
            field=models.IntegerField(default=100),
        ),
    ]
