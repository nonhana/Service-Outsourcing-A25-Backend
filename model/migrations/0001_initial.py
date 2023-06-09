# Generated by Django 4.1.7 on 2023-03-16 11:32

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("user", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Model",
            fields=[
                ("model_id", models.AutoField(primary_key=True, serialize=False)),
                ("model_name", models.CharField(max_length=100)),
                ("model_type", models.CharField(max_length=50)),
                ("model_nodes", models.TextField(max_length=10000)),
                ("model_edges", models.TextField(max_length=20000)),
                ("create_time", models.CharField(max_length=100)),
                ("update_time", models.CharField(max_length=100)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="user.user"
                    ),
                ),
            ],
        ),
    ]
