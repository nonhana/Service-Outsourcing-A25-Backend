from django.db import models
from django.contrib import admin


class User(models.Model):
    username = models.CharField(max_length=100)
    account = models.CharField(max_length=200)
    password = models.CharField(max_length=200)
    head_photo = models.CharField(
        max_length=500, default="https://dummyimage.com/400X400")
    background_photo = models.CharField(
        max_length=500, default="https://dummyimage.com/400X400")


admin.site.register(User)
