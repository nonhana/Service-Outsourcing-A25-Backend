from django.db import models
from user.models import User


class Model(models.Model):
    model_id = models.AutoField(primary_key=True)
    model_name = models.CharField(max_length=100, default="默认模型名称")
    model_type = models.CharField(max_length=50, default="默认模型类型")
    model_detail = models.CharField(max_length=200, default="该用户没有输入任何描述哦~")
    model_picture = models.CharField(
        max_length=500, default="https://dummyimage.com/400X400")
    model_nodes = models.TextField(max_length=10000)
    model_edges = models.TextField(max_length=20000)
    create_time = models.CharField(max_length=100)
    update_time = models.CharField(max_length=100)
    update_method = models.IntegerField(max_length=10, default=0)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
