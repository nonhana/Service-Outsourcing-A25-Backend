from django.shortcuts import render
from django.http import JsonResponse
import json
# 导入user表定义
from user.models import User


# 获取全部用户
def userlist(request):
    qs = User.objects.values()
    user_list = list(qs)
    return JsonResponse({
        'result_code': 0,
        'result_msg': 'get user list succeeded',
        'user_list': user_list
    })


# 用户注册
def register(request):
    # 用json.loads加载前端传来的json数据
    info = json.loads(request.body)
    User.objects.create(username=info['username'],
                        account=info['account'],
                        password=info['password'])
    return JsonResponse({
        'result_code': 0,
        'result_msg': 'user register succeeded',
    })


# 用户登录
def login(request):
    account = request.params['account']
    password = request.params['password']

    user_list = list(User.objects.values())

    flag = True
    
