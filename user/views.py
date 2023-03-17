from django.shortcuts import render
from django.http import JsonResponse
import json
# token，对象序列化
from django.core import signing, serializers
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
    user_list = list(User.objects.values())
    for item in user_list:
        if info['username'] == item['username']:
            return JsonResponse({
                'result_code': 1,
                'result_msg': '这个用户名被注册了，换一个吧！',
            })
        if info['account'] == item['account']:
            return JsonResponse({
                'result_code': 1,
                'result_msg': '这个账号已经被注册了，请重新输入！',
            })
    User.objects.create(username=info['username'],
                        account=info['account'],
                        password=info['password'])
    return JsonResponse({
        'result_code': 0,
        'result_msg': 'user register succeeded',
    })


# 用户登录
def login(request):
    account = request.GET.get('account')
    password = request.GET.get('password')
    user_list = list(User.objects.values())
    account_flag = False
    password_flag = False
    for item in user_list:
        if account == item['account']:
            account_flag = True
    if account_flag:
        for item in user_list:
            if password == item['password']:
                password_flag = True
    if account_flag == False:
        return JsonResponse({
            'result_code': 1,
            'result_msg': '您的账号尚未注册哦'
        })
    if password_flag == False:
        return JsonResponse({
            'result_code': 1,
            'result_msg': '密码输入错误！'
        })
    if account_flag and password_flag:
        source = User.objects.filter(account=account, password=password).values(
            'id', 'username', 'account')
        userinfo = []
        for item in source:
            userinfo.append(item)
        if userinfo:
            return JsonResponse({
                'result_code': 0,
                'result_msg': '登录成功！',
                'userinfo': userinfo[0]
            })
