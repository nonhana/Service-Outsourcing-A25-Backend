from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
import json
import os
import uuid
import base64
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
            'id', 'username', 'account', 'head_photo', 'background_photo')
        userinfo = []
        for item in source:
            userinfo.append(item)
        if userinfo:
            return JsonResponse({
                'result_code': 0,
                'result_msg': '登录成功！',
                'userinfo': userinfo[0]
            })


# 获取用户资料
def getuserinfo(request):
    id = request.GET.get('user_id')
    user = User.objects.filter(id=id).values(
        'id', 'username', 'account', 'head_photo', 'background_photo')[0]
    if user:
        return JsonResponse({
            'result_code': 0,
            'result_msg': '获取用户信息成功',
            'userinfo': user
        })


# 将Base64编码的图片保存到本地
def uploadphoto(request):
    def generate_image_name():
        return str(uuid.uuid4()) + ".png"
    # 获取 Base64 编码的图片数据
    base64_data = json.loads(request.body)['img_data']
    # 将 Base64 编码的图片数据解码为二进制数据
    format, imgstr = base64_data.split(';base64,')
    ext = format.split('/')[-1]
    data = ContentFile(base64.b64decode(imgstr), name='temp.' + ext)
    # 将二进制数据写入本地文件
    # 构造相对于项目根目录的路径
    filename = generate_image_name()
    file_path = os.path.join(settings.STATICFILES_DIRS[0],
                             'user_photos', filename)
    default_storage.save(file_path, data)
    img_url = 'http://127.0.0.1/static/user_photos/'+filename
    return JsonResponse({
        'result_code': 0,
        'result_msg': "保存图片成功",
        'img_url': img_url
    })


# 删除指定名称的文件
def deletefile(request):
    img_url = json.loads(request.body)['img_url']
    filename = os.path.basename(img_url)
    # 构建要删除的文件路径
    file_path = os.path.join(settings.STATICFILES_DIRS[0],
                             'user_photos', filename)

    # 检查文件是否存在
    if os.path.exists(file_path):
        # 删除文件
        os.remove(file_path)
        return JsonResponse({
            'result_code': 0,
            'result_msg': "删除图片成功",
        })
    else:
        return JsonResponse({
            'result_code': 1,
            'result_msg': "图片不存在",
        })


# 更新用户个人信息
def updateuserinfo(request):
    info = json.loads(request.body)
    id = info['id']
    user = User.objects.get(id=id)  # 直接获取模型实例对象
    user.username = info['username']
    user.head_photo = info['head_photo']
    user.background_photo = info['background_photo']
    user.save()  # 调用save()方法将更改保存到数据库
    return JsonResponse({
        'result_code': 0,
        'result_msg': '修改用户信息成功',
    })
