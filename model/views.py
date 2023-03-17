from django.shortcuts import render
from django.http import JsonResponse
import json
# 导入model表定义
from model.models import Model


# 返回所有提交过的产业链模型
def modellist(request):
    model_list = list(Model.objects.values(
        'model_id', 'model_name', 'model_detail', 'model_picture', 'user_id'))
    if len(model_list) > 0:
        return JsonResponse({
            'result_code': 0,
            'result_msg': "获取产业链列表成功",
            'model_list': model_list
        })


# 提交产业链模型到数据库
def uploadmodel(request):
    info = json.loads(request.body)
    model_list = list(Model.objects.values())
    for item in model_list:
        if info['model_nodes'] == item['model_nodes'] and info['model_edges'] == item['model_edges']:
            return JsonResponse({
                'result_code': 1,
                'result_msg': '请勿重复提交相同模型',
            })
    Model.objects.create(model_name=info['model_name'],
                         model_type=info['model_type'],
                         model_detail=info['model_detail'],
                         model_nodes=info['model_nodes'],
                         model_edges=info['model_edges'],
                         create_time=info['create_time'],
                         update_time=info['update_time'],
                         user_id=info['user_id'],)
    return JsonResponse({
        'result_code': 0,
        'result_msg': '提交模型成功',
    })


# 对指定的产业链模型进行完整性评估
def integrity(request):
    model_id = request.GET.get('model_id')
    model = Model.objects.filter(model_id=model_id).values(
        'model_id', 'model_nodes', 'model_edges')[0]

    return JsonResponse({
        'result_code': 0,
        'result_msg': "完整性分析成功",
        'model_data': model,
        'integrity_score': 100.0,
        'integrity_evaluation': "完整性分析的评价"
    })


# 对指定的产业链模型进行风险评估
def riskanalyse(request):
    model_id = request.GET.get('model_id')
    model = Model.objects.filter(model_id=model_id).values(
        'model_id', 'model_nodes', 'model_edges')[0]

    return JsonResponse({
        'result_code': 0,
        'result_msg': "风险评估成功",
        'model_data': model,
        'risk_score': 100.0,
        'risk_main': "所会发生的主要风险"
    })
