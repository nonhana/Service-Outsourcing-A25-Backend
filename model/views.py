import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
# 导入model表定义
from model.models import Model


# 定义读取文件的类
class DataSource:
    def __init__(self):
        self.model_nodes = {}
        self.model_edges = []
        # 节点类别列表
        self.node_type = ['n', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6']
        # 边类别列表
        self.edge_type = ['r1_startnode', 'r1_name', 'r1_endnode',
                          'r2_startnode', 'r2_name', 'r2_endnode',
                          'r3_startnode', 'r3_name', 'r3_endnode',
                          'r4_startnode', 'r4_name', 'r4_endnode',
                          'r5_startnode', 'r5_name', 'r5_endnode',
                          'r6_startnode', 'r6_name', 'r6_endnode']
        # 根节点
        self.n = []
        # 第一条边的相关属性
        self.r1_startnode = []
        self.r1_name = []
        self.r1_endnode = []
        # 第二节点
        self.m1 = []
        # 第二条边的相关属性
        self.r2_startnode = []
        self.r2_name = []
        self.r2_endnode = []
        # 第三节点
        self.m2 = []
        # 第三条边的相关属性
        self.r3_startnode = []
        self.r3_name = []
        self.r3_endnode = []
        # 第四节点==
        self.m3 = []
        # 第四条边的相关属性
        self.r4_startnode = []
        self.r4_name = []
        self.r4_endnode = []
        # 第五节点(公司)
        self.m4 = []
        # 第五条边的相关属性
        self.r5_startnode = []
        self.r5_name = []
        self.r5_endnode = []
        # 第六节点(产品小类)
        self.m5 = []
        # 第六条边的相关属性
        self.r6_startnode = []
        self.r6_name = []
        self.r6_endnode = []
        # 第七节点
        self.m6 = []

    def get_node_value(self, key):
        if key in self.node_type:
            return getattr(self, key)
        else:
            raise ValueError(f"Invalid key: {key}")

    def get_edge_value(self, key):
        if key in self.edge_type:
            return getattr(self, key)
        else:
            raise ValueError(f"Invalid key: {key}")

    def save_data(self, filename):
        file_handler = open(filename, mode='r')
        node_num = 0
        node_flag = False
        edge_num = 0
        edge_flag = 0
        for line in file_handler:
            # 设置读取状态
            if line.strip().find('节点start===') != -1:
                node_num = node_num+1
                node_flag = True
            if line.strip().find('节点end===') != -1:
                node_flag = False
            if line.strip().find('边start===') != -1:
                edge_num = edge_num+1
                edge_flag = 1
            if edge_flag > 0 and line.strip() == '':
                edge_flag = edge_flag+1
            if line.strip().find('边end===') != -1:
                edge_flag = 0

            # 读取节点
            if node_flag and line.strip() != '' and line.strip().find('节点end===') == -1 and line.strip().find('节点start===') == -1:
                if line.strip() not in self.get_node_value(self.node_type[node_num-1]):
                    node_item = {
                        'label': line.strip(),
                        'value': line.strip()
                    }
                    self.get_node_value(
                        self.node_type[node_num-1]).append(node_item)

            # 读取边
            if edge_flag == 1 and line.strip() != '' and line.strip().find('边end===') == -1 and line.strip().find('边start===') == -1:
                if edge_num == 1:
                    self.r1_startnode.append(line.strip())
                if edge_num == 2:
                    self.r2_startnode.append(line.strip())
                if edge_num == 3:
                    self.r3_startnode.append(line.strip())
                if edge_num == 4:
                    self.r4_startnode.append(line.strip())
                if edge_num == 5:
                    self.r5_startnode.append(line.strip())
                if edge_num == 6:
                    self.r6_startnode.append(line.strip())
            if edge_flag == 2 and line.strip() != '' and line.strip().find('边end===') == -1 and line.strip().find('边start===') == -1:
                if edge_num == 1:
                    self.r1_name.append(line.strip())
                if edge_num == 2:
                    self.r2_name.append(line.strip())
                if edge_num == 3:
                    self.r3_name.append(line.strip())
                if edge_num == 4:
                    self.r4_name.append(line.strip())
                if edge_num == 5:
                    self.r5_name.append(line.strip())
                if edge_num == 6:
                    self.r6_name.append(line.strip())
            if edge_flag == 3 and line.strip() != '' and line.strip().find('边end===') == -1 and line.strip().find('边start===') == -1:
                if edge_num == 1:
                    self.r1_endnode.append(line.strip())
                if edge_num == 2:
                    self.r2_endnode.append(line.strip())
                if edge_num == 3:
                    self.r3_endnode.append(line.strip())
                if edge_num == 4:
                    self.r4_endnode.append(line.strip())
                if edge_num == 5:
                    self.r5_endnode.append(line.strip())
                if edge_num == 6:
                    self.r6_endnode.append(line.strip())

        # 节点转为字典
        for item in self.node_type:
            self.model_nodes[item] = self.get_node_value(item)
        # 边转为字典
        for i in range(int(len(self.edge_type)/3)):
            for index in range(len(self.get_edge_value(self.edge_type[i*3+0]))):
                edge_item = {
                    'id': len(self.model_edges),
                    'start': self.get_edge_value(self.edge_type[i*3+0])[index],
                    'name': self.get_edge_value(self.edge_type[i*3+1])[index],
                    'end': self.get_edge_value(self.edge_type[i*3+2])[index]
                }
                self.model_edges.append(edge_item)


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


# 根据id获取产业链模型
def modelinfo(request):
    model_id = request.GET.get('model_id')
    model = Model.objects.filter(model_id=model_id).values()[0]

    return JsonResponse({
        'result_code': 0,
        'result_msg': "获取产业链数据成功",
        'model_data': model,
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
    model = Model.objects.create(model_name=info['model_name'],
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
        'model_id': model.model_id
    })


# 接收前端传来的txt文件，并保存在本地，之后转存数据库
@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if file:
            filename = file.name
            parent_dir = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), '..')
            file_path = os.path.join(
                parent_dir, 'static', 'model_file', filename)
            with open(file_path, 'wb') as f:
                for chunk in file.chunks():
                    f.write(chunk)

            saved_file = DataSource()
            saved_file.save_data(file_path)
            model_nodes = saved_file.model_nodes
            model_edges = saved_file.model_edges

            return JsonResponse({
                'result_code': 0,
                'result_msg': "txt文件上传成功",
                'model_nodes': model_nodes,
                'model_edges': model_edges
            })
        else:
            return JsonResponse({
                'result_code': 1,
                'result_msg': "txt文件上传失败，请重新上传"
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
        'risk_main': "所会发生的主要风险",
        'risk_method': "规避风险的主要措施"
    })
