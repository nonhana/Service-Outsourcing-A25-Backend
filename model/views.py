import os
import uuid
import base64
import random
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import json
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.nn import Linear
# 导入model表定义
from model.models import Model


# 定义读取文件的类
class FileToData:
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
                node_num = node_num + 1
                node_flag = True
            if line.strip().find('节点end===') != -1:
                node_flag = False
            if line.strip().find('边start===') != -1:
                edge_num = edge_num + 1
                edge_flag = 1
            if edge_flag > 0 and line.strip() == '':
                edge_flag = edge_flag + 1
            if line.strip().find('边end===') != -1:
                edge_flag = 0

            # 读取节点
            if node_flag and line.strip() != '' and line.strip().find('节点end===') == -1 and line.strip().find(
                    '节点start===') == -1:
                if line.strip() not in self.get_node_value(self.node_type[node_num - 1]):
                    node_item = {
                        'label': line.strip(),
                        'value': line.strip()
                    }
                    self.get_node_value(
                        self.node_type[node_num - 1]).append(node_item)

            # 读取边
            if edge_flag == 1 and line.strip() != '' and line.strip().find('边end===') == -1 and line.strip().find(
                    '边start===') == -1:
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
            if edge_flag == 2 and line.strip() != '' and line.strip().find('边end===') == -1 and line.strip().find(
                    '边start===') == -1:
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
            if edge_flag == 3 and line.strip() != '' and line.strip().find('边end===') == -1 and line.strip().find(
                    '边start===') == -1:
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
        for i in range(int(len(self.edge_type) / 3)):
            for index in range(len(self.get_edge_value(self.edge_type[i * 3 + 0]))):
                edge_item = {
                    'id': len(self.model_edges),
                    'start': self.get_edge_value(self.edge_type[i * 3 + 0])[index],
                    'name': self.get_edge_value(self.edge_type[i * 3 + 1])[index],
                    'end': self.get_edge_value(self.edge_type[i * 3 + 2])[index]
                }
                self.model_edges.append(edge_item)


# 实现邻接表
class Vertex:
    def __init__(self, key, type, name):
        self.id = key
        self.type = type
        self.name = name
        self.connectedTo = {}

    # 从这个顶点添加一个连接到另一个
    def addNeighbor(self, nbr, name, weight=0):
        self.connectedTo[nbr] = [weight, name]

    # 修改str
    def __str__(self):
        return str(self.id) + 'connectedTo' + str(
            [x.id for x in self.connectedTo])

    # 返回邻接表中的所有的项点
    def getConnections(self):
        return self.connectedTo.items()

    def getId(self):
        return self.id

    # 返回从这个顶点到作为参数顶点的边的权重和名字
    def getweight(self, nbr):
        return self.connectedTo[nbr]


# 实现图
class IndustryGraph:
    def __init__(self):
        # {}代表字典
        self.vertList = {}
        # 邻接矩阵
        self.matrix = []
        # 总的节点个数
        self.numVertices = 0
        self.visble = nx.Graph()
        # 特征向量
        self.feature_vector = []
        # 记录第n个节点的位置n
        self.position = []
        # 节点名称数组
        self.name_labels = []
        # 节点标签数组
        self.labels = []
        # 边矩阵
        self.edge_matrix = []
        # 边对象矩阵
        self.edge_list = []
        # nx的图对象
        self.G = nx.DiGraph()

    # 增加顶点
    def addVertex(self, key, type, name):
        self.position.append(self.numVertices)
        if name not in self.name_labels:
            self.name_labels.append(name)
        self.numVertices += 1
        self.visble.add_node(key)
        newVertex = Vertex(key, type, name)
        self.vertList[key] = newVertex
        return newVertex

    # 增加边
    def addEdge(self, f, t, name, const=0):
        # 起始点，目标点，权重。
        # 注意点：f,t是vertlist中的数组下标，不是target的id
        if f not in self.vertList:
            nv = self.addVertex(f, "default_type", "default_name")
        if t not in self.vertList:
            nv = self.addVertex(t, "default_type", "default_name")
        self.matrix[f][t] = const
        self.visble.add_edge(f, t)
        self.vertList[f].addNeighbor(self.vertList[t], name, const)

    # 增加对象
    def add_node_edge(self, node, edge):
        self.G.add_nodes_from(node)
        self.G.add_weighted_edges_from(edge)

    # 初始化邻接矩阵
    def initMatrix(self, nodenum):
        for i in range(nodenum):
            row = []
            for j in range(nodenum):
                row.append(0)
            self.matrix.append(row)

    # 根据邻接矩阵写出边矩阵
    def build_edge_matrix(self):
        start = []
        end = []
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if self.matrix[j][i] != 0:
                    start.append(j)
                    end.append(i)
        self.edge_matrix.append(start)
        self.edge_matrix.append(end)
        return

    # 获取所有顶点
    def getVertices(self):
        return len(self.vertList.values())

    # 使用迭代器返回所有的邻接表信息
    def __iter__(self):
        return iter(self.vertList.values())

    # 特征值
    def feature_calculate(self):
        # 计算节点的中心度
        eigenvector = nx.eigenvector_centrality(self.visble)
        list = []
        for item in eigenvector:
            list.append(eigenvector[item])
        # 计算度中心度 紧密中心度 中介中心度
        self.add_node_edge(self.name_labels, self.edge_list)
        d = nx.degree_centrality(self.G)
        c = nx.closeness_centrality(self.G)
        b = nx.betweenness_centrality(self.G)
        for v in self.G.nodes():
            feature = []
            feature.append(d[v])
            feature.append(c[v])
            feature.append(b[v])
            self.feature_vector.append(feature)
        for i in range(len(list)):
            self.feature_vector[i].append(list[i])
        return


class DataSet:
    def __init__(self, model_nodes, model_edges):
        # 添加顶点
        # 创建一个空图
        self.g = IndustryGraph()
        self.nodes_num = 0
        self.edges_num = 0
        # 根节点
        self.g.addVertex(self.g.getVertices(), "industry",
                         model_nodes['n'][0]['label'])
        self.g.labels.append(1)
        # 一级产业
        for item in model_nodes['m1']:
            if item['label'] not in self.g.name_labels:
                self.g.addVertex(self.g.getVertices(),
                                 "industry", item['label'])
                self.g.labels.append(2)
        # 二级产业
        for item in model_nodes['m2']:
            if item['label'] not in self.g.name_labels:
                self.g.addVertex(self.g.getVertices(),
                                 "industry", item['label'])
                self.g.labels.append(3)
        # 公司
        for item in model_nodes['m3']:
            if item['label'] not in self.g.name_labels:
                self.g.addVertex(self.g.getVertices(),
                                 "company", item['label'])
                self.g.labels.append(4)
        # 主营产品
        for item in model_nodes['m4']:
            if item['label'] not in self.g.name_labels:
                self.g.addVertex(self.g.getVertices(),
                                 "product", item['label'])
                self.g.labels.append(5)
        # 产品小类
        for item in model_nodes['m5']:
            if item['label'] not in self.g.name_labels:
                self.g.addVertex(self.g.getVertices(),
                                 "littleproduct", item['label'])
                self.g.labels.append(6)
        # 上游材料
        for item in model_nodes['m6']:
            if item['label'] not in self.g.name_labels:
                self.g.addVertex(self.g.getVertices(),
                                 "material", item['label'])
                self.g.labels.append(7)

        # 初始化邻接矩阵
        self.g.initMatrix(self.g.getVertices())

        # 添加边
        for i in range(len(model_edges)):
            self.g.addEdge(self.g.name_labels.index(model_edges[i]['start']), self.g.name_labels.index(
                model_edges[i]['end']), model_edges[i]['name'], 1)
            item = ()
            item += (model_edges[i]['start'],)
            item += (model_edges[i]['end'],)
            item += (1.0,)
            self.g.edge_list.append(item)
            del item

        self.g.build_edge_matrix()
        self.g.feature_calculate()

        # 定义节点特征向量x和标签y
        x = torch.tensor(self.g.feature_vector, dtype=torch.float)
        y = torch.tensor(self.g.labels, dtype=torch.float)
        # 定义边
        edge_index = torch.tensor(self.g.edge_matrix, dtype=torch.long)  # 终止点
        # 定义train_mask
        train_mask = torch.tensor(
            [(True if d is not None else False) for d in y])
        # 构建data
        self.data = Data(x=x, y=y, edge_index=edge_index,
                         train_mask=train_mask)
        self.data.num_classes = int(torch.max(self.data.y).item() + 1)
        self.nodes_num = len(self.g.name_labels)
        self.edges_num = len(self.g.edge_list)


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(520)
        self.num_features = num_features
        self.num_classes = num_classes
        self.conv1 = GATConv(self.num_features, 16)
        self.conv2 = GATConv(16, 32)
        self.conv3 = GATConv(32, self.num_classes)
        self.classifier = Linear(self.num_classes, self.num_classes)

    def forward(self, x, edge_index):
        # 3层GCN
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)
        h = h.relu()
        # 分类层
        out = self.classifier(h)
        return out, h


# ====================这边开始，写接口函数！！！====================


# 返回所有该用户提交过的产业链模型
def modellist(request):
    user_id = request.GET.get('user_id')
    model_list = list(Model.objects.filter(user_id=user_id).values(
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
                                 user_id=info['user_id'],
                                 update_method=info['update_method'])
    return JsonResponse({
        'result_code': 0,
        'result_msg': '提交模型成功',
        'model_id': model.model_id
    })


# 根据id更新产业链数据
def updatemodel(request):
    info = json.loads(request.body)
    model_id = info['model_id']
    model = Model.objects.get(model_id=model_id)  # 直接获取模型实例对象
    if info['model_nodes'] == model.model_nodes and info['model_edges'] == model.model_edges:
        return JsonResponse({
            'result_code': 1,
            'result_msg': '请先更新一下模型再提交哦~',
        })
    model.model_name = info['model_name']
    model.model_type = info['model_type']
    model.model_nodes = info['model_nodes']
    model.model_edges = info['model_edges']
    model.update_time = info['update_time']
    model.model_detail = info['model_detail']
    model.model_riskstatus = 0
    model.save()  # 调用save()方法将更改保存到数据库
    return JsonResponse({
        'result_code': 0,
        'result_msg': '修改模型成功',
    })


# 更新产业链拓扑结构图片
def updatemodelcover(request):
    info = json.loads(request.body)
    model_id = info['model_id']
    img_url = info['img_url']
    model = Model.objects.get(model_id=model_id)
    model.model_picture = img_url
    model.save()
    return JsonResponse({
        'result_code': 0,
        'result_msg': '更新模型封面图片成功',
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

            saved_file = FileToData()
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


# 将Base64编码的图片保存到本地
def save_picture(request):
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
                             'model_images', filename)
    default_storage.save(file_path, data)
    img_url = 'http://127.0.0.1/static/model_images/'+filename
    return JsonResponse({
        'result_code': 0,
        'result_msg': "保存图片成功",
        'img_url': img_url
    })


# 对指定的产业链模型进行完整性评估
def integrity(request):
    model_id = request.GET.get('model_id')
    model = Model.objects.filter(model_id=model_id).values(
        'model_id', 'model_nodes', 'model_edges', 'update_method')[0]

    model_nodes = json.loads(model['model_nodes'].replace("'", "\""))
    model_edges = json.loads(model['model_edges'].replace("'", "\""))
    # =====================测试代码===================== #
    data_source = DataSet(model_nodes=model_nodes, model_edges=model_edges)
    dataset = data_source.data
    # 加载模型
    parent_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..')
    file_path = os.path.join(
        parent_dir, 'static', 'GCN_models', 'gcn_model.pth')
    GCN_model = GCN(dataset.num_features, dataset.num_classes)
    GCN_model.load_state_dict(torch.load(file_path))
    # 测试函数

    def test(model, data):
        model.eval()
        with torch.no_grad():
            out, _ = model(data.x, data.edge_index)
            # 将out的形状从(batch_size, num_classes * heads)改为(batch_size, num_classes)
            out = out.view(-1, dataset.num_classes)
            pred = out.argmax(dim=1)
            correct = float(pred[data.train_mask].eq(
                data.y[data.train_mask]).sum().item())
            acc = correct / data.train_mask.sum().item()
        model.train()
        return acc
    # 进行测试
    test_acc = "{:.4f}".format(test(model=GCN_model, data=dataset) * 100)
    # 根据模型以及其所得分数来输出模型存在问题
    integrity_info = {
        'nodes_num': {
            'n': len(model_nodes['n']),
            'm1': len(model_nodes['m1']),
            'm2': len(model_nodes['m2']),
            'm3': len(model_nodes['m3']),
            'm4': len(model_nodes['m4']),
            'm5': len(model_nodes['m5']),
            'm6': len(model_nodes['m6']),
            'total': data_source.nodes_num
        },
        'edges_num': len(model_edges),
        'existed_questions': [],
        'solutions': []
    }
    label_list = ['n', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6']
    name_list = ['所属行业', '附属行业', '子行业', '涉及公司', '主营产品', '产品小类', '涉及材料']
    # 总体评价
    if test(model=GCN_model, data=dataset) * 100 >= 80:
        integrity_evaluation = "该模型的完整性表现很好，节点和节点之间的关系完善，是一套标准的产业链模型。"
    elif test(model=GCN_model, data=dataset) * 100 >= 60:
        integrity_evaluation = "该模型的完整性表现良好，存在着些许的缺陷，可以尝试根据提示的方案进行模型的改进。"
    elif test(model=GCN_model, data=dataset) * 100 >= 40:
        integrity_evaluation = "该模型的完整性表现可能存在着一定的问题，请根据提示的方案进行模型的改进。"
    elif test(model=GCN_model, data=dataset) * 100 >= 20:
        integrity_evaluation = "该模型存在着较大的产业链完整性缺陷，可能是手动输入模型的节点个数/边个数不足导致的。"
    else:
        integrity_evaluation = "该模型的完整性评估较低，请重新检查该模型的输入，调整后重新进行分析。"
    # 问题分析
    flag_list = [False, False, False, False]
    # 1.进行Model的结构分析
    for i in range(len(label_list) - 1):
        if len(model_nodes[label_list[i]]) > len(model_nodes[label_list[i + 1]]):
            integrity_info['existed_questions'].append(
                name_list[i + 1] + "的节点个数小于" + name_list[i] + "的个数")
            flag_list[0] = True
    for i in range(len(label_list)):
        if len(model_nodes[label_list[i]]) == 0:
            integrity_info['existed_questions'].append(
                '该模型中缺少"' + name_list[i] + '"类型的节点')
            flag_list[1] = True
    # 2.进行Model的节点和边个数的分析
    if len(model_edges) < data_source.nodes_num - 1:
        integrity_info['existed_questions'].append("该模型中存在有孤立点(无任何边)")
        flag_list[2] = True
    if data_source.nodes_num < 200:
        integrity_info['existed_questions'].append("该模型中包含的样本节点数量过少")
        flag_list[3] = True
    # 解决方案
    for i in range(len(flag_list)):
        if flag_list[i]:
            if i == 0:
                integrity_info['solutions'].append(
                    "可尝试重新调整模型，确保下一级的节点个数>=上一级的节点个数")
            elif i == 1:
                integrity_info['solutions'].append("可尝试重新调整模型，确保模型中没有空的类型节点")
            elif i == 2:
                integrity_info['solutions'].append("可尝试重新调整模型，将模型的各个节点边补充完整")
            else:
                integrity_info['solutions'].append(
                    "可尝试重新调整模型，多添加几个节点，以便于更准确的完整性分析")
    return JsonResponse({
        'result_code': 0,
        'result_msg': "完整性分析成功",
        'update_method': model['update_method'],
        'integrity_score': test_acc,
        'integrity_evaluation': integrity_evaluation,
        'integrity_info': integrity_info
    })


# 对指定的产业链模型进行风险评估
def riskanalyse(request):
    model_id = request.GET.get('model_id')
    model = Model.objects.get(model_id=model_id)

    if model.model_riskstatus == 0:
        def Markov(n):
            b = []
            for m in range(n):
                k = 1.0
                a = []
                for i in range(n):
                    if (i < n - 1):
                        j = random.uniform(0, k)
                        while (j < 0.1 or j > (1 / n)):
                            j = random.uniform(0, k)
                        a.append(j)
                        k = k - j
                    else:
                        a.append(k)
                b.append(a)
            matrix1 = np.matrix(b)
            c = []
            k = 1.0
            for m in range(n):
                if (m < n - 1):
                    l = random.uniform(0, k)
                    while (l < 0.1 or l > (1 / n)):
                        l = random.uniform(0, k)
                    c.append(j)
                    k = k - l
                else:
                    c.append(k)
            vector1 = np.matrix(c)
            for i in range(100):
                vector2 = vector1
                vector1 = vector1 * matrix1
                if ((vector2 == vector1).all()):
                    break
            li = vector1.tolist()
            ve = []
            for j in li[0]:
                mid1 = [j]
                mid2 = [1 - j]
                mid = [mid1, mid2]
                ve.append(mid)
            return ve

        sum1 = 0  # 初始化sum1
        ratedate = []  # 存每种风险的概率

        def fun(a, c, d, n):
            if (n >= m):
                b = {}
                for i in range(n):
                    b.update({c[i]: a[i]})
                prob_I = letter_infer.query(variables=[d], evidence=b)
                global sum1
                sum1 = sum1 + prob_I.values[1]
                return
            a[n] = 0
            fun(a, c, d, n + 1)
            a[n] = 1
            fun(a, c, d, n + 1)

        def solution(n):
            solut = []
            if n == 0:
                solut = ["资源风险:原料采购拓宽提供原料的渠道，适量存储一定的原材料",
                         "价格风险:关注原材料价格的变动，适时购买原材料",
                         "欺诈风险:在购买原材料时多调查卖家的情况，在接受订单时多考虑对方是否有能力接收这一订单",
                         "原料质量风险:在购买原材料时做到严格把控原材料的质量",
                         "库存原料风险:时刻保证库存原料可以满足正常生产",
                         "销售风险:根据产品市场价格和原料价格以及需求来决定生产的产品"]
            if n == 1:
                solut = ["库存积压:根据业务需求数量和产业链实际运营数量把控库存数量",
                         "坏账过高:采用先收款后发货，使用订货管理系统记录所有订单的收款状态",
                         "销售不当:做好财务预算，采购实施OTB管理模式",
                         "工程质量低劣:确保工程监管到位",
                         "竣工验收不规范:在工程项目验收中切实增加科技含量和定量分析",
                         "项目投资失控:预前进行可行性分析，结合现场实际状况做出决策",
                         "延迟或中断:及时止损，避免扩大沉没成本",
                         "外包范围、价格不合理:建立完善业务外包管理制度",
                         "商业贿赂:提高企业监管力度",
                         "承包费选择不当:根据各类业务和核心主业的关联度，合理确定承包费",
                         "监控不当:提高监管力度",
                         "未全面履行:对违约事项和违约责任进行详细的约定;债权人可以让债务人提供相应担保",
                         "合同纠纷处理不当:建立合同纠纷处理的有效机制，纠纷处理过程中处于举证有力",
                         "重大疏漏、欺诈:系统包含透明的指挥链、检查和平衡的网络以及明确概述的审计流程"]
            if n == 2:
                solut = ["产品、服务价格及供需变化:企业定价要从实现企业战略目标出发，选择恰当的定价目标，综合分析产品成本、市场需求、市场竞争等影响因素，运用科学的方法，灵活的策略，去制定顾客能够接受的价格",
                         "主要客户、供应商信用风险:认真建立供应商数据库,供应商数据库建立时要认真对供应商做好认真调查,通过调查来排除一些供应商,把合适的供应商纳入数据库",
                         "税收政策、利率、汇率、股票价格指数变化:实时跟踪，及时把控",
                         "潜在进入者、竞争者和替代品的竞争风险:提高企业竞争力"]
            if n == 3:
                solut = ["俄乌冲突:调整出口战略",
                         "管制与制裁:合作共同创建东亚产业链供应链，提高我国产业链的韧性和抗风险能力",
                         "碳边境税:对自身碳排放情况进行梳理，在充分了解碳排放边境调节机制的基础上，分析测算对企业的影响；结合国内和全球“双碳”目标以及企业自身情况，合理预计企业节碳减排空间，估算降低碳排放的具体成本",
                         "疫情:出入管理，企业防控",
                         "先进技术出口管制:加强产业链薄弱环节科技攻关，提升关键核心技术支持能力",
                         "环境保护与资源利用:优先采用可持续发展战略，协调绿色环保与产业链收益的平衡",
                         "美去中国化产业链转移风险:全方面加强国际交流合作，积极主动融入全球科技创新网络"]
            if n == 4:
                solut = ["资金运营风险:资金风险预警监测机制进行建立和完善",
                         "国家宏观经济政策:实时跟踪，及时把控",
                         "企业内部控制问题:建立科学完密的企业内部控制系统，及时改革",
                         "财政政策方向:追踪财政政策变化动向，及时根据时势变化企业财政方针",
                         "投资风险:投资前进行全面完整的投资分析",
                         "投资前进行全面完整的投资分析:建立良好的企业形象"]
            return solut

        smallrate4 = Markov(4)
        smallrate2 = Markov(2)
        smallrate3 = Markov(3)
        smallrate6 = Markov(6)
        smallrate7 = Markov(7)
        # 第一种
        letter_bn = BayesianNetwork(
            [('ZY', 'Y'), ('JG', 'Y'), ('QZ', 'Y'), ('YL', 'Y'), ('Y', 'S'), ('KC', 'Q'), ('XS', 'Q'), ('Q', 'S')])
        zy_cpd = TabularCPD(variable='ZY', variable_card=2,
                            values=smallrate4[0])
        jg_cpd = TabularCPD(variable='JG', variable_card=2,
                            values=smallrate4[1])
        qz_cpd = TabularCPD(variable='QZ', variable_card=2,
                            values=smallrate4[2])
        yl_cpd = TabularCPD(variable='YL', variable_card=2,
                            values=smallrate4[3])
        y_cpd = TabularCPD(variable='Y', variable_card=2,
                           values=[[0.2, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.9],
                                   [0.8, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.1]],
                           evidence=['ZY', 'JG', 'QZ', 'YL'], evidence_card=[2, 2, 2, 2])
        s_cpd = TabularCPD(variable='S', variable_card=2, values=[[0.2, 0.6, 0.6, 0.9], [0.8, 0.4, 0.4, 0.1]],
                           evidence=['Y', 'Q'], evidence_card=[2, 2])
        kc_cpd = TabularCPD(variable='KC', variable_card=2,
                            values=smallrate2[0])
        xs_cpd = TabularCPD(variable='XS', variable_card=2,
                            values=smallrate2[1])
        q_cpd = TabularCPD(variable='Q', variable_card=2, values=[[0.2, 0.6, 0.6, 0.9], [0.8, 0.4, 0.4, 0.1]],
                           evidence=['KC', 'XS'], evidence_card=[2, 2])
        letter_bn.add_cpds(zy_cpd, jg_cpd, qz_cpd, yl_cpd,
                           y_cpd, s_cpd, kc_cpd, xs_cpd, q_cpd)
        letter_bn.check_model()  # 检查构建的模型是否合理
        letter_bn.get_cpds()  # 网络中条件概率依赖关系
        letter_infer = VariableElimination(letter_bn)  # 变量消除
        rate1 = letter_infer.query(['S'])
        rate = round(rate1.values[1], 2)
        ratedate.append(rate)
        # 第二种
        sum1 = 0
        letter_xn = BayesianNetwork(
            [('A1', 'A'), ('A2', 'A'), ('A3', 'A'), ('B1', 'B'), ('B2', 'B'), ('B3', 'B'), ('B4', 'B'),
             ('C1', 'C'), ('C2', 'C'), ('C3', 'C'), ('D1',
                                                     'D'), ('D2', 'D'), ('D3', 'D'), ('D4', 'D'),
             ('A', 'X'), ('B', 'X'), ('C', 'X'), ('D', 'X')])
        a1_cpd = TabularCPD(variable='A1', variable_card=2,
                            values=smallrate3[0])
        a2_cpd = TabularCPD(variable='A2', variable_card=2,
                            values=smallrate3[1])
        a3_cpd = TabularCPD(variable='A3', variable_card=2,
                            values=smallrate3[2])
        a_cpd = TabularCPD(variable='A', variable_card=2,
                           values=[[0.2, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7, 0.9],
                                   [0.8, 0.6, 0.6, 0.6, 0.3, 0.3, 0.3, 0.1]],
                           evidence=['A1', 'A2', 'A3'], evidence_card=[2, 2, 2])
        b1_cpd = TabularCPD(variable='B1', variable_card=2,
                            values=smallrate4[0])
        b2_cpd = TabularCPD(variable='B2', variable_card=2,
                            values=smallrate4[1])
        b3_cpd = TabularCPD(variable='B3', variable_card=2,
                            values=smallrate4[2])
        b4_cpd = TabularCPD(variable='B4', variable_card=2,
                            values=smallrate4[3])
        b_cpd = TabularCPD(variable='B', variable_card=2,
                           values=[[0.2, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.9],
                                   [0.8, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.1]],
                           evidence=['B1', 'B2', 'B3', 'B4'], evidence_card=[2, 2, 2, 2])
        c1_cpd = TabularCPD(variable='C1', variable_card=2,
                            values=smallrate3[0])
        c2_cpd = TabularCPD(variable='C2', variable_card=2,
                            values=smallrate3[1])
        c3_cpd = TabularCPD(variable='C3', variable_card=2,
                            values=smallrate3[2])
        c_cpd = TabularCPD(variable='C', variable_card=2,
                           values=[[0.2, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7, 0.9],
                                   [0.8, 0.6, 0.6, 0.6, 0.3, 0.3, 0.3, 0.1]],
                           evidence=['C1', 'C2', 'C3'], evidence_card=[2, 2, 2])
        d1_cpd = TabularCPD(variable='D1', variable_card=2,
                            values=smallrate4[0])
        d2_cpd = TabularCPD(variable='D2', variable_card=2,
                            values=smallrate4[1])
        d3_cpd = TabularCPD(variable='D3', variable_card=2,
                            values=smallrate4[2])
        d4_cpd = TabularCPD(variable='D4', variable_card=2,
                            values=smallrate4[3])
        d_cpd = TabularCPD(variable='D', variable_card=2,
                           values=[[0.2, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.9],
                                   [0.8, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.1]],
                           evidence=['D1', 'D2', 'D3', 'D4'], evidence_card=[2, 2, 2, 2])
        x_cpd = TabularCPD(variable='X', variable_card=2,
                           values=[[0.2, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.9],
                                   [0.8, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.1]],
                           evidence=['A', 'B', 'C', 'D'], evidence_card=[2, 2, 2, 2])
        letter_xn.add_cpds(a1_cpd, a2_cpd, a3_cpd, a_cpd, b1_cpd, b2_cpd, b3_cpd, b4_cpd, b_cpd, c1_cpd, c2_cpd,
                           c3_cpd, c_cpd, d1_cpd, d2_cpd, d3_cpd, d4_cpd, d_cpd, x_cpd)
        letter_infer = VariableElimination(letter_xn)  # 变量消除
        rate1 = letter_infer.query(['X'])
        rate = round(rate1.values[1], 2)
        ratedate.append(rate)
        # 第三种
        sum1 = 0
        letter_en = BayesianNetwork(
            [('E1', 'E'), ('E2', 'E'), ('E3', 'E'), ('E4', 'E')])
        e1_cpd = TabularCPD(variable='E1', variable_card=2,
                            values=smallrate4[0])
        e2_cpd = TabularCPD(variable='E2', variable_card=2,
                            values=smallrate4[1])
        e3_cpd = TabularCPD(variable='E3', variable_card=2,
                            values=smallrate4[2])
        e4_cpd = TabularCPD(variable='E4', variable_card=2,
                            values=smallrate4[3])
        e_cpd = TabularCPD(variable='E', variable_card=2,
                           values=[[0.2, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.9],
                                   [0.8, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.1]],
                           evidence=['E1', 'E2', 'E3', 'E4'], evidence_card=[2, 2, 2, 2])
        letter_en.add_cpds(e1_cpd, e2_cpd, e3_cpd, e4_cpd, e_cpd)
        letter_en.check_model()  # 检查构建的模型是否合理
        letter_en.get_cpds()  # 网络中条件概率依赖关系
        letter_infer = VariableElimination(letter_en)  # 变量消除
        rate1 = letter_infer.query(['E'])
        rate = round(rate1.values[1], 2)
        ratedate.append(rate)

        # 第四种
        sum1 = 0
        letter_fn = BayesianNetwork(
            [('F1', 'F'), ('F2', 'F'), ('F3', 'F'), ('F4', 'F'), ('F5', 'F'), ('F6', 'F'), ('F7', 'F')])
        f1_cpd = TabularCPD(variable='F1', variable_card=2,
                            values=smallrate7[0])
        f2_cpd = TabularCPD(variable='F2', variable_card=2,
                            values=smallrate7[1])
        f3_cpd = TabularCPD(variable='F3', variable_card=2,
                            values=smallrate7[2])
        f4_cpd = TabularCPD(variable='F4', variable_card=2,
                            values=smallrate7[3])
        f5_cpd = TabularCPD(variable='F5', variable_card=2,
                            values=smallrate7[4])
        f6_cpd = TabularCPD(variable='F6', variable_card=2,
                            values=smallrate7[5])
        f7_cpd = TabularCPD(variable='F7', variable_card=2,
                            values=smallrate7[6])
        f_cpd = TabularCPD(variable='F', variable_card=2,
                           values=[
                               [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                                0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
                                0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                                0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.9],
                               [0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
                                0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                                0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
                                0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.1]],
                           evidence=['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7'], evidence_card=[2, 2, 2, 2, 2, 2, 2])
        letter_fn.add_cpds(f1_cpd, f2_cpd, f3_cpd, f4_cpd,
                           f5_cpd, f6_cpd, f7_cpd, f_cpd)
        letter_fn.check_model()  # 检查构建的模型是否合理
        letter_fn.get_cpds()  # 网络中条件概率依赖关系
        letter_infer = VariableElimination(letter_fn)  # 变量消除
        rate1 = letter_infer.query(['F'])
        rate = round(rate1.values[1], 2)
        ratedate.append(rate)

        # 第五种
        sum1 = 0
        letter_gn = BayesianNetwork(
            [('G1', 'G'), ('G2', 'G'), ('G3', 'G'), ('G4', 'G'), ('G5', 'G'), ('G6', 'G')])
        g1_cpd = TabularCPD(variable='G1', variable_card=2,
                            values=smallrate6[0])
        g2_cpd = TabularCPD(variable='G2', variable_card=2,
                            values=smallrate6[1])
        g3_cpd = TabularCPD(variable='G3', variable_card=2,
                            values=smallrate6[2])
        g4_cpd = TabularCPD(variable='G4', variable_card=2,
                            values=smallrate6[3])
        g5_cpd = TabularCPD(variable='G5', variable_card=2,
                            values=smallrate6[4])
        g6_cpd = TabularCPD(variable='G6', variable_card=2,
                            values=smallrate6[5])
        g_cpd = TabularCPD(variable='G', variable_card=2,
                           values=[
                               [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
                                0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7,
                                0.7, 0.7, 0.7, 0.7, 0.9],
                               [0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                                0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3,
                                0.3, 0.3, 0.3, 0.3, 0.1]],
                           evidence=['G1', 'G2', 'G3', 'G4', 'G5', 'G6'], evidence_card=[2, 2, 2, 2, 2, 2])
        letter_gn.add_cpds(g1_cpd, g2_cpd, g3_cpd,
                           g4_cpd, g5_cpd, g6_cpd, g_cpd)
        letter_gn.check_model()  # 检查构建的模型是否合理
        letter_gn.get_cpds()  # 网络中条件概率依赖关系
        letter_infer = VariableElimination(letter_gn)  # 变量消除
        rate1 = letter_infer.query(['G'])
        rate = round(rate1.values[1], 2)
        ratedate.append(rate)
        name = ["上游风险", "下游风险", "市场风险", "外部风险", "内部管理风险"]
        m = 0
        sum = 0
        max = 0
        for i in ratedate:
            sum = sum+i
            if i >= max:
                max = i
        scor = round(100 - 100 * (sum / 5), 2)
        for i in range(5):
            if ratedate[i] == max:
                m = i
        riskname = name[m]
        solut = solution(m)

        model.model_riskstatus = 1
        model.model_riskscore = scor
        model.model_riskmain = riskname
        model.model_riskmethods = json.dumps(solut)
        model.model_risklist = json.dumps(ratedate)
        model.save()  # 调用save()方法将更改保存到数据库
    else:
        scor = model.model_riskscore
        riskname = model.model_riskmain
        solut = json.loads(model.model_riskmethods)
        ratedate = json.loads(model.model_risklist)

    return JsonResponse({
        'result_code': 0,
        'result_msg': "风险评估成功",
        'risk_score': scor,
        'risk_main': riskname,
        'risk_method': solut,
        'risk_scorelist': ratedate,
    })
