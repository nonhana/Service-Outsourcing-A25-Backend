import os
import uuid
import base64
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
    model = Model.objects.filter(model_id=model_id).values(
        'model_id', 'model_nodes', 'model_edges')[0]

    return JsonResponse({
        'result_code': 0,
        'result_msg': "风险评估成功",
        'model_data': model,
        'risk_item_list': [],
        'risk_main': "所会发生的主要风险",
        'risk_method': "规避风险的主要措施"
    })
