import os
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear


class DataSource:
    def __init__(self, filename):
        # 根节点
        self.n_node = []
        # 第一条边的相关属性
        self.r1_startnode = []
        self.r1_name = []
        self.r1_endnode = []
        # 第二节点
        self.m1_node = []
        # 第二条边的相关属性
        self.r2_startnode = []
        self.r2_name = []
        self.r2_endnode = []
        # 第三节点
        self.m2_node = []
        # 第三条边的相关属性
        self.r3_startnode = []
        self.r3_name = []
        self.r3_endnode = []
        # 第四节点==
        self.m3_node = []
        # 第四条边的相关属性
        self.r4_startnode = []
        self.r4_name = []
        self.r4_endnode = []
        # 第五节点(公司)
        self.m4_node = []
        # 第五条边的相关属性
        self.r5_startnode = []
        self.r5_name = []
        self.r5_endnode = []
        # 第六节点(产品小类)
        self.m5_node = []
        # 第六条边的相关属性
        self.r6_startnode = []
        self.r6_name = []
        self.r6_endnode = []
        # 第七节点
        self.m6_node = []
        # 节点
        self.node = []
        # 边
        self.edge = []

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
                if node_num == 1:
                    if line.strip() not in self.n_node:
                        self.n_node.append(line.strip())
                if node_num == 2:
                    if line.strip() not in self.m1_node:
                        self.m1_node.append(line.strip())
                if node_num == 3:
                    if line.strip() not in self.m2_node:
                        self.m2_node.append(line.strip())
                if node_num == 4:
                    if line.strip() not in self.m3_node:
                        self.m3_node.append(line.strip())
                if node_num == 5:
                    if line.strip() not in self.m4_node:
                        self.m4_node.append(line.strip())
                if node_num == 6:
                    if line.strip() not in self.m5_node:
                        self.m5_node.append(line.strip())
                if node_num == 7:
                    if line.strip() not in self.m6_node:
                        self.m6_node.append(line.strip())
                self.node.append(line.strip())
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

        for i in range(len(self.r1_startnode)):
            item = ()
            item += (self.r1_startnode[i],)
            item += (self.r1_endnode[i],)
            item += (1.0,)
            self.edge.append(item)
            del item
        for i in range(len(self.r2_startnode)):
            item = ()
            item += (self.r2_startnode[i],)
            item += (self.r2_endnode[i],)
            item += (1.0,)
            self.edge.append(item)
            del item
        for i in range(len(self.r3_startnode)):
            item = ()
            item += (self.r3_startnode[i],)
            item += (self.r3_endnode[i],)
            item += (1.0,)
            self.edge.append(item)
            del item
        for i in range(len(self.r4_startnode)):
            item = ()
            item += (self.r4_startnode[i],)
            item += (self.r4_endnode[i],)
            item += (1.0,)
            self.edge.append(item)
            del item
        for i in range(len(self.r5_startnode)):
            item = ()
            item += (self.r5_startnode[i],)
            item += (self.r5_endnode[i],)
            item += (1.0,)
            self.edge.append(item)
            del item
        for i in range(len(self.r6_startnode)):
            item = ()
            item += (self.r6_startnode[i],)
            item += (self.r6_endnode[i],)
            item += (1.0,)
            self.edge.append(item)
            del item


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
        # 新的图对象
        self.G = nx.DiGraph()

    # 增加顶点，具有属性：id，类型，名称
    def addVertex(self, key, type, name):
        self.position.append(self.numVertices)
        if name not in self.name_labels:
            self.name_labels.append(name)
        self.numVertices += 1
        self.visble.add_node(key)
        newVertex = Vertex(key, type, name)
        self.vertList[key] = newVertex
        return newVertex

    # 返回某个顶点的信息
    def getVertex(self, n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    # 判断顶点是否在邻接表中
    def __contains__(self, n):
        return n in self.vertList

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

    # 遍历邻接矩阵
    def printMatrix(self):
        for i in range(len(self.matrix)):
            print(self.matrix[i])

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
    def feature_calculate(self, filename):
        # 计算节点的中心度
        eigenvector = nx.eigenvector_centrality(self.visble)
        list = []
        for item in eigenvector:
            list.append(eigenvector[item])
        # 计算度中心度 紧密中心度 中介中心度
        handler = DataSource(filename)
        self.add_node_edge(handler.node, handler.edge)
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


# 画点函数
def visualize_embedding(h, color, epoch=None, loss=None):
    # figsize:生成图像大小
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch:{epoch},Loss:{loss.item():.4f}', fontsize=16)
    plt.show()


class DataSet:
    def __init__(self, filename):
        # 读取数据
        self.handler = DataSource(filename)
        # 添加顶点
        self.g = IndustryGraph()
        # 根节点
        self.g.addVertex(self.g.getVertices(), "industry",
                         self.handler.n_node[0])
        self.g.labels.append(1)
        # 一级产业
        for item in self.handler.m1_node:
            if item not in self.g.name_labels:
                self.g.addVertex(self.g.getVertices(), "industry", item)
                self.g.labels.append(2)
        # 二级产业
        for item in self.handler.m2_node:
            if item not in self.g.name_labels:
                self.g.addVertex(self.g.getVertices(), "industry", item)
                self.g.labels.append(3)
        # 公司
        for item in self.handler.m3_node:
            if item not in self.g.name_labels:
                self.g.addVertex(self.g.getVertices(), "company", item)
                self.g.labels.append(4)
        # 主营产品
        for item in self.handler.m4_node:
            if item not in self.g.name_labels:
                self.g.addVertex(self.g.getVertices(), "product", item)
                self.g.labels.append(5)
        # 产品小类
        for item in self.handler.m5_node:
            if item not in self.g.name_labels:
                self.g.addVertex(self.g.getVertices(), "littleproduct", item)
                self.g.labels.append(6)
        # 上游材料
        for item in self.handler.m6_node:
            if item not in self.g.name_labels:
                self.g.addVertex(self.g.getVertices(), "material", item)
                self.g.labels.append(7)

        # 初始化邻接矩阵
        self.g.initMatrix(self.g.getVertices())

        # 添加边
        # 一级产业-->根节点
        for i in range(len(self.handler.r1_endnode)):
            self.g.addEdge(self.g.name_labels.index(self.handler.r1_startnode[i]), self.g.name_labels.index(
                self.handler.r1_endnode[i]), self.handler.r1_name[i], 1)
        # 二级产业-->一级产业
        for i in range(len(self.handler.r2_endnode)):
            self.g.addEdge(self.g.name_labels.index(self.handler.r2_startnode[i]), self.g.name_labels.index(
                self.handler.r2_endnode[i]), self.handler.r2_name[i], 1)
        # 公司-->二级产业
        for i in range(len(self.handler.r3_endnode)):
            self.g.addEdge(self.g.name_labels.index(self.handler.r3_startnode[i]), self.g.name_labels.index(
                self.handler.r3_endnode[i]), self.handler.r3_name[i], 1)
        # 产品-->公司
        for i in range(len(self.handler.r4_endnode)):
            self.g.addEdge(self.g.name_labels.index(self.handler.r4_startnode[i]), self.g.name_labels.index(
                self.handler.r4_endnode[i]), self.handler.r4_name[i], 1)
        # 产品小类-->产品
        for i in range(len(self.handler.r5_endnode)):
            self.g.addEdge(self.g.name_labels.index(self.handler.r5_startnode[i]), self.g.name_labels.index(
                self.handler.r5_endnode[i]), self.handler.r5_name[i], 1)
        # 上游材料-->产品小类
        for i in range(len(self.handler.r6_endnode)):
            self.g.addEdge(self.g.name_labels.index(self.handler.r6_startnode[i]), self.g.name_labels.index(
                self.handler.r6_endnode[i]), self.handler.r6_name[i], 1)

        self.g.build_edge_matrix()
        self.g.feature_calculate(filename)

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


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(520)
        self.num_features = num_features
        self.num_classes = num_classes
        self.conv1 = GCNConv(self.num_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, self.num_classes)
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


# 测试函数
def test(model, data):
    model.eval()
    with torch.no_grad():
        out, _ = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = float(pred[data.train_mask].eq(
            data.y[data.train_mask]).sum().item())
        acc = correct / data.train_mask.sum().item()
    model.train()
    return acc


if __name__ == "__main__":
    # =====================测试代码===================== #
    parent_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..')
    file_path = os.path.join(
        parent_dir, 'model_file', 'data膜材料.txt')
    dataset = DataSet(file_path).data
    # 加载模型
    model = GCN(dataset.num_features, dataset.num_classes)
    model.load_state_dict(torch.load('gcn_model.pth'))
    # 进行测试
    test_acc = test(model=model, data=dataset)
    print(f"Test Accuracy: {test_acc:.4f}")
