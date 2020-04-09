import json
from Configuration import config
import networkx as nx


def read_Emb_json(dataSet=None):
    if dataSet is None: dataSet = config.dataSet
    # 实体及关系编号
    with open(dataSet + "/vocab/relation_vocab.json") as relationFile:
        relationEmb = json.load(relationFile, encoding="utf-8")
    with open(dataSet + "/vocab/entity_vocab.json") as entityFile:
        entityEmb = json.load(entityFile, encoding="utf-8")
    return relationEmb, entityEmb


relationEmb, entityEmb = read_Emb_json()


def get_relation_num():
    return len(relationEmb)


def get_entity_num():
    return len(entityEmb)


def get_entity_dic():
    return entityEmb


def get_relation_dic():
    return relationEmb


# 编号反向对应实体及关系
#
#
# .......................

# 构建知识图谱网络（有向图）
def get_graph(dataSet=None):
    if dataSet is None:
        dataSet = config.dataSet
    graph = nx.DiGraph()
    relationDic, entityDic = read_Emb_json(dataSet)

    with open(dataSet + "/graph.txt") as graphFile:
        for line in graphFile:
            line = line.replace("\n", "")
            line = line.split("\t")
            if config.query != line[1]:
                graph.add_edge(entityDic[line[0]], entityDic[line[2]], relation=relationDic[line[1]])

    return graph
