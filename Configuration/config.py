import os
import random
import sys
import numpy
import torch

root_dir = os.path.dirname(os.path.abspath('.'))
sys.path.append(root_dir)
# 模型随机性初始化
seed = 4869
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed(seed)  # gpu
numpy.random.seed(seed)  # numpy
random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn
# -------------------------------------------------------------------------
"""                         ALL DATA SET NAME YOU CAN CHOOSE            """
# -------------------------------------------------------------------------
_Names = ["AthleteHomeStadium/", "AthletePlaysForTeam/", "AthletePlaysInLeague/",
          "AthletePlaysSport/", "OrganizationHeadQuarteredInCity/",
          "OrganizationHiredPerson/", "PersonBornInLocation/",
          "PersonLeadsOrganization/", "TeamPlaysSport/", "WorksFor/", "WN18RR/"
          ]
# -------------------------------------------------------------------------
"""                         DATA SET and PROCESSING                     """
# -------------------------------------------------------------------------
"""Manually"""
Name = "AthletePlaysInLeague/"

entity_num = 75494  # 因为子图中总是不能完全包含全部的实体和关系，因此需要手动确认全部的实体和关系数量构建统一的编码总空间大小
relation_num = 404  # 实体和关系的数量请查阅/RawDataSets/"+Name+/vocab中的json字典
query = None  # "concept:athleteplaysforteam"  关系语句(如果需要会在制作GRAPH-net时被剔除，用以多跳推理任务)

entity_dim = 8  # graph-net中的编码方式random,node2vec,or transE 编码维度dim
relation_dim = 16
entity_embedding_method = "node2vec"
relation_embedding_method = "random"

"""Automatic"""
dataSet = root_dir + "/RawDataSets/" + Name
transed_dataSet = root_dir + "/PreDataSets/tranedFile/" + Name

embedding_dir = root_dir + "/PreDataSets/Embeddings/" + Name  # embedding 存放目录,以及存放文件命名（统一标准）
if not os.path.exists(embedding_dir): os.mkdir(embedding_dir)
entity_embedding_filename = root_dir + "/PreDataSets/Embeddings/" + Name + "/entity" + str(
    entity_dim) + "." + entity_embedding_method
relation_embedding_filename = root_dir + "/PreDataSets/Embeddings/" + Name + "/relation" + str(
    relation_dim) + "." + relation_embedding_method

# -------------------------------------------------------------------------
"""                                    TRAIN                             """
# -------------------------------------------------------------------------
"""Manually"""
num_episodes = 64 * 75441  # 训练回合数(总游走路径数)
pre_model = None  # 加载模型torch state_dic
# 经验池中，训练数据的获取方法（游走策略），1，蒙特卡洛树搜索“MCTS”2.DQN策略游走“DQN_self” 3.随机游走“random”4."Policy_MCTS"
walk_method = "Policy_MCTS"
MAX_ROUND_NUMBER = 5 - 1  # MCTS中每个初始节点最大的游走步数(+1),同时限制了TCN的输入序列长度，TCN输出层最后一个元素的感受野：2kd-2d-k+2,根据游走的最长路径调整TCN的层数
tcn_layers = [entity_dim, 2 * entity_dim, 2 * entity_dim, entity_dim]
# MCTS中每个给定节点的探索次数（其中优先探索未探索过的子节点，随后根据PUCT算法探索）;设置该值时，可以参考网络的度情况，Script/Tools中有相应工具
computation_budget = 32
# MCTS中PUCT算法的探索系数，C越大越偏向于广度搜索
C = 1.44
# --------------------------------------
""" DQN SET """
# --------------------------------------
"""Manually"""
capacity = 400
learning_rate = 1e-4
batch_size = 10
gamma = 0.99
decay = 0.999
net_replace = 400

# --------------------------------------
""" Model and visualization SET """
# --------------------------------------
"""Automatic"""
# 模型 存放路径
model_dir = root_dir + "/SavedModel/" + Name
# 模型结构图 存放路径
modelGraph_dir = root_dir + "/Model/modelGraph/"

Summary_dir = root_dir + "/Result/" + Name
Summary_dir_test = root_dir + "/Result/" + Name + "/summary_test/"

# -------------------------------------------------------------------------
"""                                    TEST                             """
# -------------------------------------------------------------------------
"""Manually"""
walk_method_predict = "Policy_MCTS"
computation_budget_predict = 32
predict_C = 1.44
