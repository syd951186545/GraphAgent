import random
from Configuration import config
from Script.buildNetworkGraph import get_graph, get_relation_num
from Script.buildNetworkGraph import get_entity_dic, get_relation_dic
from Environment.MCTS_NSclass_predict import Node, State
from Environment.PolicyMCTS_predict import Policy_MCTS

entity_dic = get_entity_dic()
relation_dic = get_relation_dic()
GRAPH = get_graph()


class Environment:

    def __init__(self, Graph):
        self.reward_range = (0, 1)
        # 读取self.Graph = nx.read_gpickle("./data/DBLP_labeled.Graph")
        # 或者直接获取返回值buildNetworkGraph.get_graph()
        self.Graph = Graph

        # 初始化属性，current_node,
        self.current_node = None

        self.done = False
        # 所有的树
        self.TreeList = []

    def init_TreeList(self, test_file):
        """
        把测试文件提取成蒙特卡洛树根，存到树列表中去
        :return:
        """
        with open(test_file, "r") as df:
            for i, line in enumerate(df.readlines()):
                line = line.replace("\n", "")
                line = line.split("\t")
                if line[0] not in entity_dic or line[2] not in entity_dic or line[1] not in relation_dic:
                    continue  # 如果出现不在字典中的实体和关系则排除
                start_node = entity_dic[line[0]]
                target_node = entity_dic[line[2]]
                query = relation_dic[line[1]]
                if start_node not in GRAPH.nodes or target_node not in GRAPH.nodes:
                    i -= 1
                    continue
                state = State()
                state.set_current_node(start_node)
                state.set_current_neighbor(GRAPH[start_node])
                state.set_cumulative_choices([query, start_node])
                state.target_node = target_node

                root = Node()
                root.set_state(state)
                self.TreeList.append(root)

    def reset(self):
        # 随机取
        root = random.choice(self.TreeList)
        # 从头取，加到尾部
        # root = self.TreeList[0]
        return root

    def update(self, root):
        self.TreeList.append(root)

    def step(self, root, PolicyChooseNet=None, method="predict_MCTS"):
        if method == "Policy_MCTS":
            if PolicyChooseNet is None: raise Exception("please give PolicyNet as input or not use MCTS")
            node_t = root
            while not node_t.get_state().is_terminal():
                node_t,Q = Policy_MCTS(node_t, PolicyChooseNet)

            # 非None节点需要计算最后一次state——tup
            Qsa = Q if node_t.get_state().current_node == -1 else 0
            state_tup = node_t.get_state().state_tup
            reward = 1 if state_tup[-1] == -1 and state_tup[-2] == root.get_state().target_node else 0

            return state_tup, reward
        else:
            if method == "DQN_self":
                if PolicyChooseNet is None: raise Exception("please give PolicyNet as input or use random")
                state_tup = root.get_state().state_tup.copy()
                while len(state_tup) <= config.MAX_ROUND_NUMBER:
                    action_candidate, Qsa_values = PolicyChooseNet(state_tup)
                    action = action_candidate[Qsa_values.max(1)[1]]
                    if action != -1:
                        state_tup.append(action)
                    else:
                        state_tup.append(action)
                        break
                reward = 1 if state_tup[-1] == -1 and state_tup[-2] == root.get_state().target_node else 0
                return state_tup, reward
            return

    def render(self):
        print(self.path)

    def ComputeReward(self, action):
        return
