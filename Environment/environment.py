import random
from Configuration import config
from Script.buildNetworkGraph import get_graph, get_relation_num
from Script.buildNetworkGraph import get_entity_dic, get_relation_dic
from Environment.MCTS_NSclass import Node, State
from Environment.PolicyMCTS import Policy_MCTS

entity_dic = get_entity_dic()
relation_dic = get_relation_dic()
GRAPH = get_graph()


class Environment:

    def __init__(self, Graph):
        self.reward_range = (0, 1)
        # 读取self.Graph = nx.read_gpickle("./data/DBLP_labeled.Graph")
        # 或者直接获取返回值buildNetworkGraph.get_graph()
        self.Graph = Graph
        self._render = False
        # 初始化属性，current_node,
        self.current_node = None

        self.done = False
        # 所有的树
        self.TreeList = []

    def init_TreeList(self, train_file):
        """
        把训练文件提取成蒙特卡洛树根，存到树列表中去
        :return:
        """
        with open(train_file, "r") as df:
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
        # 小数据量随机取
        root = random.choice(self.TreeList)
        # 大数据从头取，加到尾部，确保都训练到
        # root = self.TreeList[0]

        return root

    def update(self, root):
        self.TreeList.append(root)

    def step(self, root, PolicyChooseNet=None, method=config.walk_method):
        if method == "MCTS":
            self.TreeList.remove(root)
            if PolicyChooseNet is None: print(" PolicyNet is None,will run as UCB,or you can choose other method")
            node_t = root
            reward = node_t.state.compute_reward()
            state_tup = node_t.state.state_tup
            while node_t.state.current_node != root.state.target_node:
                node_t = Policy_MCTS(node_t, PolicyChooseNet)
                if node_t is not None:
                    reward = node_t.state.compute_reward()
                    state_tup = node_t.state.state_tup
                else:
                    break
            self.update(root)  # 更新root的q和 v的值
            if self._render: print(state_tup, reward)
            return state_tup, reward
        if method == "DQN_self":
            if PolicyChooseNet is None: raise Exception("please give PolicyNet as input or use random")
            state_tup = root.get_state().state_tup.copy()
            while len(state_tup) <= 11:
                action_candidate, Qsa_values = PolicyChooseNet(state_tup)
                action = action_candidate[Qsa_values.max(1)[1]]
                if action != -1:
                    state_tup.append(action)
                else:
                    break
            reward = 1 if state_tup[-1] == root.state.target_node else 0
            if self._render: print(state_tup, reward)
            return state_tup, reward
        if method == "random":
            state = root.state
            reward = 0
            while state.is_terminal():
                state = state.get_next_state_with_random_choice()
                reward = state.compute_reward
            if self._render: print(state.state_tup, reward)
            return state.state_tup, reward

    def render(self):
        self._render = True

    def ComputeReward(self, action):
        return
