import random

from Configuration import config
from Script.buildNetworkGraph import get_graph

GRAPH = get_graph()


class Node(object):
    def __init__(self):
        self.parent = None
        self.children = []
        self.visit_times = 0
        self.quality_value = 0
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def set_parent(self, parent):
        self.parent = parent

    def get_parent(self):
        return self.parent

    def set_children(self, children):
        self.children = children

    def get_children(self):
        return self.children

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_quality_value(self):
        return self.quality_value

    def set_quality_value(self, value):
        self.quality_value = value

    def quality_value_add_n(self, n):
        self.quality_value += n

    def is_all_expand(self):
        # 包含了终止动作 -1
        if self.parent is not None:
            if len(self.children) == len(self.state.candidate_actions):
                return True
            else:
                return False
        else:
            if len(self.children) == len(self.state.candidate_actions) + 1:
                return True
            else:
                return False

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)

    def __repr__(self):
        return "Node:{},Q/N:{}/{}".format(self.state.current_node, self.quality_value, self.visit_times)


class State(object):  # 某游戏的状态，例如模拟一个数相加等于1的游戏
    def __init__(self):
        self.current_node = None  # 当前状态，网络节点编号,-1表示终止节点，无实意
        self.candidate_actions = {}  # 当前状态，网络节点邻接节点（包含对应边）
        self.current_round_index = 0  # 第几轮网络节点选择
        self.target_node = None
        self.node_parent = None

        self.state_tup = []  # 选择过程记录,选择网络节点路径

    def is_terminal(self):  # 判断游戏是否结束
        if self.current_round_index == config.MAX_ROUND_NUMBER or self.current_node == -1:
            return True
        else:
            return False

    def compute_reward(self, PolicyChooseNet):
        # 模拟终止 局面时的得分，当选择终止节点时是目标节点则应该给+1 reward
        action_candidate, Qsa_values = PolicyChooseNet(self.state_tup)
        action = action_candidate[Qsa_values.max(1)[1]]

        return 1 if self.current_node == self.target_node else 0

    def set_current_node(self, value):
        self.current_node = value

    def set_node_parent(self, value):
        self.node_parent = value

    def set_current_neighbor(self, value):
        self.candidate_actions = value

    def set_current_round_index(self, round):
        self.current_round_index = round

    def set_cumulative_choices(self, choices):
        self.state_tup = choices

    def get_next_state_with_random_choice(self, parent):  # 得到下个状态
        actions = [keys for keys in self.candidate_actions.keys()]
        actions.append(-1)
        if parent in actions:
            actions.remove(parent)
        random_choice = random.choice(actions)
        if random_choice == -1:
            next_state = State()
            next_state.set_current_node(random_choice)
            next_state.set_current_round_index(self.current_round_index + 1)
            next_state.set_cumulative_choices(self.state_tup + [random_choice])
            next_state.set_node_parent(self.current_node)
            next_state.target_node = self.target_node
        else:
            next_state = State()
            next_state.set_current_node(random_choice)
            next_state.set_current_round_index(self.current_round_index + 1)
            next_state.set_current_neighbor(GRAPH[random_choice])
            next_state.set_cumulative_choices(self.state_tup + [random_choice])
            next_state.set_node_parent(self.current_node)
            next_state.target_node = self.target_node
        return next_state

    def get_next_state_with_random_choice_without_all_expended(self, tried_child, parent):
        actions = [keys for keys in self.candidate_actions.keys()]
        actions.append(-1)
        actions = list(set(actions).difference(set(tried_child)))
        if parent in actions:
            actions.remove(parent)

        random_choice = random.choice(actions)
        if random_choice == -1:
            next_state = State()
            next_state.set_current_node(random_choice)
            next_state.set_current_round_index(self.current_round_index + 1)
            next_state.set_cumulative_choices(self.state_tup + [random_choice])
            next_state.set_node_parent(self.current_node)
            next_state.target_node = self.target_node
        else:
            next_state = State()
            next_state.set_current_node(random_choice)
            next_state.set_current_round_index(self.current_round_index + 1)
            next_state.set_current_neighbor(GRAPH[random_choice])
            next_state.set_cumulative_choices(self.state_tup + [random_choice])
            next_state.set_node_parent(self.current_node)
            next_state.target_node = self.target_node
        return next_state

    def get_next_state_with_policy(self, parent, policy_net):
        action_candidate, Qsa_values = policy_net(self.state_tup)
        if parent in action_candidate:
            Qsa_values[0, action_candidate.index(parent)] = -999

        max_choice = action_candidate[Qsa_values.max(1)[1]]
        if max_choice != -1:
            next_state = State()
            next_state.set_current_node(max_choice)
            next_state.set_current_round_index(self.current_round_index + 1)
            next_state.set_current_neighbor(GRAPH[max_choice])
            next_state.set_cumulative_choices(self.state_tup + [max_choice])
            next_state.set_node_parent(self.current_node)
            next_state.target_node = self.target_node
        else:
            next_state = State()
            next_state.set_current_node(max_choice)
            next_state.set_current_round_index(self.current_round_index + 1)
            next_state.set_cumulative_choices(self.state_tup + [max_choice])
            next_state.set_node_parent(self.current_node)
            next_state.target_node = self.target_node
        return next_state
