import math
import sys
import random

from Configuration import config
from Environment.MCTS_NSclass_train import Node,State


def PUCB(node, Policy_net, is_exploration, output=False):  # 若子节点都扩展完了，求UCB值最大的子节点
    best_score = -sys.maxsize
    best_sub_node = None
    # stop = False
    action_candidate, Qsa_values = Policy_net(node.get_state().state_tup)

    for sub_node in node.get_children():
        if is_exploration:
            # C = 1 / math.sqrt(2.0)  # C越大越偏向于广度搜索，越小越偏向于深度搜索
            C = config.C
        else:
            C = 0.0

        # PUCB:
        left = sub_node.get_quality_value() / sub_node.get_visit_times()
        # if node.get_visit_times() < 1:
        #     print(node)
        #     node.set_visit_times(1)
        right = math.log(node.get_visit_times()) / sub_node.get_visit_times()

        Qsa = Qsa_values.squeeze(0)[action_candidate.index(sub_node.state.current_node)]
        score = left + math.sqrt(C * math.sqrt(Qsa) * right)

        # UCB
        # left = sub_node.get_quality_value() / sub_node.get_visit_times()
        # right = math.sqrt(math.log(node.get_visit_times())) / sub_node.get_visit_times()
        # score = left + C * math.sqrt(right)

        # UCB--
        # left = sub_node.get_quality_value() / sub_node.get_visit_times()
        # score = left

        if score > best_score:
            best_score = score
            best_sub_node = sub_node
        if score == best_score:
            best_sub_node = random.choice([best_sub_node, sub_node])
    # 任何节点必然存在一个-1状态，除非当前路径因为长度max没有子节点
    if not node.get_children():
        Qsa = Qsa_values[0, 0]

        best_sub_node = Node()

        current_state = node.state
        stop_state = State()
        stop_state.set_current_node(-1)
        stop_state.set_current_round_index(current_state.current_round_index + 1)
        stop_state.set_cumulative_choices(current_state.state_tup + [-1])
        stop_state.set_node_parent(current_state.current_node)
        stop_state.target_node = current_state.target_node
        best_sub_node.set_state(stop_state)
        node.add_child(best_sub_node)
    return best_sub_node, Qsa


def expand_child(node):  # 得到未扩展的子节点
    tried_sub_node_id = [sub_node.get_state().current_node for sub_node in node.get_children()]
    # 在有限的扩展次数中可以考虑按照Q值扩展，当扩展足够多时子节点全能取到则可以快速随机扩展
    new_state = node.get_state().get_next_state_with_random_choice_without_all_expended(tried_sub_node_id, node.state.node_parent)
    # while new_state in tried_sub_node_states:  # 可能造成无限循环
    #     new_state = node.get_state().get_next_state_with_random_choice_without_all_expended()
    sub_node = Node()
    sub_node.set_state(new_state)
    node.add_child(sub_node)
    return sub_node


def select_node_to_simulate(node, Policy_net):  # 选择子节点的策略
    """
    把所有的子节点都加上当前节点的child
    选择最优节点（UCB策略选择）
    :param Policy_net:
    :param node:
    :return:
    """

    while not node.get_state().is_terminal():
        if node.is_all_expand():
            node, _ = PUCB(node, Policy_net, True)
        else:
            sub_node = expand_child(node)
            return sub_node
    return node


def simulate(node, Policy_net):
    """
    模拟该节点，获得最终可能回报，
    终止条件： 1.到达目标节点 2.仿真10步长（状态自身计数） 3.没有可选动作
    仿真策略：Policy_net计算价值
    仿真回报：到达目标节点+1， 否则为0
    :param node:
    :return:
    """
    current_state = node.get_state()
    while not current_state.is_terminal():
        current_state = current_state.get_next_state_with_random_choice(current_state.node_parent)

    if current_state.current_node == -1:

        return 1 if current_state.node_parent == current_state.target_node else 0
    else:
        return 0


def backup(node, reward):
    while node is not None:
        node.visit_times_add_one()
        node.quality_value_add_n(reward)
        node = node.parent


def MCTS_main(node):  # 蒙特卡洛树搜索总函数
    for i in range(config.computation_budget):
        expend_node = select_node_to_simulate(node)
        reward = simulate(expend_node)
        backup(expend_node, reward)
    best_next_node = PUCB(node, True)
    return best_next_node


def Policy_MCTS(node, Policy_net):
    for i in range(config.computation_budget):
        expend_node = select_node_to_simulate(node, Policy_net)
        reward = simulate(expend_node, Policy_net)
        backup(expend_node, reward)
    best_next_node, Q = PUCB(node, Policy_net, True, True)
    return best_next_node, Q
