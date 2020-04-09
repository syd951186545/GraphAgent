import math
import sys
import random

from Configuration import config
from Environment.MCTS_NSclass import Node


def PUCB(node, Policy_net, is_exploration):  # 若子节点都扩展完了，求UCB值最大的子节点
    best_score = -sys.maxsize
    best_sub_node = None
    if Policy_net:
        action_candidate, Qsa_values = Policy_net(node.get_state().state_tup)
    for sub_node in node.get_children():
        if is_exploration:
            # C = 1 / math.sqrt(2.0)  # C越大越偏向于广度搜索，越小越偏向于深度搜索
            C = config.C
        else:
            C = 0.0

        # PUCT:
        left = sub_node.get_quality_value() / sub_node.get_visit_times()
        right = math.sqrt(math.log(node.get_visit_times())) / sub_node.get_visit_times()
        if Policy_net:
            Qsa = Qsa_values.squeeze(0)[action_candidate.index(sub_node.state.current_node)]
        else:
            Qsa = 1
        score = left + C * Qsa * right

        # UCT:
        # left = sub_node.get_quality_value() / sub_node.get_visit_times()
        # right = math.log(node.get_visit_times()) / sub_node.get_visit_times()
        # score = left + C * math.sqrt(right)

        # UCT--
        # left = sub_node.get_quality_value() / sub_node.get_visit_times()
        # right = math.sqrt(node.get_visit_times()) / sub_node.get_visit_times()
        # score = left + C * right

        if score > best_score:
            best_score = score
            best_sub_node = sub_node
        if score == best_score:
            best_sub_node = random.choice([best_sub_node, sub_node])
    return best_sub_node


def expand_child(node):  # 得到未扩展的子节点
    tried_sub_node_id = [sub_node.get_state().current_node for sub_node in node.get_children()]
    new_state = node.get_state().get_next_state_with_random_choice_without_all_expended(tried_sub_node_id,
                                                                                        node.state.node_parent)
    # while new_state in tried_sub_node_states:  # 可能造成无限循环
    #     new_state = node.get_state().get_next_state_with_random_choice_without_all_expended()
    sub_node = Node()
    sub_node.set_state(new_state)
    node.add_child(sub_node)
    return sub_node


def select_node_to_simulate(node, Policy_net):  # 选择子节点的策略
    """
    选择扩展节点的策略，如果当前节点的有子节点未被扩展过，则选择一个扩展。
    若全部扩展过，就选择最优节点（PUCT策略选择）
    :param Policy_net:
    :param node:
    :return:
    """
    while not node.get_state().is_terminal():
        if node.is_all_expand():
            node = PUCB(node, Policy_net, True)
        else:
            sub_node = expand_child(node)
            return sub_node
    return node


def simulate(node):
    """
    模拟该节点，获得最终可能回报，
    终止条件： 1.到达目标节点 2.仿真10步长（状态自身计数） 3.没有可选动作
    仿真策略：采用快速随机走子，随机选择下一动作（节点），但排除父节点
    仿真回报：到达目标节点+1， 否则为0
    :param node:
    :return:
    """
    current_state = node.get_state()

    while not current_state.is_terminal():
        current_state = current_state.get_next_state_with_random_choice(current_state.node_parent)

    final_state_reward = current_state.compute_reward()
    return final_state_reward


def backup(node, reward):
    while node is not None:
        node.visit_times_add_one()
        node.quality_value_add_n(reward)
        node = node.parent


def MCTS_main(node):  # 蒙特卡洛树搜索总函数
    computation_budget = 1000  # 模拟的最大次数
    for i in range(computation_budget):
        expend_node = select_node_to_simulate(node)
        reward = simulate(expend_node)
        backup(expend_node, reward)
    best_next_node = PUCB(node, True)
    return best_next_node


def Policy_MCTS(node, Policy_net):
    for i in range(config.computation_budget):
        expend_node = select_node_to_simulate(node, Policy_net)
        reward = simulate(expend_node)
        backup(expend_node, reward)
    best_next_node = PUCB(node, Policy_net, True)
    return best_next_node
