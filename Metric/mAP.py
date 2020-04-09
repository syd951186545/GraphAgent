from sklearn.metrics import average_precision_score

from tensorboardX import SummaryWriter
from Configuration import config
from Environment.environment_predict import Environment
from Script.buildNetworkGraph import get_graph
from Agent.DQN import DQN
import numpy as np

GRAPH = get_graph()
env = Environment(GRAPH)
env.init_TreeList(config.dataSet + "/test.txt")
with open(config.entity_embedding_filename, "r") as f:
    head = f.readline().split("\t")
    num_of_nodes = int(head[0])

model = DQN("E:/AAAAA/ReforcementReasoning/Result/model/AthletePlaysForTeam/act_net9.model")
mAP = []

for i, node in enumerate(env.TreeList):
    y_score = np.zeros(num_of_nodes)
    for i in range(32):
        state_seq, reward = env.step(node, PolicyChooseNet=model.choose_action, method=config.walk_method_predict)
        print(state_seq, reward)
        if state_seq[-1] == -1:
            y_score[state_seq[-2]] += 1 / 32 * model.choose_action(state_seq[:-1])[1][0, 0]
        else:
            y_score[state_seq[-1]] += 1 / 32 * model.choose_action(state_seq)[1][0, 0]

    y_true = np.zeros(num_of_nodes)
    y_true[node.get_state().target_node] = 1
    aps = average_precision_score(y_true, y_score)
    mAP.append(aps)
    print(np.mean(mAP))
    print(np.std(mAP, ddof=1))
print(np.mean(mAP))
print(np.std(mAP, ddof=1))
