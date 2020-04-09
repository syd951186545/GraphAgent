from tensorboardX import SummaryWriter

from Configuration import config
from Environment.environment_train import Environment
from Script.buildNetworkGraph import get_graph
from Agent.DQN import DQN

GRAPH = get_graph()
env = Environment(GRAPH)
env.init_TreeList(config.dataSet + "/test.txt")
# writer = SummaryWriter(config.Summary_dir_test)
model = DQN(None)
rewards = 0
for i, node in enumerate(env.TreeList):
    for j in range(100):
        state_seq, reward = env.step(node, PolicyChooseNet=model.choose_action, method=config.walk_method_predict)
        print(state_seq, reward)

    rewards += reward
    acc = rewards / (10*(i+1))
    # writer.add_scalar("acc/totalnum", acc, i + 1)
    print(acc)
