from tensorboardX import SummaryWriter

from Configuration import config
from Environment.environment_predict import Environment
from Script.buildNetworkGraph import get_graph
from Agent.DQN import DQN

GRAPH = get_graph()
env = Environment(GRAPH)
env.init_TreeList(config.dataSet + "/test.txt")
# writer = SummaryWriter(config.Summary_dir_test)
model = DQN(config.model_dir+"/act_net80.model")
rewards = 0
rewardx = 0
env.render()
for i, node in enumerate(env.TreeList):

    for j in range(32):
        state_seq, reward = env.step(node, PolicyChooseNet=model.choose_action, method=config.walk_method_predict)

        rewards += reward
        rewardx += 1 if rewards>0 else 0
        acc = rewardx / (i+1)
        # writer.add_scalar("acc/totalnum", acc, i + 1)
    print(acc)
