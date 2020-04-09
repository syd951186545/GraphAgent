import os
from time import time

from tensorboardX import SummaryWriter

from Script.buildNetworkGraph import get_graph
from Environment.environment import Environment

Names = ["AthleteHomeStadium/", "AthletePlaysForTeam/", "AthletePlaysInLeague/", "AthletePlaysSport/",
         "OrganizationHeadQuarteredInCity/", "OrganizationHiredPerson/", "PersonBornInLocation/",
         "PersonLeadsOrganization/", "TeamPlaysSport/", "WorksFor/"]
# #
# Name = "AthleteHomeStadium/"
# #
for Name in Names:
    dataSet = "E:/AAAAA/ReforcementReasoning/RawDataSets/" + Name
    GRAPH = get_graph(dataSet)
    env = Environment(GRAPH)
    env.init_TreeList(dataSet + "train.txt")
    Resultdir = "E:/AAAAA/ReforcementReasoning/Result/"
    if not os.path.exists(Resultdir + Name): os.mkdir(Resultdir + Name)
    writer = SummaryWriter(Resultdir + Name)
    total = 0
    rewards = 0
    start = time()
    for eps in range(10000):
        node = env.reset()
        state, reward = env.step(node, None, "MCTS")
        rewards += reward
        total += 1
        posRate = rewards / total
        avgtime = (time() - start) / total
        if eps % 50 == 0:
            writer.add_scalar("MCTS_posRate", posRate, eps)
        if eps % 100 == 0:
            writer.add_scalar("avgtime", avgtime, eps)
