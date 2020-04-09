from myModel import myModel
from state_tup2input import transInput
from tensorboardX import SummaryWriter

state_tup = [4, 46, 2116, 86]
NEs, nodes, query, num_neighbors, action_index, action_nodes = transInput(state_tup)

model = myModel()
model.get_index(num_neighbors, action_index)
with SummaryWriter(comment='model') as w:
    w.add_graph(model, (NEs, nodes,query))
