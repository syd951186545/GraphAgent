import torch

from Script.buildNetworkGraph import get_graph
from Script.get_embedding import get_entity_emb, get_relation_emb
from Configuration import config

relation_dim = config.relation_dim
GRAPH = get_graph()
node_Tensor = get_entity_emb()
relation_Tensor = get_relation_emb()


def _transInput(state_tup):
    # inputx = [ns,n1,n2....nT],[2, 21550, 66968]
    nodes = []
    num_neighbors = []
    action_index = []
    action_nodes = []
    for i, node in enumerate(state_tup):
        neighbors = GRAPH[node]
        neinodes = []
        neirelation = []
        for nei, rdic in neighbors.items():
            neinodes.append(nei)
            neirelation.append(rdic["relation"])
        neighbors_matrix = node_Tensor[neinodes]
        neirelation = torch.LongTensor(neirelation)
        relations_matrix = relation_Tensor[neirelation]
        NodeEdgek = torch.cat((neighbors_matrix, relations_matrix), 1)
        if 0 == i:
            NE = NodeEdgek
        else:
            NE = torch.cat((NE, NodeEdgek), 0)
        a = neinodes.copy()
        a.insert(0, -1)
        # action_nodes在最后输出的动作空间使用，加了-1和action_index 不一致
        action_nodes.append(a)
        num_neighbors.append(len(neinodes))
        if i < len(state_tup) - 1:
            # action node index in NodeEdgek!!not in NE
            action_index.append(neinodes.index(state_tup[i + 1]))
        else:
            action_index.append(-1)

    return NE.view((-1, config.entity_dim + relation_dim)), num_neighbors, action_index, action_nodes


def transInput(state_tup):
    query_id = torch.LongTensor([state_tup[0]])
    state_seq = state_tup[1:]

    NEs, num_neighbors, action_index, action_nodes = _transInput(state_seq)
    nodes = node_Tensor[state_seq]
    # query 扩张到相同个数，与初始node拼接构成独特的query
    query = relation_Tensor[query_id]
    query = torch.cat((nodes[[0]], query), 1)
    query = query.expand(len(nodes), query.size()[1])
    return NEs, nodes, query, num_neighbors, action_index, action_nodes


if __name__ == '__main__':
    transInput([5, 2, 3])
