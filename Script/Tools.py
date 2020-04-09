import torch


def _give_index(tensor1D_list, index):
    """
    :param tensor1D_list:列tensor的列表[[tensor1d],[tensor1d],[tensor1d]]
    :param index:每一个向量的索引位置index组成的列表[2,1,3]
    :return:对应索引位置的元素组成一个新的tensor
    """
    result = []
    for i, tensor in enumerate(tensor1D_list):
        if i == 0:
            result = tensor[:, index[i]]
        else:
            result = torch.cat((result, tensor[:, index[i]]), 0)
    return result


def get_degrees(graph):
    degree = 0
    max_degree = 0
    for node in graph.node:
        a = graph.degree(node)
        degree += a
        if a > max_degree:
            max_degree = a
    degree /= len(graph.node)
    print("avg_degree={}".format(degree))
    print("max_degree={}".format(max_degree))
