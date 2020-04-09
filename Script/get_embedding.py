import json
import torch
from Script.buildNetworkGraph import get_relation_num
from Configuration import config


def get_emb_Tensor(emb_file, obj_num):
    with open(emb_file, "r") as f:
        head = f.readline().strip("\n").split(" ")
        print("There are {} object with {} dim in this Graph".format(head[0], head[1]))
        emb_Tensor = torch.zeros(obj_num, int(head[1]))
        for line in f.readlines():
            line = line.strip("\n").split("\t")
            if len(line) == 2:  # 防止空行
                line_id = int(line[0])
                line_emb = torch.FloatTensor(eval(line[1]))
                emb_Tensor[line_id] = line_emb
    return emb_Tensor


def get_entity_emb():
    return get_emb_Tensor(config.entity_embedding_filename,config.entity_num)


def get_relation_emb():
    return get_emb_Tensor(config.relation_embedding_filename,config.relation_num)


if __name__ == '__main__':
    get_entity_emb()
