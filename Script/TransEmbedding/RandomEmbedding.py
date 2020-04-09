import codecs
import math
import numpy as np
import os


class RandomEMB:
    def __init__(self, dataset, entity_dim=None, relation_dim=None):
        self.entity_dict = {}
        self.relation_dict = {}

        self.entity_dim = entity_dim if entity_dim else 16
        self.relation_dim = relation_dim if relation_dim else 64

        self.entity, self.relation = self.__data_loader(dataset)

    def __data_loader(self, file):
        entity2id = {}
        relation2id = {}
        file2 = file + "entity2id.txt"
        file3 = file + "relation2id.txt"

        with open(file2, 'r') as f1, open(file3, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            for line in lines1:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                entity2id[line[0]] = line[1]

            for line in lines2:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                relation2id[line[0]] = line[1]

        return entity2id, relation2id

    def embedding(self, out):
        if not os.path.exists(out): os.mkdir(out)

        for relation in self.relation:
            r_emb_temp = np.random.uniform(-6 / math.sqrt(self.relation_dim),
                                           6 / math.sqrt(self.relation_dim),
                                           self.relation_dim)
            self.relation_dict[relation] = r_emb_temp / np.linalg.norm(r_emb_temp, ord=2)

        for entity in self.entity:
            e_emb_temp = np.random.uniform(-6 / math.sqrt(self.entity_dim),
                                           6 / math.sqrt(self.entity_dim),
                                           self.entity_dim)
            self.entity_dict[entity] = e_emb_temp / np.linalg.norm(e_emb_temp, ord=2)

        with codecs.open(out + "entity"+str(self.entity_dim)+".random", "w") as f_e:
            f_e.write("{} {}\n".format(len(self.entity), self.entity_dim))
            for e in self.entity_dict.keys():
                f_e.write(self.entity[e] + "\t")
                f_e.write(str(list(self.entity_dict[e])))
                f_e.write("\n")
        with codecs.open(out + "relation"+str(self.relation_dim)+".random", "w") as f_r:
            f_r.write("{} {}\n".format(len(self.relation), self.relation_dim))
            for r in self.relation_dict.keys():
                f_r.write(self.relation[r] + "\t")
                f_r.write(str(list(self.relation_dict[r])))
                f_r.write("\n")


if __name__ == '__main__':
    Name = "AthletePlaysForTeam/"
    dataset = "E:/AAAAA/ReforcementReasoning/PreDataSets/tranedFile/" + Name
    Remb = RandomEMB(dataset)
    Remb.embedding("E:/AAAAA/ReforcementReasoning/PreDataSets/Embeddings/" + Name)
