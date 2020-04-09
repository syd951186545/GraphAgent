from Script import buildNetworkGraph
from Script.node2vec import Node2Vec
from Script.TransEmbedding.RandomEmbedding import RandomEMB
from Script.TransEmbedding.transFileData import transFile
from Script.TransEmbedding.transE.transE import data_loader, TransE
from Configuration import config

entity_num = buildNetworkGraph.get_entity_num()
relation_num = buildNetworkGraph.get_relation_num()
entity_dic = buildNetworkGraph.get_entity_dic()
relation_dic = buildNetworkGraph.get_relation_dic()

GRAPH = buildNetworkGraph.get_graph()


def node_embedding():
    if "node2vec" == config.entity_embedding_method:
        node2vec = Node2Vec(GRAPH, dimensions=config.entity_dim, walk_length=8, num_walks=20, workers=4)
        model = node2vec.fit(window=4, min_count=1, batch_words=4, workers=4)
        # Look for most similar nodes
        model.wv.most_similar('2')  # Output node names are always strings
        # Save embeddings for later use
        model.wv.save_word2vec_format(config.entity_embedding_filename)
        # Save model for later use
        # model.save(config.node_model_filename)


def edge_embedding():
    if "random" == config.relation_embedding_method:
        transFile(config.dataSet, config.transed_dataSet)
        Remb = RandomEMB(config.transed_dataSet, config.entity_dim, config.relation_dim)
        Remb.embedding(config.embedding_dir)


def transEncode():
    if "transE" == config.encode_method:
        transFile(config.dataSet, config.transed_dataSet)
        entity_set, relation_set, triple_list = data_loader(config.transed_dataSet)
        transE = TransE(entity_set, relation_set, triple_list, embedding_dim=64, learning_rate=0.01, margin=1, L1=True)
        transE.emb_initialize()
        transE.train(epochs=100)


if __name__ == '__main__':
    node_embedding()
    # edge_embedding()
